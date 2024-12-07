# Copyright 2023 Kévin Garanger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import re
from pathlib import Path

import flax
import jax
import optax
import orbax
from core import (ActivationType, ArchitectureType, DataScaler, DataScalerType,
                  DataScalingMethodType, DataScalingParams, Dataset,
                  DatasetSampler, DatasetSamplerType, EmptyDataset,
                  FeatureType, InitializerType, IORotator, LossType,
                  MetricType, ModelData, NetworkType, OptimizerType,
                  SchedulerType, SequenceDataset, Trainer, TrainState,
                  ValueAndGradDataset)
from jax import numpy as jnp
from TFENN.util.array_util import canonicalize_tuple
from util import FileIO, list_to_args

from .cmd_args import CmdArgs


def parse_args(args: argparse.Namespace | None = None):
    """Parse the command line arguments
    :param args: The command line arguments to parse. If None, the command line
    arguments will be parsed.
    :type args: argparse.Namespace, optional
    :return: The parsed command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser = CmdArgs.add_config_args(parser)
    parser = CmdArgs.add_model_args(parser)
    parser = CmdArgs.add_dataset_args(parser)
    parser = CmdArgs.add_training_args(parser)
    parser = CmdArgs.add_save_args(parser)
    parser = CmdArgs.add_logging_args(parser)

    args = parser.parse_args(args)

    args.input_shape = canonicalize_tuple(args.input_shape)
    args.output_shape = canonicalize_tuple(args.output_shape)

    args.optimizer_options = list_to_args(args.optimizer_options)
    args.optimizer_options = (
        map(float, args.optimizer_options[0]),
        {k: float(v) for k, v in args.optimizer_options[1].items()},
    )

    args.scheduler_options = list_to_args(args.scheduler_options)
    args.scheduler_options = (
        map(float, args.scheduler_options[0]),
        {k: float(v) for k, v in args.scheduler_options[1].items()},
    )

    if args.dataset_split is not None and args.val_dataset_files is not None:
        raise ValueError("dataset_split and val_dataset_files are mutually exclusive")

    args.dataset_files = sum(
        map(
            lambda p: [p] if p.is_absolute() else list(Path(".").glob(str(p))),
            args.dataset_files,
        ),
        [],
    )
    if args.val_dataset_files is not None:
        args.val_dataset_files = sum(
            map(
                lambda p: [p] if p.is_absolute() else list(Path(".").glob(str(p))),
                args.val_dataset_files,
            ),
            [],
        )

    return args


def create_print_metrics(
    args: argparse.Namespace, use_val: bool = False
) -> set[MetricType]:
    """Create the metrics to print.
    :param args: The command line arguments.
    :type args: argparse.Namespace
    :param use_val: Whether to use the validation metrics, defaults to False
    :type use_val: bool, optional
    :return: The metrics to print.
    :rtype: set[MetricType]
    """
    return (
        {
            MetricType.EPOCH,
            MetricType.TIME,
            MetricType.TRAINING_LOSS,
        }
        | set(args.print_metrics)
        | (
            {MetricType.VALIDATION_LOSS, MetricType.TRAINING_LIKE_VALIDATION_LOSS}
            if use_val
            else set()
        )
    )


def create_datasets(
    args: argparse.Namespace, key: jax.random.PRNGKey, sequential: bool = False
) -> dict[str, Dataset | None]:
    """Create the training and validation datasets.
    :param args: The command line arguments.
    :type args: argparse.Namespace
    :param key: The random key to use.
    :type key: jax.random.PRNGKey
    :param sequential: Whether to use a sequential dataset, defaults to False
    :type sequential: bool, optional
    :return: The training and validation datasets.
    :rtype: dict[str, Dataset | None]
    """
    dataset_cls = SequenceDataset if sequential else Dataset
    dataset = dataset_cls.concatenate(
        [
            (
                dataset_cls(
                    file_path=dataset_file,
                    input_shape=args.input_shape,
                    output_shape=args.output_shape,
                    input_cols=args.file_in_cols,
                    output_cols=args.file_out_cols,
                    sequence_length=args.sequence_length,
                )
                if (k != 1 or dataset_cls()._logger.setLevel(logging.WARNING) or True)
                else None
            )
            for k, dataset_file in enumerate(args.dataset_files)
        ]
    )
    if len(dataset) == 0:
        raise ValueError("Empty dataset")

    dataset_cls()._logger.setLevel(logging.INFO)
    if isinstance(dataset, SequenceDataset) and args.sequence_length is None:
        args.sequence_length = dataset[0][0].shape[0]
        logging.info(f"Setting sequence length to {args.sequence_length}")

    logging.info(f"Dataset size: {len(dataset)}")

    if args.val_dataset_files is not None:
        training_dataset = dataset
        validation_dataset = dataset_cls.concatenate(
            [
                dataset_cls(
                    file_path=val_dataset_file,
                    input_shape=args.input_shape,
                    output_shape=args.output_shape,
                    input_cols=args.file_in_cols,
                    output_cols=args.file_out_cols,
                )
                for val_dataset_file in args.val_dataset_files
            ]
        )

    elif args.dataset_split is not None and args.dataset_split < 1.0:
        training_dataset, validation_dataset = dataset.split(args.dataset_split, key)
    else:
        training_dataset = dataset
        validation_dataset = None

    if args.max_dataset_size:
        training_dataset.clip(args.max_dataset_size)

    if validation_dataset is None:
        logging.info(
            f"Training dataset size: {len(training_dataset)}\tNo validation dataset"
        )
        return {"train": training_dataset}
    else:
        logging.info(
            f"Training dataset size: {len(training_dataset)}\t"
            f"Validation dataset size: {len(validation_dataset)}"
        )
        return {"train": training_dataset, "val": validation_dataset}


def create_baseline_scaler(dataset: Dataset) -> tuple[DataScaler, DataScalingParams]:
    """Create a baseline scaler the shifts and scales the data globally by constant
    factors.
    :param dataset: The dataset to create the scaler from
    :type dataset: Dataset
    :return: The baseline scaler and the scaling parameters
    :rtype: tuple[DataScaler, tuple[DataScalingParams, DataScalingParams]]
    """

    scaler = DataScalerType.SCALAR.create(
        DataScalingMethodType.NORMAL,
        feature_axes=(),
    )
    scaler_params = dataset.fit_scalers(scaler)

    logging.info(
        "Baseline scalers:\t"
        f"inputs={scaler_params[0]}\t"
        f"outputs={scaler_params[1]}"
    )

    return scaler, scaler_params


def create_training_scaler(
    args: argparse.Namespace, dataset: Dataset
) -> tuple[DataScaler, DataScalingParams]:
    """Create a training scaler that shifts and scales the data globally by constant
    factors.
    :param args: The command line arguments
    :type args: argparse.Namespace
    :param dataset: The dataset to create the scaler from
    :type dataset: Dataset
    :return: The training scaler and the scaling parameters
    :rtype: tuple[DataScaler, tuple[DataScalingParams, DataScalingParams]]
    """

    scaled_axes = []

    if args.scale_per_feature:
        if args.data_type == FeatureType.TENSOR:
            scaled_axes += list(
                range(
                    -len(args.input_shape),
                    -len(
                        args.data_notation.create(
                            dim=args.data_tensor_dim, order=2
                        ).reduced_shape
                    ),
                )
            )
        elif args.data_type == FeatureType.SCALAR:
            scaled_axes += list(range(-len(args.input_shape), 0))
        else:
            raise ValueError(f"Invalid data type {args.data_type}")
    if args.sequence_length and args.scale_per_step:
        scaled_axes.append(-len(args.input_shape) - 1)

    logging.info(
        "Scaling training dataset of input and output shapes"
        f" {dataset.input.shape} and {dataset.output.shape}"
        f" along axes {scaled_axes}"
    )

    scaler = DataScalerType(args.data_type).create(
        args.dataset_scaling,
        feature_axes=scaled_axes,
        **(
            {
                "data_notation": args.data_notation.create(
                    dim=args.data_tensor_dim, order=2
                )
            }
            if args.data_type == FeatureType.TENSOR
            else {}
        ),
    )
    scaler_params = dataset.fit_scalers(scaler)

    logging.info(
        "Training dataset scalers:\t"
        f"inputs={scaler_params[0]}\t"
        f"outputs={scaler_params[1]}"
    )

    return scaler, scaler_params


def create_dataset_samplers(
    args: argparse.Namespace,
    datasets: dict[str, Dataset | None],
    key: jax.random.PRNGKey,
) -> dict[str, DatasetSampler | None]:
    """Create the training and validation dataset samplers.
    :param args: The command line arguments
    :type args: dict[str, Dataset | None]
    :param datasets: The datasets to create the samplers from
    :type datasets: dict[str, Dataset | None]
    :param key: The random number generator key
    :type key: jax.random.PRNGKey
    :return: The training and validation dataset samplers
    :rtype: dict[str, DatasetSampler | None]
    """

    dataset_samplers = {
        "train": args.dataset_sampler.create(
            dataset=datasets["train"],
            batch_size=args.batch_size,
            key=key,
        )
    }
    if "val" in datasets:
        dataset_samplers["val"] = DatasetSamplerType.WHOLE_DATASET.create(
            dataset=datasets["val"]
        )

    return dataset_samplers


def create_train_state(
    args: argparse.Namespace,
    params_key: jax.random.PRNGKey,
    total_steps: int | None = None,
) -> TrainState:
    """Create the training state.
    :param args: The command line arguments
    :type args: argparse.Namespace
    :param params_key: The random number generator key for the parameters
    :type params_key: jax.random.PRNGKey
    :param total_steps: The total number of training steps, defaults to None
    :type total_steps: int, optional
    :return: The training state
    :rtype: TrainState
    """

    in_dims = sum([d > 1 for d in args.input_shape])

    model_params = {
        "layer_count": args.layer_count,
        "layer_size": args.layer_size,
        "activation": args.activation,
        "output_activation": args.output_activation,
        "kernel_initializer": args.kernel_initializer,
        "bias_initializer": args.bias_initializer,
    }

    if args.feature_type == FeatureType.SCALAR:
        if len(args.output_shape) != 1:
            raise ValueError(
                "Output shape must be a 1D array when using scalar features"
            )
        model_params = model_params | {
            "output_size": args.output_shape[0],
        }
    elif args.feature_type == FeatureType.TENSOR:
        if args.data_type != FeatureType.TENSOR:
            raise ValueError(
                f"Data type {args.data_type} not supported for tensor inputs"
            )
        data_notation = args.data_notation
        input_size = jnp.prod(jax.numpy.array(args.input_shape))
        output_size = jnp.prod(jax.numpy.array(args.output_shape))
        data_notation_size = jnp.prod(
            jnp.array(
                args.data_notation.create(
                    dim=args.data_tensor_dim, order=2
                ).reduced_shape
            )
        )
        if input_size % data_notation_size:
            raise ValueError(
                f"Input shape {args.input_shape} not compatible with data notation {data_notation}"
            )
        if output_size % data_notation_size:
            raise ValueError(
                f"Output shape {args.output_shape} not compatible with data notation {data_notation}"
            )
        else:
            output_cnt = int(output_size // data_notation_size)
            logging.info(
                f"Using data notation {data_notation} with {output_cnt} outputs"
            )

        model_params = model_params | {
            "dim": args.data_tensor_dim,
            "input_notation_type": data_notation,
            "output_notation_type": data_notation,
            "sym_cls_type": args.sym_cls,
            "output_size": output_cnt,
        }
    else:
        raise ValueError(f"Feature type {args.feature_type} not supported")

    model = NetworkType((args.architecture_type, args.feature_type)).create(
        **model_params
    )

    model_data = ModelData(
        NetworkType((args.architecture_type, args.feature_type)).create,
        kwargs=model_params,
    )

    if args.use_io_rotator:
        model = IORotator(model)
        model_data = ModelData(
            IORotator,
            args=(model_data,),
        )

    inputs_shape = (
        (args.batch_size, args.sequence_length) + args.input_shape
        if args.sequence_length
        else (args.batch_size,) + args.input_shape
    )
    train_state_params = model.init(
        {"params": params_key},
        jnp.ones(inputs_shape, dtype=float),
        train=False,
    )

    params_types = set(
        map(
            lambda x: x.dtype,
            flax.traverse_util.ModelParamTraversal(
                filter_fn=lambda x, _: "kernel" in x or "bias" in x
            ).iterate(train_state_params),
        )
    )

    logging.info(f"Model parameters dtype: {params_types}")

    logging.info(
        "Model parameters count: "
        f"{sum(map(jnp.size, flax.traverse_util.TraverseTree().iterate(train_state_params['params'])))}"
    )

    if args.scheduler == SchedulerType.COC:
        args.scheduler_options[1]["transition_steps"] = total_steps

    scheduler = SchedulerType.create(
        args.scheduler, *args.scheduler_options[0], **args.scheduler_options[1]
    )

    tx = OptimizerType.create(
        args.optimizer,
        *args.optimizer_options[0],
        learning_rate=scheduler,
        **args.optimizer_options[1],
    )

    if args.gradient_clip is not None:
        tx = optax.chain(tx, optax.clip_by_block_rms(args.gradient_clip))
        logging.info(
            f"Using gradient clipping with RMS block norm {args.gradient_clip}"
        )

    if args.freeze_layers:
        p = re.compile("_[0-9]+$")

        def freeze_layer(path: tuple[str, ...]) -> bool:
            for w in path:
                res = p.search(w)
                if res:
                    return int(res.group(0)[1:]) < args.freeze_layers
            return False

        partition_optimizers = {"trainable": tx, "frozen": optax.set_to_zero()}
        param_partition = flax.core.freeze(
            flax.traverse_util.path_aware_map(
                lambda path, _: "frozen" if freeze_layer(path) else "trainable",
                train_state_params["params"],
            )
        )
        tx = optax.multi_transform(partition_optimizers, param_partition)

    train_state = TrainState.create(
        apply_fn=model.apply,
        tx=tx,
        **train_state_params,
    )

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    if args.checkpoint_dir:
        restored = checkpointer.restore(args.checkpoint_dir.resolve(), item=train_state)
        if args.reinit_opt:
            restored = restored.replace(opt_state=train_state.opt_state)

        train_state = restored

    model_data.save(args.save_dir / "model_data.pkl")
    orbax_checkpointer.save(
        args.save_dir.resolve() / "initial_state",
        train_state_params,
        force=True,
    )

    return train_state


def main(args: argparse.Namespace):
    numeric_level = getattr(logging, args.logging_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.logging_level)
    logging.basicConfig(level=numeric_level)

    for flag in args.jax_config:
        jax.config.update(flag, True)
        logging.info(f"{flag} set to True")

    key = jax.random.PRNGKey(args.seed)

    if not args.save_dir.exists():
        args.save_dir.mkdir(parents=True)

    FileIO.save(args, args.save_dir / "args.pkl")

    dataset_split_key, sampler_key, params_key = jax.random.split(key, 3)

    datasets = create_datasets(
        args,
        dataset_split_key,
        sequential=args.architecture_type.is_recurrent() or args.sequence_length,
    )

    baseline_scaler, baseline_scaler_params = create_baseline_scaler(datasets["train"])

    training_scaler, training_scaler_params = create_training_scaler(
        args, datasets["train"]
    )

    datasets["train"].apply_scalers(*training_scaler_params, training_scaler)

    if "val" in datasets:
        datasets["val"].apply_scalers(*training_scaler_params, training_scaler)

    dataset_samplers = create_dataset_samplers(args, datasets, key=sampler_key)

    loss = LossType.create(args.loss)

    loss_regularizer = (
        (lambda model_params: (jnp.sum(model_params["rotation_params"] ** 2) - 1) ** 2)
        if args.use_io_rotator and args.data_tensor_dim == 3
        else None
    )

    train_state = create_train_state(
        args,
        params_key,
        args.max_epochs
        * len(dataset_samplers["train"].dataset)
        // dataset_samplers["train"].batch_size,
    )

    trainer = Trainer(
        state=train_state,
        loss=loss,
        training_dataset_sampler=dataset_samplers["train"],
        validation_dataset_sampler=dataset_samplers.get("val"),
        baseline_scaler=baseline_scaler,
        baseline_scaler_params=baseline_scaler_params,
        training_scaler=training_scaler,
        training_scaler_params=training_scaler_params,
        loss_regularizer=loss_regularizer,
        max_epochs=args.max_epochs,
        max_time=args.max_time * 60 if args.max_time else None,
        print_epoch_interval=args.print_epoch_interval,
        print_iteration_interval=args.print_iteration_interval,
        print_metrics=create_print_metrics(args, use_val="val" in datasets),
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        save_best_model=True,
        save_opt_state=False,
    )

    trainer.run()


if __name__ == "__main__":
    args = parse_args()
    main(args)
