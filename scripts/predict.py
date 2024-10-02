import argparse
import logging
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import orbax
import tqdm

from .cmd_args import CmdArgs

        
from core import (ActivationType, ArchitectureType, DataScaler, DataScalerType,
                  DataScalingMethodType, DataScalingParams, Dataset,
                  DatasetSampler, DatasetSamplerType, EmptyDataset,
                  FeatureType, InitializerType, IORotator, LossType,
                  MetricType, ModelData, NetworkType, OptimizerType,
                  SchedulerType, SequenceDataset, Trainer, TrainState,
                  ValueAndGradDataset)

from util import FileIO



def predictions_iter(
    dataset_sampler: DatasetSampler,
    apply_fn,
    state: flax.struct.PyTreeNode,
    scalers: dict,
):
    while True:
        new_epoch, (sample_in, sample_out) = dataset_sampler.sample()
        if new_epoch:
            return
         
        pred = apply_fn(state, sample_in)
        yield (
            sample_in,
            pred,
	    scalers["output_scaler"].scale(sample_out, scalers["output_scaler_params"]),
	    scalers["output_scaler"].descale(pred, scalers["output_scaler_params"]),
            scalers["input_scaler"].descale(sample_in, scalers["input_scaler_params"]),
        )


def parse_args(args: argparse.Namespace = None):
    parser = argparse.ArgumentParser()

    parser = CmdArgs.add_config_args(parser)
    parser = CmdArgs.add_model_args(parser)
    parser = CmdArgs.add_dataset_args(parser)
    parser = CmdArgs.add_testing_args(parser)
    parser = CmdArgs.add_save_args(parser)
    parser = CmdArgs.add_logging_args(parser)

    args = parser.parse_args(args)

    if args.checkpoint_dir is None:
        raise ValueError("checkpoint_dir argument must be specified")
    if args.dataset_scaling_file is None:
        raise ValueError("dataset_normalization_file argument must be specified")

    if args.file_in_cols is None:
        args.file_in_cols = list(range(jnp.prod(jnp.array(args.input_shape))))
    if args.file_out_cols is None:
        args.file_out_cols = list(
            range(
                args.file_in_cols[-1] + 1,
                args.file_in_cols[-1] + 1 + jnp.prod(jnp.array(args.output_shape)),
            )
        )

    return args


def main(args: argparse.Namespace):
    logging.basicConfig(level=args.logging_level)

    model_params = { 
        "layer_count": args.layer_count,
        "layer_size": args.layer_size,
        "activation": args.activation,
        "output_activation": args.output_activation,
        "kernel_initializer": args.kernel_initializer,
        "bias_initializer": args.bias_initializer,
    }


    if len(args.dataset_files) > 1:
        if args.prediction_files:
            if args.prediction_files.suffix != "":
                raise ValueError(
                    "If multiple dataset files are provided, output_path must be a "
                    "directory"
                )
            if not args.prediction_files.exists():
                args.prediction_files.mkdir(parents=True)
        else:
            raise ValueError(
                "If multiple dataset files are provided, output_path must be "
                "specified"
            )
        file_paths = [
            args.prediction_files / f"{dataset_file.name}"
            for dataset_file in args.dataset_files
        ]
        
    else:
        file_paths = [args.prediction_files]
        

   
    if args.feature_type == FeatureType.SCALAR:
        
        model = NetworkType((args.architecture_type, args.feature_type)).create(    
        **model_params
        )

   	 
    elif args.feature_type == FeatureType.TENSOR:
        if args.data_type != FeatureType.TENSOR:
            raise ValueError(
                f"Data type {args.data_type} not supported for tensor inputs"
            )
        data_notation = args.data_notation
        input_size = jax.numpy.prod(jax.numpy.array(args.input_shape))
        output_size = jax.numpy.prod(jax.numpy.array(args.output_shape))
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

        model = NetworkType((args.architecture_type, args.feature_type)).create(       
            **model_params
        ) 

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    model_state = checkpointer.restore(args.checkpoint_dir)

    if args.loss is not None:
        loss_fn = LossType.create(args.loss)
    
    apply_fn = jax.jit(
        lambda state, sample: model.apply(
            state,
	    sample ,
        )
    )

    scalers = FileIO.load(args.dataset_scaling_file)

    for dataset_file,file_path in zip(args.dataset_files, file_paths):
        dataset = Dataset(
            file_path=dataset_file,
            input_shape=args.input_shape,
            output_shape=args.output_shape,
            input_cols=args.file_in_cols,
            output_cols=args.file_out_cols,
        )
         
        dataset.apply_scalers(scalers['input_scaler_params'], scalers['output_scaler_params'], scalers['input_scaler'], scalers['output_scaler'])

        input_scaler = scalers["input_scaler"]
        input_scaler_params = scalers["input_scaler_params"]

        dataset_sampler = DatasetSamplerType.SEQUENTIAL.create(
            dataset=dataset,
        )

        predictions = predictions_iter(
            dataset_sampler,
            apply_fn,
            {
               "params": model_state["params"],
            },
            scalers,
        )


        try:
            
            if file_path is not None:
                f = open(file_path, "w")
                if file_path.suffix == ".csv" and dataset_file.suffix == ".csv":
                    with open(dataset_file, "r") as g:
                        src_header = g.readline().split("\n")[0].split(",")
                        header = [
                            src_header[k]
                            for k in args.file_in_cols + args.file_out_cols
                        ]
                        f.write(",".join(header) + "\n")
                else:
                    raise ValueError(
                        f"Unsupported output file type: {file_path.suffix}"
                    )

            if args.loss is not None:
                losses = []

            for preds in tqdm.tqdm(predictions):
                for sample_in, pred, sample_out, scaled_pred, descaled_input in zip(*preds):
                    if args.loss is not None:
                        losses.append(loss_fn(pred, sample_out))
                        

                    if file_path is not None:
                        f.write(
                            ",".join(
                                [
                                    str(x)
                                    for x in descaled_input.flatten().tolist()
                                    + scaled_pred.flatten().tolist()
                                    
                                ]
                            )
                            + "\n"
                        )

            if args.loss is not None:
                loss = jnp.mean(jnp.array(losses))
                logging.info(f"Loss for dataset {dataset_file.stem}: {loss:.3e}")
                

        finally:
            if file_path is not None:
                f.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)

