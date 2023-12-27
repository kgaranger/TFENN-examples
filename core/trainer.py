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

import logging
import pickle
from collections import deque
from collections.abc import Callable
from functools import partial
from pathlib import Path
from time import time
from typing import Any

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
import tqdm
from flax import struct
from flax.core import freeze
from flax.training import train_state
from flax.traverse_util import TraverseTree as TraverseTree
from util import FileIO

from .dataset_sampler import DatasetSampler
from .metrics import Metrics, MetricType
from .scaler import DataScaler, DataScalingParams


class TrainState(train_state.TrainState):
    """TrainState with overridden apply_gradients method to return the updates for
    logging purposes."""

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return (
            self.replace(
                step=self.step + 1,
                params=new_params,
                opt_state=new_opt_state,
                **kwargs,
            ),
            updates,
        )


class Trainer:
    def __init__(
        self,
        state: TrainState,
        loss: Callable[[jnp.ndarray, jnp.ndarray], float],
        training_dataset_sampler: DatasetSampler,
        validation_dataset_sampler: DatasetSampler = None,
        baseline_scaler: DataScaler | None = None,
        baseline_scaler_params: DataScalingParams | None = None,
        training_scaler: DataScaler | None = None,
        training_scaler_params: DataScalingParams | None = None,
        loss_regularizer: Callable[[dict], float] | None = None,
        max_epochs: int = None,
        max_time: float = None,
        print_epoch_interval: int = 0,
        print_iteration_interval: int = 0,
        print_metrics: set[MetricType] = None,
        save_dir: Path = None,
        save_interval: int = 0,
        save_best_model: bool = False,
        save_opt_state: bool = False,
    ):
        if save_dir is None and (save_best_model or save_interval or save_opt_state):
            raise ValueError(
                "save_dir must be specified if save_best_model, "
                "save_interval or save_opt is True"
            )

        self.state = state
        self.loss = loss
        self.training_dataset_sampler = training_dataset_sampler
        self.validation_dataset_sampler = validation_dataset_sampler
        self.baseline_scaler = baseline_scaler
        self.baseline_scaler_params = baseline_scaler_params
        self.training_scaler = training_scaler
        self.training_scaler_params = training_scaler_params
        self.loss_regularizer = loss_regularizer
        self.max_epochs = max_epochs
        self.max_time = max_time
        self.print_iteration_interval = print_iteration_interval
        self.print_metrics = print_metrics
        self.save_dir = save_dir.resolve() if save_dir is not None else None
        self.save_interval = save_interval
        self.save_best_model = save_best_model
        self.metrics = Metrics(
            save_path=save_dir / "metrics.pkl" if save_dir else None,
            print_interval=print_epoch_interval,
            save_interval=1 if save_dir else 0,
            print_metrics=self.print_metrics,
        )

        self._reset()

    @partial(jax.jit, static_argnums=(0,))
    def _apply_loss(self, outputs, predictions, model_params=None):
        if self.loss_regularizer:
            return jax.lax.cond(
                model_params is None,
                jax.vmap(self.loss)(outputs, predictions).mean(),
                jax.vmap(self.loss)(outputs, predictions).mean()
                + self.loss_regularizer(model_params),
            )
        else:
            return jax.vmap(self.loss)(outputs, predictions).mean()

    @partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    def _eval_loss(
        self,
        state: TrainState,
        inputs: jax.typing.ArrayLike,
        outputs: jax.typing.ArrayLike,
        training_input_scaler: DataScaler,
        training_output_scaler: DataScaler,
        baseline_input_scaler: DataScaler,
        baseline_output_scaler: DataScaler,
        training_input_scaler_params: DataScalingParams,
        training_output_scaler_params: DataScalingParams,
        baseline_input_scaler_params: DataScalingParams,
        baseline_output_scaler_params: DataScalingParams,
        add_unscaled: bool = False,
    ) -> float:
        model_state = {"params": state.params}

        if (
            training_input_scaler != baseline_input_scaler
            or training_input_scaler_params != baseline_input_scaler_params
        ):
            training_like_scaled_inputs = training_input_scaler.scale(
                baseline_input_scaler.descale(inputs, baseline_input_scaler_params),
                training_input_scaler_params,
            )
        else:
            training_like_scaled_inputs = inputs
        if (
            training_output_scaler != baseline_output_scaler
            or training_output_scaler_params != baseline_output_scaler_params
        ):
            training_like_scaled_outputs = training_output_scaler.scale(
                baseline_output_scaler.descale(outputs, baseline_output_scaler_params),
                training_output_scaler_params,
            )
            same_output_scaling = False
        else:
            training_like_scaled_outputs = outputs
            same_output_scaling = True

        predictions = self.state.apply_fn(
            model_state, training_like_scaled_inputs, train=False
        )
        training_like_validation_loss = self._apply_loss(
            training_like_scaled_outputs, predictions, None
        )
        if same_output_scaling:
            validation_loss = training_like_validation_loss
        else:
            validation_like_predictions = baseline_output_scaler.scale(
                training_output_scaler.descale(
                    predictions, training_output_scaler_params
                ),
                baseline_output_scaler_params,
            )
            validation_loss = self._apply_loss(
                outputs, validation_like_predictions, None
            )

        if add_unscaled:
            unscaled_loss = self._apply_loss(
                baseline_output_scaler.descale(outputs, baseline_output_scaler_params),
                training_output_scaler.descale(
                    predictions, training_output_scaler_params
                ),
                None,
            )
        else:
            unscaled_loss = None
        return validation_loss, training_like_validation_loss, unscaled_loss

    @partial(jax.jit, static_argnums=(0,))
    def _training_step(
        self,
        state: TrainState,
        inputs,
        outputs,
    ) -> tuple[TrainState, float, Any, Any]:
        def loss_fn(params, inputs, outputs):
            predictions = self.state.apply_fn({"params": params}, inputs, train=True)
            if (
                self.training_scaler != self.baseline_scaler
                or self.training_scaler_params[1] != self.baseline_scaler_params[1]
            ):
                baseline_scaled_outputs = self.baseline_scaler.scale(
                    self.training_scaler.descale(
                        outputs, self.training_scaler_params[1]
                    ),
                    self.baseline_scaler_params[1],
                )
                baseline_scaled_predictions = self.baseline_scaler.scale(
                    self.training_scaler.descale(
                        predictions, self.training_scaler_params[1]
                    ),
                    self.baseline_scaler_params[1],
                )
            else:
                baseline_scaled_outputs = outputs
                baseline_scaled_predictions = predictions
            loss = self._apply_loss(
                baseline_scaled_outputs, baseline_scaled_predictions, params
            )
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params, inputs, outputs)

        state, updates = state.apply_gradients(grads=grads)

        return state, loss, grads, updates

    def _reset(self):
        self.epoch = 0
        self.iteration = 0
        self.min_training_loss = float("inf")
        self.min_validation_loss = float("inf")
        self.unscaled_val_loss = float("inf")
        self.training_like_validation_loss = float("inf")
        self.metrics.clear()

        try:
            self.epoch_training_loss_history.clear()
        except AttributeError:
            self.epoch_training_loss_history = deque()
        try:
            self.training_loss_history.clear()
        except AttributeError:
            self.training_loss_history = deque()
        if self.validation_dataset_sampler is not None:
            try:
                self.validation_loss_history.clear()
            except AttributeError:
                self.validation_loss_history = deque()

    def _update_losses(self) -> tuple[bool, bool]:
        self.training_loss_history.append(
            sum(self.epoch_training_loss_history)
            / len(self.epoch_training_loss_history)
        )
        self.epoch_training_loss_history.clear()
        if self.training_loss_history[-1] < self.min_training_loss:
            self.min_training_loss = self.training_loss_history[-1]
            new_best = True
        else:
            new_best = False

        if self.validation_dataset_sampler is not None:
            _, (inputs, outputs) = self.validation_dataset_sampler.sample()

            (
                validation_loss,
                training_like_validation_loss,
                unscaled_loss,
            ) = self._eval_loss(
                self.state,
                inputs,
                outputs,
                self.training_dataset_sampler.dataset.input_scaler,
                self.training_dataset_sampler.dataset.output_scaler,
                self.validation_dataset_sampler.dataset.input_scaler,
                self.validation_dataset_sampler.dataset.output_scaler,
                self.training_dataset_sampler.dataset.input_scaler_params,
                self.training_dataset_sampler.dataset.output_scaler_params,
                self.validation_dataset_sampler.dataset.input_scaler_params,
                self.validation_dataset_sampler.dataset.output_scaler_params,
                add_unscaled=MetricType.UNSCALED_VALIDATION_LOSS in self.print_metrics,
            )
            self.validation_loss_history.append(validation_loss)
            if self.validation_loss_history[-1] < self.min_validation_loss:
                self.min_validation_loss = self.validation_loss_history[-1]
                new_best_on_val = True
            else:
                new_best_on_val = False
            self.unscaled_val_loss = unscaled_loss
            self.training_like_validation_loss = training_like_validation_loss
        else:
            new_best_on_val = False

        return new_best, new_best_on_val

    def _stop_condition(self):
        if self.max_epochs is not None and self.epoch >= self.max_epochs:
            return True
        if self.max_time is not None and time() - self.start_time >= self.max_time:
            return True
        return False

    def run(self):
        self._reset()

        if self.save_best_model or self.save_interval:
            FileIO.save(
                {
                    "input_scaler": self.training_dataset_sampler.dataset.input_scaler,
                    "output_scaler": self.training_dataset_sampler.dataset.output_scaler,
                    "input_scaler_params": self.training_dataset_sampler.dataset.input_scaler_params,
                    "output_scaler_params": self.training_dataset_sampler.dataset.output_scaler_params,
                },
                self.save_dir / "scalers.pkl",
            )

        self.start_time = time()

        if self.print_iteration_interval:
            progress_bar = tqdm.tqdm(
                total=self.training_dataset_sampler.batch_count, dynamic_ncols=True
            )
            progress_bar.set_description(f"Epoch: {self.epoch}")
            self.metrics.print_fn = progress_bar.write

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        logging.getLogger("absl").setLevel(logging.ERROR)

        while not self._stop_condition():
            new_epoch, (inputs, outputs) = self.training_dataset_sampler.sample()

            (
                self.state,
                loss,
                grads,
                updates,
            ) = self._training_step(
                state=self.state,
                inputs=inputs,
                outputs=outputs,
            )

            self.epoch_training_loss_history.append(loss)

            self.iteration += 1

            if self.print_iteration_interval and (
                self.iteration % self.print_iteration_interval == 0
            ):
                progress_bar.update(self.print_iteration_interval)

            if new_epoch:
                new_best, new_best_on_val = self._update_losses()

                metrics_dict = {
                    MetricType.TIME.value: time() - self.start_time,
                    MetricType.TRAINING_LOSS.value: float(
                        self.training_loss_history[-1]
                    ),
                    MetricType.PARAMETERS_NORM.value: TraverseTree().update(
                        lambda x: float(jnp.linalg.norm(x)), self.state.params
                    ),
                    MetricType.GRADIENTS_NORM.value: TraverseTree().update(
                        lambda x: float(jnp.linalg.norm(x)), grads
                    ),
                    MetricType.UPDATES_NORM.value: TraverseTree().update(
                        lambda x: float(jnp.linalg.norm(x)), updates
                    ),
                }
                if self.validation_dataset_sampler is not None:
                    metrics_dict[MetricType.VALIDATION_LOSS.value] = float(
                        self.validation_loss_history[-1]
                    )
                    if MetricType.UNSCALED_VALIDATION_LOSS in self.print_metrics:
                        metrics_dict[MetricType.UNSCALED_VALIDATION_LOSS.value] = float(
                            self.unscaled_val_loss
                        )
                    if MetricType.TRAINING_LIKE_VALIDATION_LOSS in self.print_metrics:
                        metrics_dict[
                            MetricType.TRAINING_LIKE_VALIDATION_LOSS.value
                        ] = float(self.training_like_validation_loss)

                self.metrics.update(self.epoch, **metrics_dict)

                if new_best and self.save_best_model:
                    orbax_checkpointer.save(
                        self.save_dir / "best_state",
                        self.state,
                        force=True,
                    )

                if new_best_on_val and self.save_best_model:
                    orbax_checkpointer.save(
                        self.save_dir / "best_state_on_val",
                        self.state,
                        force=True,
                    )

                if self.save_interval > 0 and self.epoch % self.save_interval == 0:
                    orbax_checkpointer.save(
                        self.save_dir / f"state_{self.epoch}",
                        self.state,
                        force=True,
                    )

                self.epoch += 1
                self.iteration = 0

                if self.print_iteration_interval:
                    progress_bar.reset()
                    progress_bar.set_description(f"Epoch: {self.epoch}")

        if self.print_iteration_interval:
            progress_bar.close()
