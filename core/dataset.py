import logging
from pathlib import Path

import numpy as np
from jax import random
from numpy.typing import NDArray

from .feature_type import FeatureType
from .scaler import DataScaler, DataScalerType, DataScalingParams


class Dataset:
    def __init__(
        self,
        arrays: tuple[NDArray, ...] = None,
        file_path: Path = None,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        input_cols: list[int] = None,
        output_cols: list[int] = None,
        **kwargs,
    ):
        self._logger = logging.getLogger(__name__)

        self.scaled = False
        self.input_scaler = None
        self.input_scaler_params = DataScalingParams()
        self.output_scaler = None
        self.output_scaler_params = DataScalingParams()

        if arrays is not None and file_path is not None:
            raise ValueError(
                "Dataset cannot be initialized with both arrays and a file path",
            )

        if arrays is not None:
            self._from_arrays(arrays, input_shape, output_shape, **kwargs)
        elif file_path is not None:
            if file_path.suffix == ".csv":
                if input_cols is None:
                    input_cols = list(range(np.prod(np.array(input_shape))))
                if output_cols is None:
                    output_cols = range(
                        input_cols[-1] + 1,
                        input_cols[-1] + 1 + np.prod(np.array(output_shape)),
                    )
                self._logger.info(f"Loading dataset from {file_path.name}")
                self._load_from_csv(
                    file_path,
                    input_shape,
                    output_shape,
                    input_cols,
                    output_cols,
                    **kwargs,
                )
            else:
                raise ValueError(
                    f"Format {file_path.suffix} not known for datasets files"
                )
        else:
            self._from_arrays((np.empty((0,)), np.empty((0,))))

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        return self.input[index], self.output[index]

    def _from_arrays(
        self,
        arrays: tuple[NDArray, ...],
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        **kwargs,
    ):
        in_array = arrays[0]
        out_array = arrays[1]

        if input_shape is not None:
            self.input = in_array.reshape((in_array.shape[0],) + tuple(input_shape))
        else:
            self.input = in_array
        if output_shape is not None:
            self.output = out_array.reshape((in_array.shape[0],) + tuple(output_shape))
        else:
            self.output = out_array

    def _load_from_csv(
        self,
        file_path: Path,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        input_cols: list[int],
        output_cols: list[int],
        **kwargs,
    ):
        with open(file_path, "r") as f:
            header = f.readline().strip().split(",")
            self._logger.info(f"Input columns: {[header[c] for c in input_cols]}")
            self._logger.info(f"Output columns: {[header[c] for c in output_cols]}")

        rows = np.loadtxt(file_path, delimiter=",", skiprows=1)

        size = rows.shape[0]
        self.input = rows[:, input_cols].reshape((size,) + tuple(input_shape))
        self.output = rows[:, output_cols].reshape((size,) + tuple(output_shape))

    def fit_scalers(
        self,
        input_scaler: DataScalerType,
        output_scaler: DataScalerType | None = None,
        **kwargs,
    ) -> tuple[DataScalingParams, DataScalingParams]:
        if output_scaler is None:
            output_scaler = input_scaler
        return (
            input_scaler.compute_params(self.input, **kwargs),
            output_scaler.compute_params(self.output, **kwargs),
        )

    def apply_scalers(
        self,
        input_scaler_params: DataScalingParams,
        output_scaler_params: DataScalingParams,
        input_scaler: DataScalerType,
        output_scaler: DataScalerType | None = None,
        force: bool = False,
    ):
        if output_scaler is None:
            output_scaler = input_scaler
        if self.scaled:
            if (
                input_scaler == self.input_scaler
                and output_scaler == self.output_scaler
            ):
                self._logger.info("Dataset already scaled, chaining scalers")
                self.input_scaler_params = DataScalingParams.chain(
                    first=self.input_scaler_params,
                    second=input_scaler_params,
                )
                self.output_scaler_params = DataScalingParams.chain(
                    first=self.output_scaler_params,
                    second=output_scaler_params,
                )
            else:
                if force:
                    self._logger.warning("Dataset already scaled, replacing scalers")
                    self.input_scaler = input_scaler
                    self.output_scaler = output_scaler
                    self.input_scaler_params = input_scaler_params
                    self.output_scaler_params = output_scaler_params
                else:
                    raise ValueError("Dataset already scaled with different scalers")
        else:
            self.input_scaler = input_scaler
            self.output_scaler = output_scaler
            self.input_scaler_params = input_scaler_params
            self.output_scaler_params = output_scaler_params

        self.input = input_scaler.scale(self.input, input_scaler_params)
        self.output = output_scaler.scale(self.output, output_scaler_params)
        self.scaled = True

    def fit_and_apply_scalers(
        self,
        input_scaler: DataScalerType,
        output_scaler: DataScalerType,
        **kwargs,
    ) -> tuple[DataScalingParams, DataScalingParams]:
        input_scaler_params, output_scaler_params = self.fit_scalers(
            input_scaler, output_scaler, **kwargs
        )
        self.apply_scalers(
            input_scaler,
            output_scaler,
            input_scaler_params,
            output_scaler_params,
        )
        return input_scaler_params, output_scaler_params

    def scale_input(self, x: NDArray) -> NDArray:
        return self.input_scaler.scale(x, self.input_scaler_params)

    def scale_output(self, y: NDArray) -> NDArray:
        return self.output_scaler.scale(y, self.output_scaler_params)

    def descale_input(self, x: NDArray) -> NDArray:
        return self.input_scaler.descale(x, self.input_scaler_params)

    def descale_output(self, y: NDArray) -> NDArray:
        return self.output_scaler.descale(y, self.output_scaler_params)

    def clip(self, max_samples: int | None):
        if max_samples is not None:
            self.input = self.input[:max_samples]
            self.output = self.output[:max_samples]

    def split(self, ratio: float = 0.8, key: random.PRNGKey = None):
        if ratio < 0 or ratio > 1:
            raise ValueError(
                f"Ratio must be between 0 and 1, got {ratio} instead",
            )

        split_index = round(ratio * len(self))
        if key is None:
            self._logger.info(f"Unshuffled dataset split at index {split_index}")
            d1 = Dataset(arrays=self[:split_index])
            d2 = Dataset(arrays=self[split_index:])
        else:
            self._logger.info(f"Randomly shuffled dataset split at index {split_index}")
            perm = random.permutation(key, len(self))
            d1 = Dataset(arrays=self[perm[:split_index]])
            d2 = Dataset(arrays=self[perm[split_index:]])

        if self.scaled:
            d1.scaled = True
            d1.input_scaler = self.input_scaler
            d1.output_scaler = self.output_scaler
            d1.input_scaler_params = self.input_scaler_params
            d1.output_scaler_params = self.output_scaler_params
            d2.scaled = True
            d2.input_scaler = self.input_scaler
            d2.output_scaler = self.output_scaler
            d2.input_scaler_params = self.input_scaler_params
            d2.output_scaler_params = self.output_scaler_params

        return d1, d2

    def __add__(self, other: "Dataset") -> "Dataset":
        if self.scaled != other.scaled:
            raise ValueError(
                "Cannot add datasets with different scaling status",
            )
        if self.scaled:
            if self.input_scaler != other.input_scaler:
                raise ValueError(
                    "Cannot add datasets with different input scalers",
                )
            if self.input_scaler_params != other.input_scaler_params:
                raise ValueError(
                    "Cannot add datasets with different input scaling params",
                )
            if self.output_scaler != other.output_scaler:
                raise ValueError(
                    "Cannot add datasets with different output scalers",
                )
            if self.output_scaler_params != other.output_scaler_params:
                raise ValueError(
                    "Cannot add datasets with different output scaling params",
                )

        dataset = Dataset(
            arrays=(
                np.concatenate((self.input, other.input), axis=0),
                np.concatenate((self.output, other.output), axis=0),
            )
        )
        dataset.scaled = self.scaled
        dataset.input_scaler = self.input_scaler
        dataset.input_scaler_params = self.input_scaler_params
        dataset.output_scaler = self.output_scaler
        dataset.output_scaler_params = self.output_scaler_params

        return dataset

    @classmethod
    def concatenate(cls, datasets: list["Dataset"]) -> "Dataset":
        if len(datasets) == 0:
            return EmptyDataset()

        if len(datasets) == 1:
            return datasets[0]

        if not all([d.scaled == datasets[0].scaled for d in datasets]):
            raise ValueError(
                "Cannot concatenate datasets with different scaling status",
            )
        if not all([d.input_scaler == datasets[0].input_scaler for d in datasets]):
            raise ValueError(
                "Cannot concatenate datasets with different input scalers",
            )
        if not all(
            [d.input_scaler_params == datasets[0].input_scaler_params for d in datasets]
        ):
            raise ValueError(
                "Cannot concatenate datasets with different input scaling params",
            )
        if not all([d.output_scaler == datasets[0].output_scaler for d in datasets]):
            raise ValueError(
                "Cannot concatenate datasets with different output scalers",
            )
        if not all(
            [
                d.output_scaler_params == datasets[0].output_scaler_params
                for d in datasets
            ]
        ):
            raise ValueError(
                "Cannot concatenate datasets with different output scaling params",
            )

        dataset = cls(
            arrays=(
                np.concatenate([d.input for d in datasets], axis=0),
                np.concatenate([d.output for d in datasets], axis=0),
            )
        )
        dataset.scaled = datasets[0].scaled
        dataset.input_scaler = datasets[0].input_scaler
        dataset.input_scaler_params = datasets[0].input_scaler_params
        dataset.output_scaler = datasets[0].output_scaler
        dataset.output_scaler_params = datasets[0].output_scaler_params

        return dataset


class EmptyDataset(Dataset):
    """Empty dataset to allow the use of the built-in sum function on datasets"""

    def __init__(self):
        super().__init__(arrays=(np.empty((0,)), np.empty((0,))))

    def __add__(self, other: Dataset) -> Dataset:
        return other


class SequenceDataset(Dataset):
    def __init__(
        self,
        arrays: tuple[NDArray, ...] = None,
        file_path: Path = None,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        input_cols: list[int] = None,
        output_cols: list[int] = None,
        sequence_length: int | None = None,
        **kwargs,
    ):
        super().__init__(
            arrays=arrays,
            file_path=file_path,
            input_shape=input_shape,
            output_shape=output_shape,
            input_cols=input_cols,
            output_cols=output_cols,
            sequence_length=sequence_length,
            **kwargs,
        )

    def _from_arrays(
        self,
        arrays: tuple[NDArray, ...],
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        sequence_length: int | None = None,
        **kwargs,
    ):
        in_array = arrays[0]
        out_array = arrays[1]

        if sequence_length is None:
            if input_shape is not None:
                if input_shape == in_array.shape[2:]:  # Assume provided input is in
                    # the right shape
                    self.input = in_array
                else:  # Assume only one sequence
                    self.input = in_array.reshape((1, -1) + tuple(input_shape))
            else:
                if in_array.ndim == 1:  # Assume only one sequence
                    self.input = np.expand_dims(in_array, axis=0)
                else:  # Assume provided input is in the right shape
                    self.input = in_array

            if output_shape is not None:
                if output_shape == out_array.shape[2:]:  # Assume provided output is in
                    # the right shape
                    self.output = out_array
                else:
                    self.output = out_array.reshape((1, -1) + tuple(output_shape))
            else:
                if out_array.ndim == 1:  # Assume only one sequence
                    self.output = np.expand_dims(out_array, axis=0)
                else:  # Assume provided output is in the right shape
                    self.output = out_array
            if self.output.shape[:2] != self.input.shape[:2]:
                raise ValueError(
                    "Input and output must have the same number of sequences"
                    " and the same sequence length, "
                    f"got {self.input.shape[:2]} and {self.output.shape[:2]}"
                )

        else:
            if input_shape is not None:
                self.input = in_array.reshape(
                    (-1, sequence_length) + tuple(input_shape)
                )
            else:
                if in_array.ndim == 1:
                    self.input = in_array.reshape((-1, sequence_length))
                elif in_array.ndim == 2:
                    self.input = in_array.reshape(
                        (-1, sequence_length, in_array.shape[-1])
                    )
                else:
                    if in_array.shape[1] != sequence_length:
                        raise ValueError(
                            f"Sequence length {sequence_length} does not match "
                            f"the number of samples {in_array.shape[1]}"
                        )
            if output_shape is not None:
                self.output = out_array.reshape(
                    (-1, sequence_length) + tuple(output_shape)
                )
            else:
                if out_array.ndim == 1:
                    self.output = out_array.reshape((-1, sequence_length))
                elif out_array.ndim == 2:
                    self.output = out_array.reshape(
                        (-1, sequence_length, out_array.shape[-1])
                    )
                else:
                    if out_array.shape[1] != sequence_length:
                        raise ValueError(
                            f"Sequence length {sequence_length} does not match "
                            f"the number of samples {out_array.shape[1]}"
                        )
            if self.output.shape[0] != self.input.shape[0]:
                raise ValueError(
                    "Input and output must have the same number of sequences"
                    f" got {self.input.shape[0]} and {self.output.shape[0]}"
                )

    def _load_from_csv(
        self,
        file_path: Path,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        input_cols: list[int],
        output_cols: list[int],
        sequence_length: int | None = None,
        **kwargs,
    ):
        with open(file_path, "r") as f:
            header = f.readline().strip().split(",")
            try:
                self._logger.info(f"Input columns: {[header[c] for c in input_cols]}")
            except IndexError:
                raise IndexError(
                    f"Input columns {input_cols} are out of range for file {file_path}"
                    f" with header {header}"
                )
            try:
                self._logger.info(f"Output columns: {[header[c] for c in output_cols]}")
            except IndexError:
                raise IndexError(
                    f"Output columns {output_cols} are out of range for file "
                    f"{file_path} with header {header}"
                )

        rows = np.loadtxt(file_path, delimiter=",", skiprows=1)

        size = rows.shape[0]
        if sequence_length:
            if size % sequence_length != 0:
                raise ValueError(
                    f"Sequence length {sequence_length} does not match the number "
                    f"of samples {size}"
                )
            size = size // sequence_length
        else:
            size = 1
            sequence_length = -1
        self.input = rows[:, input_cols].reshape(
            (size, sequence_length) + tuple(input_shape)
        )
        self.output = rows[:, output_cols].reshape(
            (size, sequence_length) + tuple(output_shape)
        )


class ValueAndGradDataset(Dataset):
    def __init__(
        self,
        value_dataset: Dataset,
        grad_dataset: Dataset,
    ):
        assert len(value_dataset) == len(grad_dataset)

        self.value_dataset = value_dataset
        self.grad_dataset = grad_dataset

        self.scaled = value_dataset.scaled
        self.input_scaler = value_dataset.input_scaler
        self.input_scaler_params = DataScalingParams()
        self.output_scaler = value_dataset.output_scaler
        self.output_scaler_params = DataScalingParams()

    def __len__(self):
        return len(self.value_dataset)

    def __getitem__(self, index):
        return tuple(zip(self.value_dataset[index], self.grad_dataset[index]))

    def fit_scalers(
        self,
        input_scaler: DataScalerType,
        output_scaler: DataScalerType,
    ) -> tuple[DataScalingParams, DataScalingParams]:
        return self.value_dataset.fit_scalers(input_scaler, output_scaler)

    def apply_scalers(
        self,
        input_scaler: DataScalerType,
        output_scaler: DataScalerType,
        input_scaler_params: DataScalingParams,
        output_scaler_params: DataScalingParams,
        force: bool = False,
    ):
        self.value_dataset.apply_scalers(
            input_scaler,
            output_scaler,
            input_scaler_params,
            output_scaler_params,
            force=force,
        )
        self.grad_dataset.apply_scalers(
            input_scaler,
            output_scaler,
            input_scaler_params,
            output_scaler_params=DataScalingParams(
                shift=0.0,
                scale=self.value_dataset.output_scaler.scale
                / self.value_dataset.input_scaler.scale,
            ),
            force=force,
        )

        raise NotImplementedError("TODO: different scaling types for value and grad")

    def clip(self, max_samples: int | None):
        self.value_dataset.clip(max_samples)
        self.grad_dataset.clip(max_samples)

    def split(self, ratio: float = 0.8):
        return tuple(
            map(
                lambda d_it: ValueAndGradDataset(*list(d_it)),
                zip(self.value_dataset.split(ratio), self.grad_dataset.split(ratio)),
            )
        )
