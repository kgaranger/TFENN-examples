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

import abc
import dataclasses
import logging
from collections.abc import Callable, Sequence

import jax
import numpy as np
from TFENN.core.symmetric_tensor_representation import SymmetricTensorNotation
from TFENN.util.array_util import canonicalize_tuple, normalize_axes
from TFENN.util.enum_input import EnumInputClass, EnumInputFun

from .feature_type import FeatureType


class DataScalerType(EnumInputClass):
    """Data normalizer type."""

    SCALAR = FeatureType.SCALAR
    TENSOR = FeatureType.TENSOR

    @classmethod
    @property
    def obj_map(cls) -> dict["DataScalerType", "DataScaler"]:
        return {cls.SCALAR: ScalarScaler, cls.TENSOR: SymmetricTensorScaler}


@dataclasses.dataclass(frozen=True)
class DataScalingParams:
    """Data scaling parameters."""

    shift: float | np.ndarray = 0.0
    scale: float | np.ndarray = 1.0

    @staticmethod
    def chain(
        *, first: "DataScalingParams", second: "DataScalingParams"
    ) -> (
        "DataScalingParams"
    ):  # accepts keyword-only arguments to prevent being used in wrong order
        """Chain two scaling parameters."""
        return DataScalingParams(
            scale=second.scale * first.scale,
            shift=first.shift + first.scale * second.shift,
        )

    def __hash__(self):
        return hash((str(self.shift), str(self.scale)))

    def __eq__(self, other):
        return (
            isinstance(other, DataScalingParams)
            and np.allclose(self.shift, other.shift)
            and np.allclose(self.scale, other.scale)
        )


class DataScalingMethodType(EnumInputFun):
    """Data scaling method type."""

    MAX = "max"
    NORMAL = "normal"
    NONE = None

    @classmethod
    @property
    def fun_map(
        cls,
    ) -> dict["DataScalingMethodType", Callable[[np.ndarray], DataScalingParams]]:
        return {
            cls.MAX: (
                lambda x, axis=None: (np.max(x, axis=axis) + np.min(x, axis=axis)) / 2,
                lambda x, axis=None: np.maximum(
                    1e-6, (np.max(x, axis=axis) - np.min(x, axis=axis)) / 2
                ),
            ),
            cls.NORMAL: (
                lambda x, axis=None: np.mean(x, axis=axis),
                lambda x, axis=None: np.std(x, axis=axis),
            ),
            cls.NONE: (lambda x: 0.0, lambda x: 1.0),
        }


@dataclasses.dataclass(frozen=True)
class DataScaler(abc.ABC):
    method: DataScalingMethodType
    feature_axes: Sequence[int]

    def __post_init__(self):
        object.__setattr__(self, "feature_axes", canonicalize_tuple(self.feature_axes))

    @abc.abstractmethod
    def compute_params(
        self,
        data: np.ndarray,
    ) -> DataScalingParams:
        pass

    @abc.abstractmethod
    def scale(self, data: np.ndarray, params: DataScalingParams) -> np.ndarray:
        pass

    @abc.abstractmethod
    def descale(self, data: np.ndarray, params: DataScalingParams) -> np.ndarray:
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass


@dataclasses.dataclass(frozen=True)
class ScalarScaler(DataScaler):
    """Scalar data normalizer."""

    def compute_params(
        self,
        data: np.ndarray,
    ) -> DataScalingParams:
        feature_axes = normalize_axes(canonicalize_tuple(self.feature_axes), data.ndim)
        op_axes = tuple(set(range(data.ndim)) - set(feature_axes))
        method = self.method.create()
        params = (method[0](data, axis=op_axes), method[1](data, axis=op_axes))

        return DataScalingParams(shift=params[0], scale=params[1])

    def scale(self, data: np.ndarray, params: DataScalingParams) -> np.ndarray:
        feature_axes = normalize_axes(canonicalize_tuple(self.feature_axes), data.ndim)
        batch_axes = tuple(set(range(data.ndim)) - set(feature_axes))

        normalized_data = (
            data - np.expand_dims(params.shift, batch_axes)
        ) / np.expand_dims(params.scale, batch_axes)

        try:
            return normalized_data.reshape(data.shape)
        except ValueError:
            raise ValueError(
                f"Expected normalized data to have shape {data.shape}, "
                f"but got {normalized_data.shape}."
            )

    def descale(self, data: np.ndarray, params: DataScalingParams) -> np.ndarray:
        feature_axes = normalize_axes(canonicalize_tuple(self.feature_axes), data.ndim)
        batch_axes = tuple(set(range(data.ndim)) - set(feature_axes))

        denormalized_data = data * np.expand_dims(
            params.scale, batch_axes
        ) + np.expand_dims(params.shift, batch_axes)

        try:
            return denormalized_data.reshape(data.shape)
        except ValueError:
            raise ValueError(
                f"Expected denormalized data to have shape {data.shape}, "
                f"but got {denormalized_data.shape}."
            )

    def __eq__(self, other):
        return isinstance(other, ScalarScaler) and self.method == other.method


@dataclasses.dataclass(frozen=True)
class SymmetricTensorScaler(DataScaler):
    data_notation: SymmetricTensorNotation
    tensors_axis: int | Sequence[int] | None = None
    # old = 0

    def __post_init__(self):
        if self.tensors_axis is None:
            object.__setattr__(
                self, "tensors_axis", range(-len(self.data_notation.reduced_shape), 0)
            )
        super().__post_init__()

    def compute_params(
        self,
        data: np.ndarray,
    ) -> DataScalingParams:
        tensors_axis = normalize_axes(canonicalize_tuple(self.tensors_axis), data.ndim)

        data = np.diagonal(
            self.data_notation.to_full(data, tensors_axis=tensors_axis),
            axis1=tensors_axis[0],
            axis2=-1,
        )
        method = self.method.create()
        feature_axes = normalize_axes(canonicalize_tuple(self.feature_axes), data.ndim)
        op_axes = tuple(set(range(data.ndim)) - set(feature_axes))
        params = (method[0](data, axis=op_axes), method[1](data, axis=op_axes))

        return DataScalingParams(shift=params[0], scale=params[1])

    def scale(
        self,
        data: np.ndarray,
        params: DataScalingParams,
    ) -> np.ndarray:
        tensors_axis = normalize_axes(canonicalize_tuple(self.tensors_axis), data.ndim)
        feature_axes = normalize_axes(canonicalize_tuple(self.feature_axes), data.ndim)
        batch_axes = tuple(
            set(range(data.ndim)) - set(tensors_axis) - set(feature_axes)
        )

        normalized_data = (
            data
            - self.data_notation.to_reduced(
                np.expand_dims(params.shift, axis=batch_axes + tensors_axis + (-1,))
                * np.expand_dims(
                    np.eye(self.data_notation.dim), axis=batch_axes + feature_axes
                ),
                tensors_axis=tensors_axis + (-1,),
            )
        ) / np.expand_dims(params.scale, axis=batch_axes + tensors_axis)

        try:
            return normalized_data.reshape(data.shape)
        except ValueError:
            raise ValueError(
                f"Expected normalized data to have shape {data.shape}, "
                f"but got {normalized_data.shape}."
            )

    def descale(
        self,
        data: np.ndarray,
        params: DataScalingParams,
    ) -> np.ndarray:
        tensors_axis = normalize_axes(canonicalize_tuple(self.tensors_axis), data.ndim)
        batch_axis = [axis for axis in range(data.ndim) if axis not in tensors_axis]

        tensors_axis = normalize_axes(canonicalize_tuple(self.tensors_axis), data.ndim)
        feature_axes = normalize_axes(canonicalize_tuple(self.feature_axes), data.ndim)
        batch_axes = tuple(
            set(range(data.ndim)) - set(tensors_axis) - set(feature_axes)
        )

        denormalized_data = data * np.expand_dims(
            params.scale, axis=batch_axes + tensors_axis
        ) + self.data_notation.to_reduced(
            np.expand_dims(params.shift, axis=batch_axes + tensors_axis + (-1,))
            * np.expand_dims(
                np.eye(self.data_notation.dim), axis=batch_axes + feature_axes
            ),
            tensors_axis=tensors_axis + (-1,),
        )

        try:
            return denormalized_data.reshape(data.shape)
        except ValueError:
            raise ValueError(
                f"Expected denormalized data to have shape {data.shape}, "
                f"but got {denormalized_data.shape}."
            )

    def __eq__(self, other):
        return (
            isinstance(other, SymmetricTensorScaler)
            and self.method == other.method
            and self.data_notation == other.data_notation
            and self.tensors_axis == other.tensors_axis
        )
