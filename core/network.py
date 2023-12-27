import dataclasses
from abc import abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any

import jax
from flax import linen as nn
from jax import numpy as jnp
from jax.typing import ArrayLike
from TFENN.core.symmetric_tensor_representation import (
    MandelNotation, SymmetricTensorNotationType, SymmetricTensorRepresentation,
    TensorSymmetryClassType)
from TFENN.core.tensor_layer import (DenseSymmetricTensor,
                                     GRUCellSymmetricTensor,
                                     RotateSymmetricTensor, TensorActivation)
from TFENN.util.enum_input import EnumInputClass

from .activation import ActivationType
from .feature_type import FeatureType
from .initializer import InitializerType


class ArchitectureType(Enum):
    MLP = "mlp"
    GRU = "gru"
    LSTM = "lstm"

    def is_recurrent(self):
        return self in (ArchitectureType.GRU, ArchitectureType.LSTM)


class NetworkType(EnumInputClass):
    MLPScalarFeaturesNetwork = (ArchitectureType.MLP, FeatureType.SCALAR)
    MLPTensorFeaturesNetwork = (ArchitectureType.MLP, FeatureType.TENSOR)
    GRUScalarFeaturesNetwork = (ArchitectureType.GRU, FeatureType.SCALAR)
    GRUTensorFeaturesNetwork = (ArchitectureType.GRU, FeatureType.TENSOR)
    LSTMScalarFeaturesNetwork = (ArchitectureType.LSTM, FeatureType.SCALAR)
    LSTMTensorFeaturesNetwork = (ArchitectureType.LSTM, FeatureType.TENSOR)

    @classmethod
    @property
    def obj_map(cls) -> dict["NetworkType", "Network"]:
        return {
            cls.MLPScalarFeaturesNetwork: MLPScalarFeaturesNetwork,
            cls.MLPTensorFeaturesNetwork: MLPTensorFeaturesNetwork,
            cls.GRUScalarFeaturesNetwork: GRUScalarFeaturesNetwork,
            cls.GRUTensorFeaturesNetwork: GRUTensorFeaturesNetwork,
            cls.LSTMScalarFeaturesNetwork: LSTMScalarFeaturesNetwork,
            cls.LSTMTensorFeaturesNetwork: LSTMTensorFeaturesNetwork,
        }

    def is_recurrent(self):
        return self.value[0].is_recurrent()


@dataclasses.dataclass(frozen=True)
class ModelData:
    """Dataclass for storing a model initialization function and its arguments.
    :param init_fn: Function that initializes the model.
    :type init_fn: Callable
    :param args: The positional arguments to the model's init function.
    :type args: tuple
    :param kwargs: The keyword arguments of the model.
    :type kwargs: dict
    """

    init_fn: Callable
    args: tuple = dataclasses.field(default_factory=tuple)
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def save(self, path: str):
        """Saves the model data to a file.
        :param path: The path to the file.
        :type path: str
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """Loads the model data from a file.
        :param path: The path to the file.
        :type path: str
        :return: The model data.
        :rtype: ModelData
        """
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)

    def init(self):
        """Initializes the model.
        :return: The model.
        :rtype: nn.Module
        """
        return self.init_fn(*self.args, **self.kwargs)


@dataclasses.dataclass(kw_only=True)
class IORotator(nn.Module):
    """Model that rotates some input tensors according to a learnable rotation matrix,
    passes them through a provided model, and then rotates the output back.
    :param model: The model to rotate the inputs through.
    :type model: nn.Module
    """

    model: nn.Module

    def setup(self):
        self.rotation = RotateSymmetricTensor(
            dim=self.model.input_notation.dim,
            rotation_init=self.model.kernel_initializer,
        )
        self.notation = MandelNotation(
            order=self.model.input_notation.order, dim=self.model.input_notation.dim
        )

    @nn.compact
    def __call__(self, x: ArrayLike, train: bool = False) -> ArrayLike:
        if not isinstance(self.model.input_notation, MandelNotation):
            x = self.model.input_notation.to_full(x)
            x = self.notation.to_reduced(x)

        x = self.rotation(x, transpose=True)

        if not isinstance(self.model.input_notation, MandelNotation):
            x = self.notation.to_full(x)
            x = self.model.input_notation.to_reduced(x)

        x = self.model(x)

        if not isinstance(self.model.input_notation, MandelNotation):
            x = self.model.input_notation.to_full(x)
            x = self.notation.to_reduced(x)

        x = self.rotation(x, transpose=False)

        if not isinstance(self.model.input_notation, MandelNotation):
            x = self.notation.to_full(x)
            x = self.model.input_notation.to_reduced(x)

        return x


@dataclasses.dataclass(kw_only=True)
class Network(nn.Module):
    layer_size: int
    layer_count: int
    output_size: int
    activation: nn.Module | ActivationType
    output_activation: nn.Module | ActivationType | None = None
    kernel_initializer: jax.nn.initializers.Initializer | InitializerType = (
        nn.initializers.lecun_normal()
    )
    bias_initializer: jax.nn.initializers.Initializer | InitializerType = (
        nn.initializers.zeros
    )

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.activation, ActivationType):
            self.activation = ActivationType.create(self.activation)
        if isinstance(self.output_activation, ActivationType):
            self.output_activation = ActivationType.create(self.output_activation)
        if isinstance(self.kernel_initializer, InitializerType):
            self.kernel_initializer = InitializerType.create(self.kernel_initializer)
        if isinstance(self.bias_initializer, InitializerType):
            self.bias_initializer = InitializerType.create(self.bias_initializer)

    @abstractmethod
    @nn.compact
    def __call__(self, x: ArrayLike, train: bool = False) -> ArrayLike:
        return


@dataclasses.dataclass(kw_only=True)
class TensorFeaturesNetwork(Network):
    dim: int = 3
    sym_cls_type: TensorSymmetryClassType = TensorSymmetryClassType.NONE
    input_notation_type: SymmetricTensorNotationType = (
        SymmetricTensorNotationType.MANDEL
    )
    output_notation_type: SymmetricTensorNotationType = (
        SymmetricTensorNotationType.MANDEL
    )

    def __post_init__(self):
        super().__post_init__()

    def setup(self):
        self.input_notation = self.input_notation_type.create(order=2, dim=self.dim)
        self.output_notation = self.output_notation_type.create(order=2, dim=self.dim)
        self.feature_notation = SymmetricTensorNotationType.MANDEL.create(
            order=2, dim=self.dim
        )
        self.kernel_rep = SymmetricTensorRepresentation(
            order=4,
            dim=self.dim,
            notation_type=SymmetricTensorNotationType.MANDEL,
            sym_cls_type=self.sym_cls_type,
        )
        self.bias_rep = SymmetricTensorRepresentation(
            order=2,
            dim=self.dim,
            notation_type=SymmetricTensorNotationType.MANDEL,
            sym_cls_type=self.sym_cls_type,
        )


@dataclasses.dataclass(kw_only=True)
class MLPScalarFeaturesNetwork(Network):
    @nn.compact
    def __call__(self, x: ArrayLike, train: bool = False) -> ArrayLike:
        for k in range(self.layer_count):
            x = nn.Dense(
                self.layer_size,
                kernel_init=self.kernel_initializer,
                bias_init=self.bias_initializer,
            )(x)
            x = self.activation(x)
        x = nn.Dense(
            self.output_size,
            kernel_init=self.kernel_initializer,
            bias_init=self.bias_initializer,
        )(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


@dataclasses.dataclass(kw_only=True)
class MLPTensorFeaturesNetwork(TensorFeaturesNetwork):
    @nn.compact
    def __call__(
        self,
        x: ArrayLike,
        train: bool = False,
        tensor_basis: ArrayLike | None = None,
    ) -> ArrayLike:
        if not isinstance(self.input_notation, MandelNotation):
            x = self.input_notation.to_full(x)
            x = self.feature_notation.to_reduced(x)
        for k in range(self.layer_count):
            x = DenseSymmetricTensor(
                kernel_rep=self.kernel_rep,
                bias_rep=self.bias_rep,
                features=self.layer_size,
                kernel_init=self.kernel_initializer,
                bias_init=self.bias_initializer,
            )(x, tensor_basis=tensor_basis)
            x = TensorActivation(self.activation, self.feature_notation)(x)
        x = DenseSymmetricTensor(
            kernel_rep=self.kernel_rep,
            bias_rep=self.bias_rep,
            features=self.output_size,
            kernel_init=self.kernel_initializer,
            bias_init=self.bias_initializer,
        )(x, tensor_basis=tensor_basis)
        if self.output_activation is not None:
            x = TensorActivation(self.output_activation, self.feature_notation)(x)
        if not isinstance(self.output_notation, MandelNotation):
            x = self.feature_notation.to_full(x)
            x = self.output_notation.to_reduced(x)
        return x


@dataclasses.dataclass(kw_only=True)
class GRUScalarFeaturesNetwork(Network):
    @nn.compact
    def __call__(self, x: ArrayLike, train: bool = False) -> ArrayLike:
        for k in range(self.layer_count):
            x = nn.RNN(
                nn.GRUCell(
                    self.layer_size,
                    kernel_init=self.kernel_initializer,
                    bias_init=self.bias_initializer,
                    activation_fn=self.activation,
                    param_dtype=float,
                )
            )(x)
        x = nn.Dense(
            self.output_size,
            kernel_init=self.kernel_initializer,
            bias_init=self.bias_initializer,
        )(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


@dataclasses.dataclass(kw_only=True)
class GRUTensorFeaturesNetwork(TensorFeaturesNetwork):
    @nn.compact
    def __call__(
        self,
        x: ArrayLike,
        train: bool = False,
        tensor_basis: ArrayLike | None = None,
    ) -> ArrayLike:
        if not isinstance(self.input_notation, MandelNotation):
            x = self.input_notation.to_full(x)
            x = self.feature_notation.to_reduced(x)

        for k in range(self.layer_count):
            x = nn.RNN(
                GRUCellSymmetricTensor(
                    kernel_rep=self.kernel_rep,
                    bias_rep=self.bias_rep,
                    features=self.layer_size,
                    kernel_init=self.kernel_initializer,
                    bias_init=self.bias_initializer,
                    activation_fn=self.activation,
                )
            )(x)

        x = DenseSymmetricTensor(
            kernel_rep=self.kernel_rep,
            bias_rep=self.bias_rep,
            features=self.output_size,
            kernel_init=self.kernel_initializer,
            bias_init=self.bias_initializer,
        )(x, tensor_basis=tensor_basis)

        if self.output_activation is not None:
            x = TensorActivation(self.feature_notation, self.output_activation)(x)
        if not isinstance(self.output_notation, MandelNotation):
            x = self.feature_notation.to_full(x)
            x = self.output_notation.to_reduced(x)
        return x


@dataclasses.dataclass(kw_only=True)
class LSTMScalarFeaturesNetwork(Network):
    @nn.compact
    def __call__(self, x: ArrayLike, train: bool = False) -> ArrayLike:
        for k in range(self.layer_count):
            x = nn.RNN(
                nn.LSTMCell(
                    self.layer_size,
                    kernel_init=self.kernel_initializer,
                    bias_init=self.bias_initializer,
                    activation_fn=self.activation,
                    param_dtype=float,
                )
            )(x)
        x = nn.Dense(
            self.output_size,
            kernel_init=self.kernel_initializer,
            bias_init=self.bias_initializer,
        )(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


@dataclasses.dataclass(kw_only=True)
class LSTMTensorFeaturesNetwork(TensorFeaturesNetwork):
    def setup(self) -> None:
        raise NotImplementedError

    @nn.compact
    def __call__(
        self,
        x: ArrayLike,
        train: bool = False,
        tensor_basis: ArrayLike | None = None,
    ) -> ArrayLike:
        if not isinstance(self.input_notation, MandelNotation):
            x = self.input_notation.to_full(x)
            x = self.feature_notation.to_reduced(x)

        for k in range(self.layer_count):
            x = nn.RNN(
                LSTMCellSymmetricTensor(
                    kernel_rep=self.kernel_rep,
                    bias_rep=self.bias_rep,
                    features=self.layer_size,
                    kernel_init=self.kernel_initializer,
                    bias_init=self.bias_initializer,
                    activation_fn=self.activation,
                )
            )(x)

        x = DenseSymmetricTensor(
            kernel_rep=self.kernel_rep,
            bias_rep=self.bias_rep,
            features=self.output_size,
            kernel_init=self.kernel_initializer,
            bias_init=self.bias_initializer,
        )(x, tensor_basis=tensor_basis)

        if self.output_activation is not None:
            x = TensorActivation(self.feature_notation, self.output_activation)(x)
        if not isinstance(self.output_notation, MandelNotation):
            x = self.feature_notation.to_full(x)
            x = self.output_notation.to_reduced(x)
        return x
