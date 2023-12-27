from flax import linen as nn
from TFENN.util.enum_input import EnumInputFun


class ActivationType(EnumInputFun):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    TANH = "tanh"
    LOGISTIC = "logistic"
    NONE = "none"

    @classmethod
    @property
    def fun_map(cls) -> dict["ActivationType", nn.Module]:
        return {
            ActivationType.RELU: nn.activation.relu,
            ActivationType.LEAKY_RELU: nn.activation.leaky_relu,
            ActivationType.TANH: nn.activation.tanh,
            ActivationType.LOGISTIC: nn.activation.sigmoid,
            ActivationType.NONE: lambda x: x,
        }
