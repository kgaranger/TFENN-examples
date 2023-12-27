import jax
from flax import linen as nn
from TFENN.util.enum_input import EnumInputClass


class InitializerType(EnumInputClass):
    NORMAL = "normal"
    UNIFORM = "uniform"
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    HE_NORMAL = "he_normal"
    HE_UNIFORM = "he_uniform"
    LE_CUN_NORMAL = "le_cun_normal"
    LE_CUN_UNIFORM = "le_cun_uniform"
    ZERO = "zero"

    @classmethod
    @property
    def obj_map(cls) -> dict["InitializerType", jax.nn.initializers.Initializer]:
        return {
            InitializerType.NORMAL: nn.initializers.normal,
            InitializerType.UNIFORM: nn.initializers.uniform,
            InitializerType.XAVIER_UNIFORM: nn.initializers.xavier_uniform,
            InitializerType.XAVIER_NORMAL: nn.initializers.xavier_normal,
            InitializerType.HE_NORMAL: nn.initializers.he_normal,
            InitializerType.HE_UNIFORM: nn.initializers.he_uniform,
            InitializerType.LE_CUN_NORMAL: nn.initializers.lecun_normal,
            InitializerType.LE_CUN_UNIFORM: nn.initializers.lecun_uniform,
            InitializerType.ZERO: nn.initializers.zeros_init,
        }
