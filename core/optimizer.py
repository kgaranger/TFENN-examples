import optax
from TFENN.util.enum_input import EnumInputClass


class OptimizerType(EnumInputClass):
    ADAM = "adam"
    ADAMW = "adamw"

    @classmethod
    @property
    def obj_map(cls) -> dict["OptimizerType", optax.GradientTransformation]:
        return {
            OptimizerType.ADAM: optax.adam,
            OptimizerType.ADAMW: optax.adamw,
        }
