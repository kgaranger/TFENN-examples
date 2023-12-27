import optax
from TFENN.util.enum_input import EnumInputClass


class SchedulerType(EnumInputClass):
    NONE = None
    COC = "coc"

    @classmethod
    @property
    def obj_map(cls) -> dict["SchedulerType", optax.Schedule]:
        return {
            SchedulerType.NONE: optax.constant_schedule,
            SchedulerType.COC: optax.cosine_onecycle_schedule,
        }
