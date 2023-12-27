from typing import Callable

from jax import numpy as jnp
from TFENN.util.enum_input import EnumInputFun


class LossType(EnumInputFun):
    SE = "se"
    NSE = "nse"

    @classmethod
    @property
    def fun_map(cls) -> dict["LossType", Callable[[jnp.ndarray, jnp.ndarray], float]]:
        return {
            LossType.SE: lambda y_true, y_pred: ((y_true - y_pred) ** 2).sum(),
            LossType.NSE: lambda y_true, y_pred: ((y_true - y_pred) ** 2).sum()
            / jnp.clip((y_true**2).sum(), a_min=1e-3),
        }
