import logging
import pickle
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable


class MetricType(Enum):
    EPOCH = "epoch"
    TRAINING_LOSS = "training_loss"
    VALIDATION_LOSS = "validation_loss"
    UNSCALED_VALIDATION_LOSS = "unscaled_validation_loss"
    TRAINING_LIKE_VALIDATION_LOSS = "training_like_validation_loss"
    PARAMETERS_NORM = "parameters_norm"
    GRADIENTS_NORM = "gradients_norm"
    UPDATES_NORM = "updates_norm"
    TIME = "time"

    @classmethod
    @property
    def format_str(cls) -> dict["MetricType", str]:
        """Formatting specifier for each metric type."""
        return {
            cls.EPOCH: "d",
            cls.TRAINING_LOSS: ".3e",
            cls.VALIDATION_LOSS: ".3e",
            cls.UNSCALED_VALIDATION_LOSS: ".3e",
            cls.TRAINING_LIKE_VALIDATION_LOSS: ".3e",
            cls.PARAMETERS_NORM: ".3e",
            cls.GRADIENTS_NORM: ".3e",
            cls.UPDATES_NORM: ".3e",
            cls.TIME: ".2f",
        }

    @classmethod
    @property
    def can_show_best(cls) -> dict["MetricType", Callable[[float, float], bool]]:
        """Whether the metric can show a best value and how to compare them."""
        return {
            cls.TRAINING_LOSS: lambda x, y: x < y,
            cls.VALIDATION_LOSS: lambda x, y: x < y,
            cls.UNSCALED_VALIDATION_LOSS: lambda x, y: x < y,
            cls.TRAINING_LIKE_VALIDATION_LOSS: lambda x, y: x < y,
        }

    def pretty(self) -> str:
        """Pretty name of the metric."""
        return self.value.replace("_", " ").capitalize()


class Metrics:
    def __init__(
        self,
        save_path: Path = None,
        print_interval: int = 0,
        save_interval: int = 0,
        print_fn: Callable = logging.info,
        print_metrics: Iterable[MetricType] = MetricType,
    ):
        self.save_path = save_path
        self.print_interval = print_interval
        self.save_interval = save_interval
        self.print_fn = print_fn
        self.print_metrics = sorted(
            print_metrics, key=lambda x: {t: k for k, t in enumerate(MetricType)}[x]
        )

        self.metrics = []
        self.metrics_best = {
            metric_type: None
            for metric_type in self.print_metrics
            if metric_type in MetricType.can_show_best
        }
        self.logger = logging.getLogger(__name__)

        self.last_print = -print_interval
        self.last_save = -save_interval

    def update(self, epoch: int, **kwargs):
        self.metrics.append(
            {MetricType.EPOCH.value: epoch, **{k: v for k, v in kwargs.items()}}
        )
        for metric_type_val in kwargs:
            if MetricType.can_show_best.get(MetricType(metric_type_val)):
                if self.metrics_best[
                    MetricType(metric_type_val)
                ] is None or MetricType.can_show_best[MetricType(metric_type_val)](
                    kwargs[metric_type_val],
                    self.metrics_best[MetricType(metric_type_val)],
                ):
                    self.metrics_best[MetricType(metric_type_val)] = kwargs[
                        metric_type_val
                    ]

        if (
            self.print_interval > 0
            and len(self.metrics) - self.last_print >= self.print_interval
        ):
            self.print_last()
            self.last_print = len(self.metrics)

        if (
            self.save_interval > 0
            and len(self.metrics) - self.last_save >= self.save_interval
        ):
            self.save()
            self.last_save = len(self.metrics)

    def print_last(self):
        if len(self.metrics) > 0:
            try:
                self.print_fn(
                    "\t".join(
                        [
                            f"{metric_type.pretty()}: "
                            f"{self.metrics[-1][metric_type.value]:{MetricType.format_str[metric_type]}}"
                            + (
                                f" (best: {self.metrics_best[metric_type]:{MetricType.format_str[metric_type]}})"
                                if metric_type in MetricType.can_show_best
                                else ""
                            )
                            for metric_type in self.print_metrics
                        ]
                    )
                )
            except KeyError as e:
                self.logger.warning(
                    f"{e} not present in provided metrics during last update"
                )

    def save(self):
        if self.save_path is not None:
            with open(self.save_path, "ab") as f:
                for metric in self.metrics[self.last_save :]:
                    pickle.dump(metric, f)
        else:
            self.logger.warning("No save path specified")

    def clear(self):
        self.metrics = []
        self.last_print = -self.print_interval
        self.last_save = -self.save_interval

        if self.save_path is not None and self.save_path.exists():
            self.save_path.unlink()
