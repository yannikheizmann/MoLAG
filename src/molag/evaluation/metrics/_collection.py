from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._base import MetricsBase


class CombinedMetrics(MetricsBase):
    """Update multiple streaming metric implementations in one pass."""

    def __init__(self, metrics: Sequence[MetricsBase]) -> None:
        if not metrics:
            raise ValueError("metrics must contain at least one implementation")
        self._metrics = list(metrics)

    def reset(self) -> None:
        for metric in self._metrics:
            metric.reset()

    def update(self, **values: Any) -> None:
        for metric in self._metrics:
            metric.update(**values)

    def compute(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for metric in self._metrics:
            values = metric.compute()
            duplicates = result.keys() & values.keys()
            if duplicates:
                names = ", ".join(sorted(duplicates))
                raise ValueError(f"duplicate metric names: {names}")
            result.update(values)
        return result
