from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .metrics import MetricsBase
from molag.inference import PredictionCache


@dataclass(frozen=True)
class CalibrationResult:
    """Selected threshold and complete metrics across the searched grid."""

    threshold: float
    objective: str
    objective_value: float
    metrics_by_threshold: dict[float, dict[str, float]]


class ThresholdCalibrator:
    """Select an affinity threshold from one pass over frozen scenes."""

    def __init__(
        self,
        metric_factory: Callable[[float], MetricsBase],
        objective: str,
        thresholds: list[float],
    ) -> None:
        if not thresholds:
            raise ValueError("thresholds must contain at least one value")
        self._objective = objective
        self._metrics = {
            threshold: metric_factory(threshold) for threshold in thresholds
        }

    def calibrate(self, predictions: PredictionCache) -> CalibrationResult:
        """Select a threshold entirely from reusable cached predictions."""
        for metric in self._metrics.values():
            metric.reset()
        for scene in predictions:
            values = {
                "logits": scene.edge_logits,
                "labels": scene.edge_labels,
                "tracker_labels": scene.point_labels[:, 0],
                "edge_index": scene.edge_index,
                "coordinates": scene.coordinates,
            }
            for metric in self._metrics.values():
                metric.update(**values)

        metrics_by_threshold: dict[float, dict[str, float]] = {}
        for threshold, metric in self._metrics.items():
            values = metric.compute()
            if self._objective not in values:
                available = ", ".join(sorted(values)) or "none"
                raise ValueError(
                    f"metric does not provide objective {self._objective!r}; "
                    f"available values: {available}"
                )
            metrics_by_threshold[threshold] = values
        threshold = max(
            metrics_by_threshold,
            key=lambda candidate: (
                metrics_by_threshold[candidate][self._objective],
                candidate,
            ),
        )
        return CalibrationResult(
            threshold=threshold,
            objective=self._objective,
            objective_value=metrics_by_threshold[threshold][self._objective],
            metrics_by_threshold=metrics_by_threshold,
        )
