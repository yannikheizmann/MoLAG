from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from molag.utils import resolve_device

from .metrics import MetricsBase


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
        model,
        dataset: Dataset,
        data_collator: Callable,
        metric_factory: Callable[[float], MetricsBase],
        objective: str,
        thresholds: list[float],
        batch_size: int,
        device: str | torch.device,
        dataloader_num_workers: int = 0,
    ) -> None:
        if not thresholds:
            raise ValueError("thresholds must contain at least one value")
        self._model = model
        self._device = resolve_device(device)
        self._objective = objective
        self._metrics = {
            threshold: metric_factory(threshold) for threshold in thresholds
        }
        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
            collate_fn=data_collator,
        )

    def calibrate(self) -> CalibrationResult:
        self._model.to(self._device)
        self._model.eval()
        for metric in self._metrics.values():
            metric.reset()

        with torch.inference_mode():
            for inputs in self._loader:
                outputs = self._model(data=inputs["data"].to(self._device))
                values = {
                    "logits": outputs["edge_logits"].detach().float().cpu().numpy(),
                    "labels": inputs["edge_labels"].cpu().numpy(),
                    "inputs": inputs,
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
