"""Evaluate raw MoLAG predictions with composable streaming metrics."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from molag.inference import PredictionCache, PredictionGenerator

from .metrics import MetricsBase


class Evaluator:
    """Batched prediction and metric evaluation for a fixed dataset."""

    def __init__(
        self,
        model,
        dataset: Dataset,
        data_collator,
        metrics: MetricsBase,
        batch_size: int,
        device: str | torch.device,
        dataloader_num_workers: int = 0,
    ) -> None:
        self._metrics = metrics
        self._generator = PredictionGenerator(
            model=model,
            dataset=dataset,
            data_collator=data_collator,
            batch_size=batch_size,
            device=device,
            dataloader_num_workers=dataloader_num_workers,
        )

    def predict(self) -> PredictionCache:
        """Run inference once and retain the raw output for each scene."""
        return self._generator.predict()

    def evaluate(
        self,
        predictions: PredictionCache | None = None,
    ) -> dict[str, float]:
        """Compute the configured metrics from new or cached predictions."""
        cache = predictions if predictions is not None else self.predict()
        return self.evaluate_predictions(cache, self._metrics)

    @staticmethod
    def evaluate_predictions(
        predictions: PredictionCache,
        metrics: MetricsBase,
    ) -> dict[str, float]:
        """Compute metrics from cached predictions without loading a model.

        The metric state is reset before the cache is consumed. The supplied metric
        object retains its detailed records after this method returns.
        """
        metrics.reset()
        for scene in predictions:
            metrics.update(
                logits=scene.edge_logits,
                labels=scene.edge_labels,
                tracker_labels=scene.point_labels[:, 0],
                edge_index=scene.edge_index,
                coordinates=scene.coordinates,
            )
        return metrics.compute()

    def breakdown(self) -> dict[str, Any]:
        """Return structured diagnostics from the latest evaluation pass."""
        return self._metrics.breakdown()

    def sample_records(self) -> list[dict[str, Any]]:
        """Return scene-level records from the latest evaluation pass."""
        return self._metrics.sample_records()

    def tracker_records(self) -> list[dict[str, Any]]:
        """Return tracker-level records from the latest evaluation pass."""
        return self._metrics.tracker_records()
