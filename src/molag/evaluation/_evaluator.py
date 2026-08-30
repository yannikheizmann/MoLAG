from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from molag.model.gnn.blocks import upper_tri_mask

from .metrics import MetricsBase
from ._predictions import PredictionCache, ScenePrediction


class Evaluator:
    """Run batched model inference with modular streaming metrics."""

    def __init__(
        self,
        model,
        dataset: Dataset,
        data_collator: Callable,
        metrics: MetricsBase,
        batch_size: int,
        device: str | torch.device,
        dataloader_num_workers: int = 0,
    ) -> None:
        self._model = model
        self._metrics = metrics
        self._device = torch.device(device)
        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
            collate_fn=data_collator,
        )

    def predict(self) -> PredictionCache:
        """Run the model once and retain raw outputs separately for every scene."""
        self._model.to(self._device)
        self._model.eval()
        scenes: list[ScenePrediction] = []
        with torch.inference_mode():
            for inputs in self._loader:
                data = inputs["data"].to(self._device)
                outputs: dict[str, Any] = self._model(data=data)
                scenes.extend(
                    self._split_batch(
                        inputs,
                        outputs["edge_logits"].detach().float().cpu().numpy(),
                    )
                )
        return PredictionCache(scenes)

    def evaluate(
        self,
        predictions: PredictionCache | None = None,
    ) -> dict[str, float]:
        """Compute configured metrics from new or previously cached predictions."""
        cache = predictions if predictions is not None else self.predict()
        return self.evaluate_predictions(cache, self._metrics)

    @staticmethod
    def evaluate_predictions(
        predictions: PredictionCache,
        metrics: MetricsBase,
    ) -> dict[str, float]:
        """Compute metrics from cached predictions without loading a model."""
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

    @staticmethod
    def _split_batch(inputs: dict[str, Any], logits) -> list[ScenePrediction]:
        data = inputs["data"].cpu()
        pair_edges = data.edge_index[:, upper_tri_mask(data.edge_index)].numpy()
        pointers = data.ptr.numpy()
        tracker_labels = inputs["tracker_labels"].cpu().numpy()
        led_labels = inputs["led_labels"].cpu().numpy()
        edge_labels = inputs["edge_labels"].cpu().numpy()
        if pair_edges.shape[1] != len(logits):
            raise ValueError("logits must match the unordered graph edges")

        scenes = []
        for index in range(data.num_graphs):
            node_start, node_stop = (int(value) for value in pointers[index : index + 2])
            edge_mask = (pair_edges[0] >= node_start) & (pair_edges[0] < node_stop)
            scenes.append(
                ScenePrediction(
                    coordinates=data.x[node_start:node_stop].numpy().copy(),
                    point_labels=np.column_stack(
                        (
                            tracker_labels[node_start:node_stop],
                            led_labels[node_start:node_stop],
                        )
                    ),
                    edge_index=pair_edges[:, edge_mask] - node_start,
                    edge_logits=np.asarray(logits[edge_mask], dtype=np.float32),
                    edge_labels=np.asarray(edge_labels[edge_mask], dtype=np.int64),
                )
            )
        return scenes

    def breakdown(self) -> dict[str, Any]:
        """Return structured diagnostics from the latest evaluation pass."""
        return self._metrics.breakdown()

    def sample_records(self) -> list[dict[str, Any]]:
        """Return scene-level records from the latest evaluation pass."""
        return self._metrics.sample_records()

    def tracker_records(self) -> list[dict[str, Any]]:
        """Return tracker-level records from the latest evaluation pass."""
        return self._metrics.tracker_records()
