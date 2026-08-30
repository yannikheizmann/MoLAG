from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from molag.model.gnn.blocks import upper_tri_mask
from molag.utils import resolve_device

from ._predictions import PredictionCache, ScenePrediction


class PredictionGenerator:
    """Run batched model inference and preserve raw outputs per scene."""

    def __init__(
        self,
        model,
        dataset: Dataset,
        data_collator: Callable,
        batch_size: int,
        device: str | torch.device,
        dataloader_num_workers: int = 0,
    ) -> None:
        self._model = model
        self._device = resolve_device(device)
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
            node_start, node_stop = (
                int(value) for value in pointers[index : index + 2]
            )
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
