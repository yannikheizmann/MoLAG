from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from molag.model.gnn.blocks import full_edge_index, upper_tri_mask


class PyGTrackingAffinityCollator:
    """Build complete graphs and same-tracker targets from dataset samples."""

    @staticmethod
    def _get_edge_features(coordinates: Tensor, edge_index: Tensor) -> Tensor:
        source, destination = edge_index
        relative = coordinates[destination] - coordinates[source]
        squared_distance = relative.square().sum(dim=-1)
        distance = squared_distance.sqrt().clamp_min(1e-8)
        direction = relative / distance.unsqueeze(-1)
        return torch.cat((squared_distance.unsqueeze(-1), direction), dim=-1)

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("features must contain at least one sample")

        graphs: list[Data] = []
        tracker_labels_per_graph: list[Tensor] = []
        led_labels_per_graph: list[Tensor] = []

        for sample in features:
            coordinates = sample["x"]
            labels = sample["y"]
            self._validate_sample(coordinates, labels)

            edge_index = full_edge_index(coordinates.shape[0])
            graphs.append(Data(x=coordinates, edge_index=edge_index))
            tracker_labels_per_graph.append(labels[:, 0])
            led_labels_per_graph.append(labels[:, 1])

        batch = Batch.from_data_list(graphs)
        batch.edge_attr = self._get_edge_features(batch.x, batch.edge_index)

        tracker_labels = torch.cat(tracker_labels_per_graph).long()
        led_labels = torch.cat(led_labels_per_graph).long()
        source, destination = batch.edge_index[:, upper_tri_mask(batch.edge_index)]
        edge_labels = (
            (tracker_labels[source] == tracker_labels[destination])
            & (tracker_labels[source] >= 0)
            & (tracker_labels[destination] >= 0)
        ).long()

        return {
            "data": batch,
            "tracker_labels": tracker_labels,
            "led_labels": led_labels,
            "edge_labels": edge_labels,
            "labels": edge_labels,
        }

    @staticmethod
    def _validate_sample(coordinates: Any, labels: Any) -> None:
        if not isinstance(coordinates, Tensor) or not isinstance(labels, Tensor):
            raise TypeError("sample x and y values must be torch tensors")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("sample x must have shape (num_nodes, 2)")
        if labels.ndim != 2 or labels.shape != coordinates.shape:
            raise ValueError("sample y must have shape (num_nodes, 2)")
        if coordinates.shape[0] == 0:
            raise ValueError("samples must contain at least one node")
