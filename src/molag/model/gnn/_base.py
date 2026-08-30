"""Abstract interface for graph-neural-network models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch_geometric.data import Batch, Data

from molag.model._base import ModelBase


class GraphNeuralNetworkBase(ModelBase, ABC):
    """Graph neural network operating on PyTorch Geometric data."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError("in_dim must be positive")
        self.in_dim = in_dim

    @staticmethod
    def ensure_batch(data: Data | Batch) -> Batch:
        """Wrap a single graph in a batch and preserve existing batches."""
        return data if isinstance(data, Batch) else Batch.from_data_list([data])

    @abstractmethod
    def forward(self, data: Data | Batch, **kwargs: Any) -> dict[str, Any]:
        """Run the graph network and return its named outputs."""
