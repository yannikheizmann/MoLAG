from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor

from .._categories import EdgeCategories
from .._spanning_tree import SpanningForest


class AffinityLossContextBase(ABC):
    """Inputs and derived graph structures shared by affinity-loss components."""

    def __init__(
        self,
        *,
        edge_logits: Tensor,
        edge_labels: Tensor,
        node_embeddings: Tensor,
        tracker_labels: Tensor,
        batch_vec: Tensor,
        edge_index: Tensor,
        n_scenes: int | None = None,
    ) -> None:
        self.edge_logits = edge_logits
        self.edge_labels = edge_labels
        self.node_embeddings = node_embeddings
        self.tracker_labels = tracker_labels
        self.batch_vec = batch_vec
        self.edge_index = edge_index
        self.n_scenes = n_scenes if n_scenes is not None else (
            int(batch_vec.max().item()) + 1 if batch_vec.numel() else 0
        )

    @property
    def zero(self) -> Tensor:
        return self.edge_logits.sum() * 0 + self.node_embeddings.sum() * 0

    @property
    @abstractmethod
    def edge_categories(self) -> EdgeCategories: ...

    @property
    @abstractmethod
    def spanning_forest(self) -> SpanningForest: ...
