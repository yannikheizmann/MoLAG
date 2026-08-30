"""Complete lazy context for the scaled-conjunction affinity objective."""

from functools import cached_property
from typing import Any

from .._categories import EdgeCategories
from .._spanning_tree import SpanningForest, maximum_spanning_forest
from ._base import AffinityLossContextBase


class FullAffinityLossContext(AffinityLossContextBase):
    """Lazy context containing every derived affinity-loss structure."""

    def __init__(self, *, max_tracker_nodes: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if max_tracker_nodes < 1:
            raise ValueError("max_tracker_nodes must be positive")
        self.max_tracker_nodes = max_tracker_nodes

    @cached_property
    def edge_categories(self) -> EdgeCategories:
        """Categorise edges once and cache the resulting masks."""
        return EdgeCategories.from_graph(self.edge_index, self.tracker_labels)

    @cached_property
    def spanning_forest(self) -> SpanningForest:
        """Construct maximum spanning trees once and cache the result."""
        return maximum_spanning_forest(
            self.edge_logits,
            self.edge_index,
            self.tracker_labels,
            self.batch_vec,
            self.edge_categories.same_real,
            max_tracker_nodes=self.max_tracker_nodes,
        )
