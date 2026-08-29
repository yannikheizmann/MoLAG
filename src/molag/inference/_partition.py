from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffinityPartition:
    """Connected-component partition of an affinity graph."""

    component_ids: np.ndarray

    @classmethod
    def from_graph(
        cls,
        n_nodes: int,
        edge_index: np.ndarray,
        positive_edges: np.ndarray,
    ) -> AffinityPartition:
        """Partition nodes using the active undirected affinity edges."""
        if n_nodes < 0:
            raise ValueError("n_nodes must not be negative")
        edges = np.asarray(edge_index, dtype=np.int64)
        positive = np.asarray(positive_edges, dtype=bool).reshape(-1)
        if edges.shape != (2, positive.size):
            raise ValueError(
                f"edge_index shape {edges.shape} does not align with "
                f"{positive.size} edges"
            )
        if edges.size and (edges.min() < 0 or edges.max() >= n_nodes):
            raise ValueError("edge_index contains a node outside the graph")

        adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
        for source, target, active in zip(
            edges[0].tolist(),
            edges[1].tolist(),
            positive.tolist(),
        ):
            if active:
                adjacency[source].append(target)
                adjacency[target].append(source)

        assignments = np.full(n_nodes, -1, dtype=np.int64)
        component_id = 0
        for start in range(n_nodes):
            if assignments[start] >= 0:
                continue
            assignments[start] = component_id
            queue = [start]
            for node in queue:
                for neighbor in adjacency[node]:
                    if assignments[neighbor] < 0:
                        assignments[neighbor] = component_id
                        queue.append(neighbor)
            component_id += 1
        return cls(component_ids=assignments)

    @property
    def groups(self) -> tuple[np.ndarray, ...]:
        """Return node indices grouped by connected component."""
        return tuple(
            np.flatnonzero(self.component_ids == component_id)
            for component_id in np.unique(self.component_ids)
        )
