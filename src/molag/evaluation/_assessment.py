from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from molag.inference import AffinityPartition


@dataclass(frozen=True)
class PartitionAssessment:
    """Ground-truth diagnostics for an affinity partition."""

    correct: bool
    real_only_correct: bool
    has_real_merge: bool
    has_real_split: bool
    spurious_bridge: bool
    partition: AffinityPartition
    real_only_partition: AffinityPartition

    @classmethod
    def from_graph(
        cls,
        tracker_labels: np.ndarray,
        edge_index: np.ndarray,
        positive_edges: np.ndarray,
    ) -> PartitionAssessment:
        """Evaluate thresholded affinity edges against tracker labels."""
        labels = np.asarray(tracker_labels, dtype=np.int64).reshape(-1)
        edges = np.asarray(edge_index, dtype=np.int64)
        positive = np.asarray(positive_edges, dtype=bool).reshape(-1)
        partition = AffinityPartition.from_graph(labels.size, edges, positive)
        has_merge, has_split = cls._real_failures(labels, partition)

        source, destination = edges
        real_edges = (labels[source] >= 0) & (labels[destination] >= 0)
        real_only_partition = AffinityPartition.from_graph(
            labels.size,
            edges[:, real_edges],
            positive[real_edges],
        )
        real_only_merge, real_only_split = cls._real_failures(
            labels,
            real_only_partition,
        )
        return cls(
            correct=not has_merge and not has_split,
            real_only_correct=not real_only_merge and not real_only_split,
            has_real_merge=has_merge,
            has_real_split=has_split,
            spurious_bridge=has_merge and not real_only_merge,
            partition=partition,
            real_only_partition=real_only_partition,
        )

    @staticmethod
    def _real_failures(
        labels: np.ndarray,
        partition: AffinityPartition,
    ) -> tuple[bool, bool]:
        real = labels >= 0
        has_merge = any(
            np.unique(labels[group][real[group]]).size > 1
            for group in partition.groups
        )
        has_split = any(
            np.unique(partition.component_ids[labels == tracker_id]).size > 1
            for tracker_id in np.unique(labels[real])
        )
        return has_merge, has_split
