"""Assess connected-component predictions against physical tracker labels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from molag.inference import AffinityPartition


class GroupingFailureMode(StrEnum):
    """Mutually exclusive scene-level grouping outcome."""

    CORRECT = "correct"
    FALSE_MERGE = "false_merge"
    FALSE_SPLIT = "false_split"
    SPURIOUS_BRIDGE = "spurious_bridge"
    MIXED = "mixed"


@dataclass(frozen=True)
class TrackerAssessment:
    """Recovery diagnostics for one real tracker in a predicted partition."""

    tracker_id: int
    n_leds: int
    correct: bool
    has_merge: bool
    has_split: bool
    component_id: int | None

    @property
    def failure_mode(self) -> GroupingFailureMode:
        """Classify this tracker's mutually exclusive grouping outcome."""
        if self.correct:
            return GroupingFailureMode.CORRECT
        if self.has_merge and self.has_split:
            return GroupingFailureMode.MIXED
        if self.has_merge:
            return GroupingFailureMode.FALSE_MERGE
        return GroupingFailureMode.FALSE_SPLIT


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
    trackers: tuple[TrackerAssessment, ...]
    n_spurious: int

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
        trackers = cls._assess_trackers(labels, partition)
        return cls(
            correct=not has_merge and not has_split,
            real_only_correct=not real_only_merge and not real_only_split,
            has_real_merge=has_merge,
            has_real_split=has_split,
            spurious_bridge=has_merge and not real_only_merge,
            partition=partition,
            real_only_partition=real_only_partition,
            trackers=trackers,
            n_spurious=int((labels < 0).sum()),
        )

    @property
    def failure_mode(self) -> GroupingFailureMode:
        """Classify the scene's mutually exclusive grouping outcome."""
        if self.correct:
            return GroupingFailureMode.CORRECT
        if self.has_real_merge and self.has_real_split:
            return GroupingFailureMode.MIXED
        if self.spurious_bridge:
            return GroupingFailureMode.SPURIOUS_BRIDGE
        if self.has_real_merge:
            return GroupingFailureMode.FALSE_MERGE
        return GroupingFailureMode.FALSE_SPLIT

    @property
    def n_trackers(self) -> int:
        """Return the number of real trackers in the scene."""
        return len(self.trackers)

    @property
    def n_trackers_correct(self) -> int:
        """Return the number of intact, unmerged real trackers."""
        return sum(int(tracker.correct) for tracker in self.trackers)

    def complete_tracker_counts(self, num_leds: int) -> tuple[int, int, int]:
        """Return complete, strictly recovered, and extractable tracker counts."""
        if num_leds < 1:
            raise ValueError("num_leds must be positive")
        complete = [tracker for tracker in self.trackers if tracker.n_leds == num_leds]
        recovered = sum(int(tracker.correct) for tracker in complete)
        extractable = len(
            {
                tracker.component_id
                for tracker in complete
                if tracker.component_id is not None
            }
        )
        return len(complete), recovered, extractable

    @staticmethod
    def _assess_trackers(
        labels: np.ndarray,
        partition: AffinityPartition,
    ) -> tuple[TrackerAssessment, ...]:
        real = labels >= 0
        assessments: list[TrackerAssessment] = []
        for tracker_id in np.unique(labels[real]):
            nodes = np.flatnonzero(labels == tracker_id)
            component_ids = np.unique(partition.component_ids[nodes])
            has_split = component_ids.size > 1
            has_merge = any(
                np.unique(
                    labels[(partition.component_ids == component_id) & real]
                ).size
                > 1
                for component_id in component_ids
            )
            assessments.append(
                TrackerAssessment(
                    tracker_id=int(tracker_id),
                    n_leds=int(nodes.size),
                    correct=not has_split and not has_merge,
                    has_merge=has_merge,
                    has_split=has_split,
                    component_id=(
                        None if has_split else int(component_ids.item())
                    ),
                )
            )
        return tuple(assessments)

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
