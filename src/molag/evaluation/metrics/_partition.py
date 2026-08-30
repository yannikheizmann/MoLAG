import math
from collections import defaultdict
from typing import Any

import numpy as np
from torch_geometric.data import Batch

from molag.model.gnn.blocks import upper_tri_mask
from molag.utils.statistics import clustered_ratio_interval, wilson_interval

from molag.evaluation._assessment import PartitionAssessment

from ._base import MetricsBase


class PartitionMetrics(MetricsBase):
    """Accumulate component-level correctness diagnostics by scene."""

    def __init__(
        self,
        threshold: float = 0.5,
        complete_tracker_leds: int | None = None,
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must lie strictly between 0 and 1")
        if complete_tracker_leds is None:
            from molag.dataset.tracker import TriangularTracker

            complete_tracker_leds = TriangularTracker.num_leds()
        if complete_tracker_leds < 1:
            raise ValueError("complete_tracker_leds must be positive")
        self._logit_threshold = math.log(threshold / (1 - threshold))
        self._complete_tracker_leds = complete_tracker_leds
        self.reset()

    def reset(self) -> None:
        self._scenes = 0
        self._correct = 0
        self._real_merges = 0
        self._real_splits = 0
        self._spurious_bridges = 0
        self._assessments: list[PartitionAssessment] = []
        self._spurious_attached: list[int] = []

    def update(self, **values) -> None:
        logits = np.asarray(values["logits"], dtype=np.float32).reshape(-1)
        inputs = values.get("inputs")
        if inputs is not None:
            self._update_batch(logits, inputs)
            return
        self._update_scene(
            logits=logits,
            tracker_labels=values["tracker_labels"],
            edge_index=values["edge_index"],
        )

    def _update_batch(self, logits: np.ndarray, inputs) -> None:
        data: Batch = inputs["data"]
        tracker_labels = inputs["tracker_labels"].detach().cpu().numpy()
        pair_edges = data.edge_index[:, upper_tri_mask(data.edge_index)].cpu().numpy()
        batch = data.batch.detach().cpu().numpy()
        pointers = data.ptr.detach().cpu().numpy()
        if pair_edges.shape[1] != logits.size:
            raise ValueError("logits must match the unordered graph edges")

        for scene in range(data.num_graphs):
            start, stop = int(pointers[scene]), int(pointers[scene + 1])
            edge_mask = batch[pair_edges[0]] == scene
            self._update_scene(
                logits=logits[edge_mask],
                tracker_labels=tracker_labels[start:stop],
                edge_index=pair_edges[:, edge_mask] - start,
            )

    def _update_scene(
        self,
        logits: np.ndarray,
        tracker_labels,
        edge_index,
    ) -> None:
        assessment = PartitionAssessment.from_graph(
            tracker_labels=tracker_labels,
            edge_index=edge_index,
            positive_edges=logits >= self._logit_threshold,
        )
        self._scenes += 1
        self._correct += int(assessment.correct)
        self._real_merges += int(assessment.has_real_merge)
        self._real_splits += int(assessment.has_real_split)
        self._spurious_bridges += int(assessment.spurious_bridge)
        self._assessments.append(assessment)
        labels = np.asarray(tracker_labels)
        edges = np.asarray(edge_index)
        positive = np.asarray(logits) >= self._logit_threshold
        row, col = edges[:, positive]
        cross_type = (labels[row] >= 0) != (labels[col] >= 0)
        attached = np.concatenate((row[cross_type], col[cross_type]))
        self._spurious_attached.append(
            int(np.unique(attached[labels[attached] < 0]).size)
        )

    def compute(self) -> dict[str, float]:
        if self._scenes == 0:
            return {}
        real_only_correct = sum(
            int(assessment.real_only_correct) for assessment in self._assessments
        )
        total_trackers = sum(
            assessment.n_trackers for assessment in self._assessments
        )
        recovered_trackers = sum(
            assessment.n_trackers_correct for assessment in self._assessments
        )
        complete_counts = [
            assessment.complete_tracker_counts(self._complete_tracker_leds)
            for assessment in self._assessments
        ]
        total_complete = sum(counts[0] for counts in complete_counts)
        recovered_complete = sum(counts[1] for counts in complete_counts)
        extractable_complete = sum(counts[2] for counts in complete_counts)
        complete_recovery_interval = clustered_ratio_interval(
            [counts[1] for counts in complete_counts],
            [counts[0] for counts in complete_counts],
        )
        complete_extractable_interval = clustered_ratio_interval(
            [counts[2] for counts in complete_counts],
            [counts[0] for counts in complete_counts],
        )
        grouping_interval = wilson_interval(self._correct, self._scenes)
        total_spurious = sum(
            assessment.n_spurious for assessment in self._assessments
        )
        result = {
            "partition_accuracy": self._correct / self._scenes,
            "partition_accuracy_ci95_low": grouping_interval[0],
            "partition_accuracy_ci95_high": grouping_interval[1],
            "partition_accuracy_real_only": real_only_correct / self._scenes,
            "real_merge_rate": self._real_merges / self._scenes,
            "real_split_rate": self._real_splits / self._scenes,
            "spurious_bridge_rate": self._spurious_bridges / self._scenes,
            "spurious_attachment_rate": (
                sum(self._spurious_attached) / total_spurious
                if total_spurious
                else 0.0
            ),
            "tracker_recovery_rate": (
                recovered_trackers / total_trackers if total_trackers else 0.0
            ),
            "complete_tracker_share": (
                total_complete / total_trackers if total_trackers else 0.0
            ),
            "complete_tracker_recovery": (
                recovered_complete / total_complete if total_complete else 0.0
            ),
            "complete_tracker_recovery_ci95_low": complete_recovery_interval[0],
            "complete_tracker_recovery_ci95_high": complete_recovery_interval[1],
            "complete_tracker_extractable": (
                extractable_complete / total_complete if total_complete else 0.0
            ),
            "complete_tracker_extractable_ci95_low": complete_extractable_interval[0],
            "complete_tracker_extractable_ci95_high": complete_extractable_interval[1],
        }
        return result

    def breakdown(self) -> dict[str, Any]:
        by_n_trackers: dict[int, dict[str, int | float]] = defaultdict(
            lambda: {
                "n_scenes": 0,
                "n_correct": 0,
                "n_trackers": 0,
                "n_trackers_correct": 0,
                "n_complete_trackers": 0,
                "n_complete_trackers_correct": 0,
                "n_complete_trackers_extractable": 0,
            }
        )
        by_failure_mode: dict[str, int] = defaultdict(int)
        by_n_spurious: dict[str, dict[str, int | float]] = defaultdict(
            lambda: {
                "n_scenes": 0,
                "n_correct": 0,
                "n_spurious": 0,
                "n_spurious_attached": 0,
            }
        )
        by_visible_leds: dict[int, dict[str, int | float]] = defaultdict(
            lambda: {
                "n_trackers": 0,
                "n_correct": 0,
                "n_false_merge": 0,
                "n_false_split": 0,
                "n_mixed": 0,
            }
        )

        for assessment, n_attached in zip(
            self._assessments,
            self._spurious_attached,
            strict=True,
        ):
            complete, recovered, extractable = assessment.complete_tracker_counts(
                self._complete_tracker_leds
            )
            tracker_bucket = by_n_trackers[assessment.n_trackers]
            tracker_bucket["n_scenes"] += 1
            tracker_bucket["n_correct"] += int(assessment.correct)
            tracker_bucket["n_trackers"] += assessment.n_trackers
            tracker_bucket["n_trackers_correct"] += assessment.n_trackers_correct
            tracker_bucket["n_complete_trackers"] += complete
            tracker_bucket["n_complete_trackers_correct"] += recovered
            tracker_bucket["n_complete_trackers_extractable"] += extractable

            by_failure_mode[assessment.failure_mode.value] += 1
            spurious_key = (
                "0"
                if assessment.n_spurious == 0
                else "1-2"
                if assessment.n_spurious <= 2
                else "3+"
            )
            spurious_bucket = by_n_spurious[spurious_key]
            spurious_bucket["n_scenes"] += 1
            spurious_bucket["n_correct"] += int(assessment.correct)
            spurious_bucket["n_spurious"] += assessment.n_spurious
            spurious_bucket["n_spurious_attached"] += n_attached

            for tracker in assessment.trackers:
                led_bucket = by_visible_leds[tracker.n_leds]
                led_bucket["n_trackers"] += 1
                led_bucket["n_correct"] += int(tracker.correct)
                if not tracker.correct:
                    led_bucket[f"n_{tracker.failure_mode.value}"] += 1

        for bucket in by_n_trackers.values():
            scenes = int(bucket["n_scenes"])
            trackers = int(bucket["n_trackers"])
            complete = int(bucket["n_complete_trackers"])
            bucket["partition_accuracy"] = bucket["n_correct"] / scenes
            bucket["tracker_recovery_rate"] = (
                bucket["n_trackers_correct"] / trackers if trackers else 0.0
            )
            bucket["complete_tracker_recovery"] = (
                bucket["n_complete_trackers_correct"] / complete
                if complete
                else 0.0
            )
            bucket["complete_tracker_extractable"] = (
                bucket["n_complete_trackers_extractable"] / complete
                if complete
                else 0.0
            )
        for bucket in by_n_spurious.values():
            scenes = int(bucket["n_scenes"])
            spurious = int(bucket["n_spurious"])
            bucket["partition_accuracy"] = bucket["n_correct"] / scenes
            bucket["spurious_attachment_rate"] = (
                bucket["n_spurious_attached"] / spurious if spurious else 0.0
            )
        for bucket in by_visible_leds.values():
            trackers = int(bucket["n_trackers"])
            bucket["recovery_rate"] = bucket["n_correct"] / trackers
            for failure in ("false_merge", "false_split", "mixed"):
                bucket[f"{failure}_rate"] = bucket[f"n_{failure}"] / trackers

        empty_spurious = {
            "n_scenes": 0,
            "n_correct": 0,
            "n_spurious": 0,
            "n_spurious_attached": 0,
            "partition_accuracy": 0.0,
            "spurious_attachment_rate": 0.0,
        }
        return {
            "by_n_trackers": {
                str(key): value for key, value in sorted(by_n_trackers.items())
            },
            "by_failure_mode": dict(sorted(by_failure_mode.items())),
            "by_n_spurious": {
                key: by_n_spurious.get(key, empty_spurious.copy())
                for key in ("0", "1-2", "3+")
            },
            "by_visible_leds": {
                str(key): value for key, value in sorted(by_visible_leds.items())
            },
        }
