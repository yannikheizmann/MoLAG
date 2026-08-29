import math

import numpy as np
from torch_geometric.data import Batch

from molag.model.gnn.blocks import upper_tri_mask

from molag.evaluation._assessment import PartitionAssessment

from ._base import MetricsBase


class PartitionMetrics(MetricsBase):
    """Accumulate component-level correctness diagnostics by scene."""

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must lie strictly between 0 and 1")
        self._logit_threshold = math.log(threshold / (1 - threshold))
        self.reset()

    def reset(self) -> None:
        self._scenes = 0
        self._correct = 0
        self._real_merges = 0
        self._real_splits = 0
        self._spurious_bridges = 0

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

    def compute(self) -> dict[str, float]:
        if self._scenes == 0:
            return {}
        return {
            "partition_accuracy": self._correct / self._scenes,
            "real_merge_rate": self._real_merges / self._scenes,
            "real_split_rate": self._real_splits / self._scenes,
            "spurious_bridge_rate": self._spurious_bridges / self._scenes,
        }
