"""Compute affinity metrics restricted to pairs of real detections."""

from __future__ import annotations

import numpy as np

from molag.model.gnn.blocks import upper_tri_mask

from ._affinity import AffinityMetrics


class RealAffinityMetrics(AffinityMetrics):
    """Accumulate affinity metrics only for edges joining two real detections."""

    def update(self, **values) -> None:
        """Accumulate logits for edges whose endpoints are both real."""
        logits = np.asarray(values["logits"], dtype=np.float32).reshape(-1)
        labels = np.asarray(values["labels"], dtype=np.int64).reshape(-1)
        inputs = values.get("inputs")
        if inputs is None:
            tracker_labels = np.asarray(values["tracker_labels"], dtype=np.int64)
            edge_index = np.asarray(values["edge_index"], dtype=np.int64)
        else:
            data = inputs["data"]
            edge_index = data.edge_index[
                :, upper_tri_mask(data.edge_index)
            ].cpu().numpy()
            tracker_labels = inputs["tracker_labels"].detach().cpu().numpy()
        if edge_index.shape != (2, logits.size):
            raise ValueError("edge_index must match the unordered affinity logits")
        row, col = edge_index
        real_edges = (tracker_labels[row] >= 0) & (tracker_labels[col] >= 0)
        super().update(logits=logits[real_edges], labels=labels[real_edges])

    def compute(self) -> dict[str, float]:
        """Return real-real edge classification metrics."""
        return {
            name.replace("edge_", "real_real_edge_", 1): value
            for name, value in super().compute().items()
        }
