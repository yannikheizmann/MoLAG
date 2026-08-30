from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from molag.config import CALIBRATION_RESULT_FILENAME
from molag.dataset import PyGTrackingAffinityCollator
from molag.model.gnn.blocks import upper_tri_mask

from ._partition import AffinityPartition


@dataclass(frozen=True)
class InferenceResult:
    """Affinity predictions and resulting candidate tracker groups."""

    edge_index: np.ndarray
    affinities: np.ndarray
    partition: AffinityPartition

    @property
    def groups(self) -> tuple[np.ndarray, ...]:
        return self.partition.groups


class MoLAGPredictor:
    """Run calibrated MoLAG inference on one coordinate set."""

    def __init__(
        self,
        model,
        threshold: float,
        device: str | torch.device = "cpu",
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must lie strictly between 0 and 1")
        self._model = model
        self._threshold = threshold
        self._device = torch.device(device)

    @classmethod
    def from_run_directory(
        cls,
        run_directory: str | Path,
        device: str | torch.device = "cpu",
    ) -> MoLAGPredictor:
        """Load a model and its calibrated threshold from one run directory."""
        from molag.evaluation import ModelLoader

        run = Path(run_directory)
        calibration = run / CALIBRATION_RESULT_FILENAME
        if not calibration.is_file():
            raise FileNotFoundError(f"calibration result not found: {calibration}")
        payload = json.loads(calibration.read_text())
        threshold = payload.get("threshold") if isinstance(payload, dict) else None
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 < threshold < 1
        ):
            raise ValueError(
                "calibration result must contain a threshold strictly between 0 and 1"
            )
        return cls(
            model=ModelLoader.from_run_directory(run, device),
            threshold=float(threshold),
            device=device,
        )

    def predict(self, coordinates: Tensor | np.ndarray) -> InferenceResult:
        """Predict affinities and candidate groups for localised image points."""
        tensor = torch.as_tensor(coordinates, dtype=torch.float32)
        tensor = self._normalize_coordinates(tensor)
        graph = PyGTrackingAffinityCollator.build_graph(tensor).to(self._device)
        self._model.to(self._device)
        self._model.eval()
        with torch.inference_mode():
            logits = self._model(data=graph)["edge_logits"]

        pair_edges = graph.edge_index[:, upper_tri_mask(graph.edge_index)]
        affinities = logits.sigmoid().detach().cpu().numpy()
        edge_index = pair_edges.detach().cpu().numpy()
        partition = AffinityPartition.from_graph(
            n_nodes=tensor.shape[0],
            edge_index=edge_index,
            positive_edges=affinities >= self._threshold,
        )
        return InferenceResult(
            edge_index=edge_index,
            affinities=affinities,
            partition=partition,
        )

    @staticmethod
    def _normalize_coordinates(coordinates: Tensor) -> Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape (num_nodes, 2)")
        if coordinates.shape[0] == 0:
            raise ValueError("coordinates must contain at least one node")
        if not torch.isfinite(coordinates).all():
            raise ValueError("coordinates must contain only finite values")
        centered = coordinates - coordinates.mean(dim=0, keepdim=True)
        scale = torch.linalg.vector_norm(centered, dim=1).max()
        return centered / scale if scale > 0 else centered
