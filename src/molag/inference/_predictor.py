"""Provide calibrated single-scene inference for localised image points."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from molag.config import CALIBRATION_RESULT_FILENAME
from molag.dataset import PyGTrackingAffinityCollator
from molag.utils import resolve_device

from ._generator import PredictionGenerator
from ._partition import AffinityPartition


@dataclass(frozen=True)
class InferenceResult:
    """Pairwise affinities and their connected-component partition."""

    edge_index: np.ndarray
    affinities: np.ndarray
    partition: AffinityPartition

    @property
    def groups(self) -> tuple[np.ndarray, ...]:
        """Return node indices grouped into candidate trackers."""
        return self.partition.groups


class MoLAGPredictor:
    """Calibrated MoLAG inference for one coordinate set."""

    def __init__(
        self,
        model,
        threshold: float,
        device: str | torch.device | None = "auto",
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must lie strictly between 0 and 1")
        self._model = model
        self._threshold = threshold
        self._device = resolve_device(device)

    @classmethod
    def from_run_directory(
        cls,
        run_directory: str | Path,
        device: str | torch.device | None = "auto",
    ) -> MoLAGPredictor:
        """Load a model and its calibrated threshold from a run directory."""
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

    @classmethod
    def from_hub(
        cls,
        model_id: str,
        revision: str | None = None,
        token: str | bool | None = None,
        device: str | torch.device | None = "auto",
    ) -> MoLAGPredictor:
        """Load a model and calibration result from a Hugging Face repository."""
        from molag.evaluation import ModelLoader

        snapshot = ModelLoader.from_hub(
            model_id=model_id,
            revision=revision,
            token=token,
        )
        return cls.from_run_directory(snapshot, device=device)

    def predict(self, coordinates: Tensor | np.ndarray) -> InferenceResult:
        """Predict affinities and candidate groups for localised image points.

        Coordinates are centred and max-norm scaled exactly as in dataset generation
        before the complete affinity graph is constructed.
        """
        tensor = torch.as_tensor(coordinates, dtype=torch.float32)
        tensor = self._normalize_coordinates(tensor)
        labels = torch.full((tensor.shape[0], 2), -1, dtype=torch.long)
        prediction = PredictionGenerator(
            model=self._model,
            dataset=[{"x": tensor, "y": labels}],
            data_collator=PyGTrackingAffinityCollator(),
            batch_size=1,
            device=self._device,
        ).predict()[0]
        affinities = torch.from_numpy(prediction.edge_logits).sigmoid().numpy()
        edge_index = prediction.edge_index
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
