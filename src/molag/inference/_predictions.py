"""Represent and persist reusable raw predictions for fixed scenes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ScenePrediction:
    """Coordinates, targets, graph edges, and raw logits for one scene."""

    coordinates: np.ndarray
    point_labels: np.ndarray
    edge_index: np.ndarray
    edge_logits: np.ndarray
    edge_labels: np.ndarray

    def __post_init__(self) -> None:
        n_nodes = self.coordinates.shape[0]
        n_edges = self.edge_logits.size
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape (num_nodes, 2)")
        if self.point_labels.shape != (n_nodes, 2):
            raise ValueError("point_labels must have shape (num_nodes, 2)")
        if self.edge_index.shape != (2, n_edges):
            raise ValueError("edge_index must have shape (2, num_edges)")
        if self.edge_labels.shape != (n_edges,):
            raise ValueError("edge_labels must have shape (num_edges,)")
        if not np.issubdtype(self.point_labels.dtype, np.integer):
            raise ValueError("point_labels must have an integer dtype")
        if not np.issubdtype(self.edge_index.dtype, np.integer):
            raise ValueError("edge_index must have an integer dtype")
        if not np.issubdtype(self.edge_labels.dtype, np.integer):
            raise ValueError("edge_labels must have an integer dtype")
        if not np.isfinite(self.coordinates).all():
            raise ValueError("coordinates must contain only finite values")
        if not np.isfinite(self.edge_logits).all():
            raise ValueError("edge_logits must contain only finite values")
        if self.edge_index.size and (
            self.edge_index.min() < 0 or self.edge_index.max() >= n_nodes
        ):
            raise ValueError("edge_index contains a node outside the scene")


class PredictionCache:
    """Ordered, reusable raw predictions for a fixed sequence of scenes.

    Caches preserve enough input and target data to calibrate thresholds, recompute
    existing metrics, or add new metrics without loading the model or source dataset.
    """

    FORMAT_VERSION = 1

    def __init__(self, scenes: Iterable[ScenePrediction]) -> None:
        self._scenes = tuple(scenes)
        if not self._scenes:
            raise ValueError("prediction cache must contain at least one scene")

    def __len__(self) -> int:
        return len(self._scenes)

    def __iter__(self) -> Iterator[ScenePrediction]:
        return iter(self._scenes)

    def __getitem__(self, index: int) -> ScenePrediction:
        return self._scenes[index]

    def to_npz(self, path: str | Path) -> Path:
        """Save concatenated, pickle-free arrays with scene boundary offsets.

        Returns:
            Path to the compressed NumPy archive.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        node_counts = np.asarray(
            [scene.coordinates.shape[0] for scene in self], dtype=np.int64
        )
        edge_counts = np.asarray(
            [scene.edge_logits.size for scene in self], dtype=np.int64
        )
        node_offsets = np.concatenate(([0], np.cumsum(node_counts)))
        edge_offsets = np.concatenate(([0], np.cumsum(edge_counts)))
        with destination.open("wb") as stream:
            np.savez_compressed(
                stream,
                format_version=np.asarray(self.FORMAT_VERSION, dtype=np.int64),
                node_offsets=node_offsets,
                edge_offsets=edge_offsets,
                coordinates=np.concatenate(
                    [scene.coordinates.astype(np.float32, copy=False) for scene in self]
                ),
                point_labels=np.concatenate(
                    [scene.point_labels.astype(np.int64, copy=False) for scene in self]
                ),
                edge_index=np.concatenate(
                    [scene.edge_index.astype(np.int64, copy=False) for scene in self],
                    axis=1,
                ),
                edge_logits=np.concatenate(
                    [scene.edge_logits.astype(np.float32, copy=False) for scene in self]
                ),
                edge_labels=np.concatenate(
                    [scene.edge_labels.astype(np.int64, copy=False) for scene in self]
                ),
            )
        return destination

    @classmethod
    def from_npz(cls, path: str | Path) -> PredictionCache:
        """Load and validate a cache without enabling object deserialisation."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"prediction cache not found: {source}")
        with np.load(source, allow_pickle=False) as values:
            cls._validate_archive(values)
            version = int(values["format_version"])
            if version != cls.FORMAT_VERSION:
                raise ValueError(f"unsupported prediction-cache version: {version}")
            node_offsets = values["node_offsets"]
            edge_offsets = values["edge_offsets"]
            scenes = []
            for index in range(node_offsets.size - 1):
                node_start, node_stop = node_offsets[index : index + 2]
                edge_start, edge_stop = edge_offsets[index : index + 2]
                scenes.append(
                    ScenePrediction(
                        coordinates=values["coordinates"][node_start:node_stop].copy(),
                        point_labels=values["point_labels"][node_start:node_stop].copy(),
                        edge_index=values["edge_index"][:, edge_start:edge_stop].copy(),
                        edge_logits=values["edge_logits"][edge_start:edge_stop].copy(),
                        edge_labels=values["edge_labels"][edge_start:edge_stop].copy(),
                    )
                )
        return cls(scenes)

    @classmethod
    def _validate_archive(cls, values) -> None:
        required = {
            "format_version",
            "node_offsets",
            "edge_offsets",
            "coordinates",
            "point_labels",
            "edge_index",
            "edge_logits",
            "edge_labels",
        }
        missing = required.difference(values.files)
        if missing:
            raise ValueError(
                "prediction cache is missing arrays: " + ", ".join(sorted(missing))
            )
        if values["format_version"].shape != ():
            raise ValueError("format_version must be a scalar")

        node_offsets = values["node_offsets"]
        edge_offsets = values["edge_offsets"]
        for name, offsets in (
            ("node_offsets", node_offsets),
            ("edge_offsets", edge_offsets),
        ):
            if offsets.ndim != 1 or offsets.size < 2:
                raise ValueError(f"{name} must be a one-dimensional boundary array")
            if not np.issubdtype(offsets.dtype, np.integer):
                raise ValueError(f"{name} must have an integer dtype")
            if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
                raise ValueError(f"{name} must start at zero and be nondecreasing")
        if node_offsets.size != edge_offsets.size:
            raise ValueError("node_offsets and edge_offsets must describe equal scenes")

        coordinates = values["coordinates"]
        point_labels = values["point_labels"]
        edge_index = values["edge_index"]
        edge_logits = values["edge_logits"]
        edge_labels = values["edge_labels"]
        n_nodes = int(node_offsets[-1])
        n_edges = int(edge_offsets[-1])
        if coordinates.shape != (n_nodes, 2):
            raise ValueError("coordinates do not match node_offsets")
        if point_labels.shape != (n_nodes, 2):
            raise ValueError("point_labels do not match node_offsets")
        if edge_index.shape != (2, n_edges):
            raise ValueError("edge_index does not match edge_offsets")
        if edge_logits.shape != (n_edges,):
            raise ValueError("edge_logits do not match edge_offsets")
        if edge_labels.shape != (n_edges,):
            raise ValueError("edge_labels do not match edge_offsets")
