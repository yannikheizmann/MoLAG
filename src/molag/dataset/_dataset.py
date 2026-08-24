from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from molag.config import MAX_SCENE_GENERATION_ATTEMPTS

from ._config import DatasetConfig, PoseConfig
from .modifiers import ModifierBase
from .sample import Sample
from .tracker import TrackerBase

LOGGER = logging.getLogger(__name__)


class _EmptyProjectionError(Exception):
    """Raised when a scene has no visible projected points."""


class TrackingDataset(Dataset):
    """Deterministically generate synthetic tracker scenes on demand."""

    def __init__(
        self,
        size: int,
        num_trackers: int | tuple[int, int],
        TrackerClass: type[TrackerBase],
        seed: int = 0,
        modifiers: Sequence[ModifierBase] | None = None,
        pose_config: PoseConfig | None = None,
    ) -> None:
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("size must be a positive integer")
        if not isinstance(TrackerClass, type) or not issubclass(
            TrackerClass, TrackerBase
        ):
            raise TypeError("TrackerClass must implement TrackerBase")

        if isinstance(num_trackers, int):
            minimum = maximum = num_trackers
        else:
            if len(num_trackers) != 2:
                raise ValueError("num_trackers range must contain two values")
            minimum, maximum = num_trackers
        if minimum < 1:
            raise ValueError("minimum tracker count must be at least 1")
        if maximum < minimum:
            raise ValueError("maximum tracker count must not be below the minimum")
        if maximum > TrackerClass.num_unique_ids():
            raise ValueError(
                f"maximum tracker count cannot exceed "
                f"{TrackerClass.num_unique_ids()}"
            )

        self._size = size
        self._num_trackers_min = minimum
        self._num_trackers_max = maximum
        self._TrackerClass = TrackerClass
        self._seed = int(seed)
        self._pose_config = pose_config or PoseConfig()

        configured_modifiers = list(modifiers or [])
        self._pre_norm_modifiers = [
            modifier
            for modifier in configured_modifiers
            if modifier.stage == "pre_norm"
        ]
        self._post_norm_modifiers = [
            modifier
            for modifier in configured_modifiers
            if modifier.stage == "post_norm"
        ]

    @property
    def num_trackers_range(self) -> tuple[int, int]:
        return self._num_trackers_min, self._num_trackers_max

    @classmethod
    def from_config(cls, config: DatasetConfig) -> TrackingDataset:
        return cls(
            size=config.size,
            num_trackers=config.num_trackers_range,
            TrackerClass=config.tracker_class,
            seed=config.seed,
            modifiers=config.modifiers,
            pose_config=config.pose,
        )

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise TypeError("dataset index must be an integer")
        if not 0 <= idx < self._size:
            raise IndexError(idx)

        rng = np.random.default_rng(self._seed + idx)
        num_trackers = int(
            rng.integers(self._num_trackers_min, self._num_trackers_max + 1)
        )

        for _ in range(MAX_SCENE_GENERATION_ATTEMPTS):
            sample = Sample(
                num_trackers=num_trackers,
                TrackerClass=self._TrackerClass,
                rng=rng,
                pose_cfg=self._pose_config,
            )
            if self._has_occlusion(sample):
                continue
            try:
                return self._build_item(sample, rng)
            except _EmptyProjectionError:
                continue

        LOGGER.warning(
            "Sample %d: occlusion filter not satisfied after %d attempts.",
            idx,
            MAX_SCENE_GENERATION_ATTEMPTS,
        )
        for _ in range(MAX_SCENE_GENERATION_ATTEMPTS):
            sample = Sample(
                num_trackers=num_trackers,
                TrackerClass=self._TrackerClass,
                rng=rng,
                pose_cfg=self._pose_config,
            )
            try:
                item = self._build_item(sample, rng)
            except _EmptyProjectionError:
                continue
            item["warning"] = (
                "maximum attempts reached; occlusion filter was not satisfied"
            )
            return item

        raise RuntimeError(
            f"could not generate a visible dataset sample {idx} after "
            f"{2 * MAX_SCENE_GENERATION_ATTEMPTS} attempts"
        )

    def _has_occlusion(self, sample: Sample) -> bool:
        world_coordinates = sample.get_world_coords()
        num_trackers = len(sample.get_trackers())

        for occluder_index in range(num_trackers):
            vertex0, vertex1, vertex2 = world_coordinates[occluder_index, :3]
            first_edge = vertex1 - vertex0
            second_edge = vertex2 - vertex0
            for tracker_index in range(num_trackers):
                if tracker_index == occluder_index:
                    continue
                for led in world_coordinates[tracker_index]:
                    system = np.column_stack((first_edge, second_edge, -led))
                    try:
                        intersection, *_ = np.linalg.lstsq(
                            system,
                            -vertex0,
                            rcond=None,
                        )
                    except np.linalg.LinAlgError:
                        continue
                    first, second, ray = intersection
                    if (
                        first >= 0.0
                        and second >= 0.0
                        and first + second <= 1.0
                        and 0.0 <= ray <= 1.0
                    ):
                        return True
        return False

    def _build_item(
        self,
        sample: Sample,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        coordinates, labels = sample.get_data()

        for modifier in self._pre_norm_modifiers:
            if rng.random() < modifier.probability:
                coordinates, labels = modifier.apply(coordinates, labels, rng)
        if len(coordinates) == 0:
            raise _EmptyProjectionError

        coordinates = (
            coordinates - coordinates.mean(axis=0, keepdims=True)
        ).astype(np.float32)
        scale = float(np.linalg.norm(coordinates, axis=1).max())
        if scale > 0:
            coordinates = coordinates / scale

        for modifier in self._post_norm_modifiers:
            if rng.random() < modifier.probability:
                coordinates, labels = modifier.apply(coordinates, labels, rng)

        return {
            "x": torch.from_numpy(np.asarray(coordinates, dtype=np.float32)),
            "y": torch.from_numpy(np.asarray(labels, dtype=np.int64)),
        }
