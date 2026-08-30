"""Represent one synthetic scene of uniquely coded trackers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from molag.dataset._config import PoseConfig
from molag.dataset.tracker import CameraIntrinsics, TrackerBase

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
IntArray = NDArray[np.int64]


class Sample:
    """One synthetic scene containing uniquely coded rigid trackers."""

    def __init__(
        self,
        num_trackers: int,
        TrackerClass: type[TrackerBase],
        rng: np.random.Generator,
        pose_cfg: PoseConfig | None = None,
    ) -> None:
        if isinstance(num_trackers, bool) or not isinstance(num_trackers, int):
            raise TypeError("num_trackers must be an integer")
        if not isinstance(TrackerClass, type) or not issubclass(
            TrackerClass, TrackerBase
        ):
            raise TypeError("TrackerClass must implement TrackerBase")
        if num_trackers < 1:
            raise ValueError("num_trackers must be at least 1")
        if num_trackers > TrackerClass.num_unique_ids():
            raise ValueError(
                f"num_trackers cannot exceed the {TrackerClass.num_unique_ids()} "
                f"unique IDs provided by {TrackerClass.__name__}"
            )
        self._num_trackers = num_trackers
        self._TrackerClass = TrackerClass
        self._rng = rng
        self._pose_cfg = pose_cfg
        self._trackers = self._instantiate_trackers()

    def _get_tracker_ids(self) -> list[int]:
        identifiers = self._rng.choice(
            self._TrackerClass.num_unique_ids(),
            size=self._num_trackers,
            replace=False,
        )
        return [int(identifier) for identifier in identifiers]

    def _instantiate_trackers(self) -> list[TrackerBase]:
        return [
            self._TrackerClass.from_id(identifier, self._rng, self._pose_cfg)
            for identifier in self._get_tracker_ids()
        ]

    def get_world_coords(self) -> FloatArray:
        """Return world coordinates with separate tracker and LED axes."""
        return np.stack(
            [tracker.get_leds_world_coords() for tracker in self._trackers],
            axis=0,
        )

    def get_trackers(self) -> list[TrackerBase]:
        """Return a copy of the scene's tracker collection."""
        return list(self._trackers)

    def get_data(self) -> tuple[Float32Array, IntArray]:
        """Return visible image points and ``(tracker_id, led_index)`` labels."""

        num_leds = self._TrackerClass.num_leds()
        projected, valid = CameraIntrinsics.project_sample(
            self.get_world_coords(),
            L=num_leds,
        )
        tracker_ids = np.array(
            [tracker.id for tracker in self._trackers],
            dtype=np.int64,
        )
        id_grid = np.repeat(tracker_ids[:, None], num_leds, axis=1)
        led_grid = np.broadcast_to(
            np.arange(num_leds, dtype=np.int64),
            (self._num_trackers, num_leds),
        )

        coordinates = projected[valid].astype(np.float32, copy=False)
        labels = np.stack((id_grid[valid], led_grid[valid]), axis=1)
        return coordinates, labels.astype(np.int64, copy=False)
