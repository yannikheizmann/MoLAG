from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from molag.utils import GeometryUtils

if TYPE_CHECKING:
    from molag.dataset import PoseConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TrackerPose:
    """Rigid transformation from tracker coordinates to world coordinates."""

    R: FloatArray
    t: FloatArray

    def __post_init__(self) -> None:
        rotation = np.array(self.R, dtype=np.float64, copy=True)
        translation = np.array(self.t, dtype=np.float64, copy=True)
        if rotation.shape != (3, 3):
            raise ValueError(f"R must have shape (3, 3), got {rotation.shape}")
        if translation.shape != (3,):
            raise ValueError(f"t must have shape (3,), got {translation.shape}")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError("R must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
            raise ValueError("R must have determinant 1")
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "R", rotation)
        object.__setattr__(self, "t", translation)

    @classmethod
    def identity(cls) -> TrackerPose:
        return cls(R=np.eye(3), t=np.zeros(3))

    @classmethod
    def sample(
        cls,
        rng: np.random.Generator,
        pose_cfg: PoseConfig | None = None,
    ) -> TrackerPose:
        from molag.dataset import PoseConfig

        config = pose_cfg or PoseConfig()
        translation = np.array(
            [
                rng.uniform(config.x_min, config.x_max),
                rng.uniform(config.y_min, config.y_max),
                rng.uniform(config.z_min, config.z_max),
            ]
        )
        direction = GeometryUtils.normalize(translation)
        facing_rotation = GeometryUtils.rotate_negative_z_to(direction)
        tilt_rotation = GeometryUtils.random_tilt_about(
            direction,
            math.radians(config.max_tilt_deg),
            rng,
        )
        return cls(R=tilt_rotation @ facing_rotation, t=translation)
