"""Project tracker coordinates through the configured pinhole camera."""

from __future__ import annotations

import math

import numpy as np

from molag.config import (
    CAMERA_HEIGHT_PIXELS,
    CAMERA_HORIZONTAL_FIELD_OF_VIEW_DEG,
    CAMERA_WIDTH_PIXELS,
)

from ._base import FloatArray


class CameraIntrinsics:
    """Pinhole camera model used to project tracker LEDs into image space."""

    @classmethod
    def aspect(cls) -> float:
        """Return the image width-to-height ratio."""
        return CAMERA_WIDTH_PIXELS / CAMERA_HEIGHT_PIXELS

    @classmethod
    def hfov_deg(cls) -> float:
        """Return the horizontal field of view in degrees."""
        return CAMERA_HORIZONTAL_FIELD_OF_VIEW_DEG

    @classmethod
    def vfov_deg(cls) -> float:
        """Derive the vertical field of view in degrees."""
        horizontal = math.radians(CAMERA_HORIZONTAL_FIELD_OF_VIEW_DEG)
        vertical = 2.0 * math.atan(
            math.tan(horizontal / 2.0)
            * (CAMERA_HEIGHT_PIXELS / CAMERA_WIDTH_PIXELS)
        )
        return math.degrees(vertical)

    @classmethod
    def fx(cls) -> float:
        """Return the horizontal focal length in pixels."""
        horizontal = math.radians(CAMERA_HORIZONTAL_FIELD_OF_VIEW_DEG)
        return (CAMERA_WIDTH_PIXELS / 2.0) / math.tan(horizontal / 2.0)

    @classmethod
    def fy(cls) -> float:
        """Return the vertical focal length in pixels."""
        vertical = math.radians(cls.vfov_deg())
        return (CAMERA_HEIGHT_PIXELS / 2.0) / math.tan(vertical / 2.0)

    @classmethod
    def cx(cls) -> float:
        """Return the horizontal principal point in pixels."""
        return CAMERA_WIDTH_PIXELS / 2.0

    @classmethod
    def cy(cls) -> float:
        """Return the vertical principal point in pixels."""
        return CAMERA_HEIGHT_PIXELS / 2.0

    @classmethod
    def _project(cls, world_coordinates: FloatArray) -> tuple[FloatArray, np.ndarray]:
        points = np.asarray(world_coordinates, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"world_coordinates must have shape (N, 3), got {points.shape}"
            )

        depth = points[:, 2]
        in_front = depth > 1e-6
        safe_depth = np.maximum(depth, 1e-6)
        x = points[:, 0] / safe_depth
        y = points[:, 1] / safe_depth

        pixels = np.stack(
            [
                cls.fx() * x + cls.cx(),
                cls.fy() * -y + cls.cy(),
            ],
            axis=1,
        )
        in_frame = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] <= CAMERA_WIDTH_PIXELS - 1)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] <= CAMERA_HEIGHT_PIXELS - 1)
        )
        return pixels, in_front & in_frame

    @classmethod
    def project_sample(
        cls,
        leds_world: FloatArray,
        L: int,
    ) -> tuple[FloatArray, np.ndarray]:
        """Project tracker LEDs and return image coordinates with visibility."""
        coordinates = np.asarray(leds_world, dtype=np.float64)
        if coordinates.ndim != 3 or coordinates.shape[2] != 3:
            raise ValueError(
                f"leds_world must have shape (T, L, 3), got {coordinates.shape}"
            )
        if isinstance(L, bool) or not isinstance(L, int) or L < 1:
            raise ValueError("L must be a positive integer")
        if coordinates.shape[1] != L:
            raise ValueError(
                f"leds_world contains {coordinates.shape[1]} LEDs per tracker, "
                f"but L={L}"
            )

        num_trackers = coordinates.shape[0]
        pixels, valid = cls._project(coordinates.reshape(num_trackers * L, 3))
        return pixels.reshape(num_trackers, L, 2), valid.reshape(num_trackers, L)
