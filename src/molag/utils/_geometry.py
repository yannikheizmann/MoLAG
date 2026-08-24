from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class GeometryUtils:
    """Rotation and vector operations used for tracker pose sampling."""

    @staticmethod
    def normalize(vector: FloatArray, epsilon: float = 1e-12) -> FloatArray:
        vector = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > epsilon else np.zeros_like(vector)

    @staticmethod
    def rodrigues(axis: FloatArray, angle: float) -> FloatArray:
        x, y, z = GeometryUtils.normalize(axis)
        if np.allclose((x, y, z), 0.0):
            return np.eye(3)
        cross_product = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=np.float64,
        )
        return (
            np.eye(3)
            + math.sin(angle) * cross_product
            + (1.0 - math.cos(angle)) * (cross_product @ cross_product)
        )

    @staticmethod
    def rotate_negative_z_to(direction: FloatArray) -> FloatArray:
        negative_z = np.array([0.0, 0.0, -1.0])
        target = GeometryUtils.normalize(direction)
        axis = np.cross(negative_z, target)
        cosine = float(np.clip(-target[2], -1.0, 1.0))
        return GeometryUtils.rodrigues(axis, math.acos(cosine))

    @staticmethod
    def random_tilt_about(
        direction: FloatArray,
        maximum_angle: float,
        rng: np.random.Generator,
    ) -> FloatArray:
        target = GeometryUtils.normalize(direction)
        helper = np.array([rng.random(), rng.random(), 0.0])
        axis = np.cross(helper, target)
        if np.linalg.norm(axis) < 1e-12:
            axis = np.cross(np.array([1.0, 0.0, 0.0]), target)
            if np.linalg.norm(axis) < 1e-12:
                axis = np.cross(np.array([0.0, 1.0, 0.0]), target)
        angle = rng.uniform(0.0, maximum_angle)
        return GeometryUtils.rodrigues(axis, angle)

