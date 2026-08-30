"""Construct the coded seven-LED triangular tracker geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from molag.config import (
    TRIANGULAR_TRACKER_BARYCENTRIC_TRANSFORM,
    TRIANGULAR_TRACKER_CANDIDATE_COORDINATES,
    TRIANGULAR_TRACKER_SIDE_LENGTH_MM,
)

from ._base import FloatArray, TrackerCodeBase, TrackerGeometryBase
from ._code import TriangularTrackerCode


@dataclass(frozen=True, slots=True)
class TriangularTrackerGeometry(TrackerGeometryBase):
    """Seven-LED triangular tracker geometry."""

    code: TriangularTrackerCode

    @classmethod
    def from_code(cls, code: TrackerCodeBase) -> TriangularTrackerGeometry:
        """Construct geometry for a triangular tracker code."""
        if not isinstance(code, TriangularTrackerCode):
            raise TypeError("code must be a TriangularTrackerCode")
        return cls(code=code)

    @classmethod
    def num_leds(cls) -> int:
        """Return the number of LEDs in the triangular geometry."""
        return 7

    @property
    def center(self) -> FloatArray:
        """Return the centroid of the three tracker vertices."""
        return self._vertices().mean(axis=0)

    def as_array(self) -> FloatArray:
        """Return vertex and coded side-LED coordinates."""
        vertices = self._vertices()
        side_leds = self._side_leds(vertices)
        return np.concatenate((vertices, side_leds), axis=0)

    @staticmethod
    def _vertices() -> FloatArray:
        height = (
            0.5 * math.sqrt(3.0) * TRIANGULAR_TRACKER_SIDE_LENGTH_MM
        )
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [TRIANGULAR_TRACKER_SIDE_LENGTH_MM, 0.0, 0.0],
                [0.5 * TRIANGULAR_TRACKER_SIDE_LENGTH_MM, height, 0.0],
            ],
            dtype=np.float64,
        )

    def _side_leds(self, vertices: FloatArray) -> FloatArray:
        v0, v1, v2 = vertices
        side_bases = (
            (v0, v1 - v0, v2 - v0),
            (v1, v2 - v1, v0 - v1),
            (v2, v0 - v2, v1 - v2),
        )

        positions: list[FloatArray] = []
        for side_index, (origin, first_axis, second_axis) in enumerate(side_bases):
            selected_code = self.code[side_index]
            for candidate_index in range(3):
                place_led = (
                    candidate_index != selected_code
                    if side_index == 0
                    else candidate_index == selected_code
                )
                if place_led:
                    first, second = (
                        TRIANGULAR_TRACKER_BARYCENTRIC_TRANSFORM
                        @ TRIANGULAR_TRACKER_CANDIDATE_COORDINATES[
                            :, candidate_index
                        ]
                    )
                    positions.append(
                        origin + first * first_axis + second * second_axis
                    )

        return np.stack(positions, axis=0)
