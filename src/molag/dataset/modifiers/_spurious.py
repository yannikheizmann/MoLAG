"""Add unassigned detections to synthetic scenes."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from ._base import FloatArray, IntArray, ModifierBase, ModifierStage


class SpuriousBlobsModifier(ModifierBase):
    """Add spurious points inside the real detections' bounding box."""

    type: Literal["SpuriousBlobs"] = "SpuriousBlobs"
    min_blobs: int = Field(default=1, ge=1)
    max_blobs: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def validate_blob_range(self) -> SpuriousBlobsModifier:
        """Validate the inclusive range of spurious detections."""
        if self.max_blobs < self.min_blobs:
            raise ValueError("max_blobs must be greater than or equal to min_blobs")
        return self

    @property
    def stage(self) -> ModifierStage:
        """Add spurious detections before coordinate normalisation."""
        return "pre_norm"

    def apply(
        self,
        x: FloatArray,
        y: IntArray,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, IntArray]:
        """Sample spurious detections within the real-point bounding box."""
        num_blobs = int(rng.integers(self.min_blobs, self.max_blobs + 1))
        real_coordinates = x[y[:, 0] >= 0]
        if len(real_coordinates):
            lower = real_coordinates.min(axis=0)
            upper = real_coordinates.max(axis=0)
        else:
            lower = np.zeros(2, dtype=np.float32)
            upper = np.ones(2, dtype=np.float32)

        coordinates = rng.uniform(lower, upper, size=(num_blobs, 2)).astype(
            np.float32
        )
        existing_minimum = int(y[:, 0].min()) if len(y) else 0
        first_id = min(-1, existing_minimum - 1)
        tracker_ids = np.arange(
            first_id,
            first_id - num_blobs,
            -1,
            dtype=np.int64,
        )
        led_ids = np.full(num_blobs, -1, dtype=np.int64)
        labels = np.stack((tracker_ids, led_ids), axis=1)
        return (
            np.concatenate((x, coordinates), axis=0),
            np.concatenate((y, labels), axis=0),
        )
