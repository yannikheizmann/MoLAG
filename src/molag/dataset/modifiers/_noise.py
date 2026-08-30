"""Perturb normalised image coordinates with Gaussian noise."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from ._base import FloatArray, IntArray, ModifierBase, ModifierStage


class PixelNoiseModifier(ModifierBase):
    """Add independent Gaussian noise to normalised coordinates."""

    type: Literal["PixelNoise"] = "PixelNoise"
    std: float = Field(default=0.02, gt=0.0)

    @property
    def stage(self) -> ModifierStage:
        """Apply noise after coordinate normalisation."""
        return "post_norm"

    def apply(
        self,
        x: FloatArray,
        y: IntArray,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, IntArray]:
        """Add independent Gaussian noise without changing labels."""
        noise = rng.normal(0.0, self.std, size=x.shape).astype(np.float32)
        return x + noise, y
