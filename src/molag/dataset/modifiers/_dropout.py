from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from ._base import FloatArray, IntArray, ModifierBase, ModifierStage


class DropoutModifier(ModifierBase):
    """Randomly remove real LEDs while retaining a per-tracker minimum."""

    type: Literal["Dropout"] = "Dropout"
    drop_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    min_leds_per_tracker: int = Field(default=3, ge=1)

    @property
    def stage(self) -> ModifierStage:
        return "pre_norm"

    def apply(
        self,
        x: FloatArray,
        y: IntArray,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, IntArray]:
        keep = np.ones(len(x), dtype=np.bool_)
        for tracker_id in np.unique(y[:, 0]):
            if tracker_id < 0:
                continue
            indices = np.flatnonzero(y[:, 0] == tracker_id)
            num_leds = len(indices)
            num_dropped = int(rng.binomial(num_leds, self.drop_probability))
            num_kept = min(
                num_leds,
                max(self.min_leds_per_tracker, num_leds - num_dropped),
            )
            if num_kept < num_leds:
                kept = rng.choice(indices, size=num_kept, replace=False)
                keep[indices] = False
                keep[kept] = True
        return x[keep].copy(), y[keep].copy()

