from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
ModifierStage = Literal["pre_norm", "post_norm"]


class ModifierBase(BaseModel, ABC):
    """Interface for a stochastic coordinate-set transformation."""

    model_config = ConfigDict(extra="forbid")

    probability: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    @abstractmethod
    def stage(self) -> ModifierStage:
        """Return when the modifier runs relative to normalisation."""

    @abstractmethod
    def apply(
        self,
        x: FloatArray,
        y: IntArray,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, IntArray]:
        """Transform one coordinate and label array pair."""

