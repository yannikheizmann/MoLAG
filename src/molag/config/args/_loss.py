from __future__ import annotations

import math
from typing import Any

from pydantic import Field, PositiveFloat, PositiveInt, field_validator

from molag.utils.argparsing import AdditionalArgsBase


class LossArgs(AdditionalArgsBase):
    """Scaled-conjunction affinity-loss configuration."""

    connectivity_weight: float = Field(default=1.0, ge=0)
    connectivity_margin: float = Field(default=1.0, ge=0)
    separation_weight: float = Field(default=0.46, ge=0)
    separation_margin: float = Field(default=1.0, ge=0)
    spurious_bridge_weight: float = Field(default=0.25, ge=0)
    spurious_margin: float = Field(default=0.0, ge=0)
    max_tracker_nodes: PositiveInt = 7
    aggregation_beta: PositiveFloat = 1.0
    delta_nontree: float = Field(default=3.0, ge=0)
    eps_spur: float = Field(default=0.01, ge=0)
    conjunct_scaling_power: float = Field(default=0.5, ge=0, le=1)
    separation_scaling_power: float | None = Field(default=None, ge=0, le=1)
    eligible_scene_mean: bool = True
    supcon_weight: float = Field(default=0.0, ge=0)
    supcon_temperature: PositiveFloat = 0.2

    @field_validator("aggregation_beta", "delta_nontree", mode="before")
    @classmethod
    def accept_infinity(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"inf", "infinity"}:
            return math.inf
        return value
