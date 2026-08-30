"""Define the composable affinity-loss arguments."""

from __future__ import annotations

import math
from typing import Any

from pydantic import Field, PositiveFloat, PositiveInt, field_validator

from molag.utils.argparsing import AdditionalArgsBase


class LossArgs(AdditionalArgsBase):
    """Scaled-conjunction affinity-loss configuration."""

    connectivity_weight: float = Field(
        default=1.0, ge=0, description="Weight of the tracker-connectivity term."
    )
    connectivity_margin: float = Field(
        default=1.0, ge=0, description="Minimum affinity margin on tree edges."
    )
    separation_weight: float = Field(
        default=0.46, ge=0, description="Weight of the tracker-separation term."
    )
    separation_margin: float = Field(
        default=1.0, ge=0, description="Maximum affinity margin between trackers."
    )
    spurious_bridge_weight: float = Field(
        default=0.25, ge=0, description="Weight of the spurious-bridge term."
    )
    spurious_margin: float = Field(
        default=0.0, ge=0, description="Maximum affinity margin for spurious edges."
    )
    max_tracker_nodes: PositiveInt = Field(
        default=7, description="Largest real tracker component handled by the loss."
    )
    aggregation_beta: PositiveFloat = Field(
        default=1.0,
        description=(
            "Inverse temperature for smooth edge aggregation; infinity uses max."
        ),
    )
    delta_nontree: float = Field(
        default=3.0,
        ge=0,
        description="Penalty offset applied to non-tree same-tracker edges.",
    )
    eps_spur: float = Field(
        default=0.01,
        ge=0,
        description="Numerical floor used when aggregating spurious connections.",
    )
    conjunct_scaling_power: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Scene-size scaling exponent for conjunction terms.",
    )
    separation_scaling_power: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Optional scene-size scaling exponent for separation.",
    )
    eligible_scene_mean: bool = Field(
        default=True,
        description="Average each term only across scenes eligible for that term.",
    )
    supcon_weight: float = Field(
        default=0.0, ge=0, description="Weight of supervised contrastive learning."
    )
    supcon_temperature: PositiveFloat = Field(
        default=0.2, description="Temperature of the contrastive similarity logits."
    )

    @field_validator("aggregation_beta", "delta_nontree", mode="before")
    @classmethod
    def accept_infinity(cls, value: Any) -> Any:
        """Convert textual infinity values accepted by YAML and the CLI."""
        if isinstance(value, str) and value.strip().lower() in {"inf", "infinity"}:
            return math.inf
        return value
