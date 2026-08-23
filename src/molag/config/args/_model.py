from __future__ import annotations

import math
from typing import Any

from pydantic import Field, PositiveFloat, PositiveInt, field_validator

from molag.utils.argparsing import AdditionalArgsBase


class ModelArgs(AdditionalArgsBase):
    """MoLAG architecture and loss configuration."""

    in_dim: PositiveInt = Field(
        default=2,
        description="Number of input features for each localised LED point.",
    )
    hidden_dims: list[PositiveInt] = Field(
        default_factory=lambda: [128, 256, 512, 1024, 2048, 2048, 1024],
        min_length=1,
        description="Output widths of the successive EdgeConv blocks.",
    )
    edge_feature_dim: PositiveInt = Field(
        default=3,
        description="Number of geometric edge features supplied to every block.",
    )
    edge_head_dims: list[PositiveInt] = Field(
        default_factory=lambda: [128],
        min_length=1,
        description="Hidden widths of the symmetric affinity head.",
    )

    connectivity_weight: float = Field(
        default=1.0,
        ge=0,
        description="Weight of the within-tracker connectivity term.",
    )
    connectivity_margin: float = Field(
        default=1.0,
        ge=0,
        description="Positive-logit margin for within-tracker connectivity.",
    )
    separation_weight: float = Field(
        default=0.46,
        ge=0,
        description="Weight of the between-tracker separation term.",
    )
    separation_margin: float = Field(
        default=1.0,
        ge=0,
        description="Negative-logit margin for between-tracker separation.",
    )
    spurious_bridge_weight: float = Field(
        default=0.25,
        ge=0,
        description="Weight of paths connecting real trackers through spurious points.",
    )
    spurious_margin: float = Field(
        default=0.0,
        ge=0,
        description="Logit margin for paths whose internal nodes are spurious.",
    )
    max_tracker_nodes: PositiveInt = Field(
        default=7,
        description="Maximum visible real points in one tracker constellation.",
    )
    aggregation_beta: PositiveFloat = Field(
        default=1.0,
        description="Sharpness of the soft maximum within each loss condition.",
    )
    delta_nontree: float = Field(
        default=3.0,
        ge=0,
        description="Penalty-unit handicap applied to non-tree within-tracker edges.",
    )
    eps_spur: float = Field(
        default=0.01,
        ge=0,
        description="Relative weight of tracker-to-spurious-point attachment conditions.",
    )
    conjunct_scaling_power: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Scene-size scaling exponent for connectivity conditions.",
    )
    separation_scaling_power: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Scene-size scaling exponent for separation conditions. None follows "
            "conjunct_scaling_power."
        ),
    )
    eligible_scene_mean: bool = Field(
        default=True,
        description="Average each loss family only over scenes where it applies.",
    )
    supcon_weight: float = Field(
        default=0.0,
        ge=0,
        description="Weight of the optional per-scene supervised contrastive term.",
    )
    supcon_temperature: PositiveFloat = Field(
        default=0.2,
        description="Temperature used by the supervised contrastive similarity.",
    )

    @field_validator("aggregation_beta", "delta_nontree", mode="before")
    @classmethod
    def accept_infinity(cls, value: Any) -> Any:
        """Accept readable infinity spellings for reference-loss experiments."""

        if isinstance(value, str) and value.strip().lower() in {"inf", "infinity"}:
            return math.inf
        return value
