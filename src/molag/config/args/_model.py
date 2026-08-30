"""Define MoLAG architecture arguments."""

from __future__ import annotations

from pydantic import Field, PositiveInt

from molag.utils.argparsing import AdditionalArgsBase


class ModelArgs(AdditionalArgsBase):
    """MoLAG architecture configuration."""

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
    message_alignment: PositiveInt = Field(
        default=8,
        description="Input-width alignment used by EdgeConv message MLPs.",
    )
    edge_head_dims: list[PositiveInt] = Field(
        default_factory=lambda: [128],
        min_length=1,
        description="Hidden widths of the symmetric affinity head.",
    )
