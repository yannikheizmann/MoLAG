from __future__ import annotations

from pathlib import Path

from pydantic import Field, PositiveInt

from molag.utils.argparsing import AdditionalArgsBase


class EvalDatasetGenerationArgs(AdditionalArgsBase):
    """Configuration for materializing a frozen evaluation dataset."""

    name: str = Field(
        default="evaluation",
        min_length=1,
        description="Human-readable identifier stored with the dataset.",
    )
    dataset_profile: Path = Field(
        default=Path("src/molag/dataset/profiles/molag_standard.yaml"),
        description="DatasetConfig YAML used to generate candidate scenes.",
    )
    size: PositiveInt = Field(
        default=10_000,
        description="Number of scenes to materialize.",
    )
    seed: int = Field(
        default=5_010_000,
        description="First candidate RNG seed used by the dataset.",
    )
    output: Path = Field(
        default=Path("evaluation/evaluation.yaml"),
        description="Destination YAML file for the frozen dataset.",
    )
    description: str = Field(
        default="",
        description="Optional notes stored with the dataset.",
    )
