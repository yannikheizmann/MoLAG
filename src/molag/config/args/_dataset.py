from __future__ import annotations

from pathlib import Path

from pydantic import Field, PositiveInt

from molag.utils.argparsing import AdditionalArgsBase


class DatasetArgs(AdditionalArgsBase):
    """Dataset profile selection and generated split sizes."""

    dataset_profile: Path = Field(
        default=Path("src/molag/dataset/profiles/molag_standard.yaml"),
        description="YAML profile specifying synthetic dataset generation.",
    )
    train_size: PositiveInt = Field(
        default=5_000_000,
        description="Number of deterministically generated training scenes.",
    )
    eval_size: PositiveInt = Field(
        default=10_000,
        description=(
            "Number of in-training evaluation scenes. Their seed range begins "
            "after the training dataset and is therefore disjoint from it."
        ),
    )
