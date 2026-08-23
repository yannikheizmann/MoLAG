from __future__ import annotations

from pathlib import Path

from pydantic import Field

from molag.utils.argparsing import PydanticArgsBase

from ._dataset import DatasetArgs
from ._model import ModelArgs
from ._training import TrainingArgs


class Args(PydanticArgsBase):
    """Composition root for the argument groups of a MoLAG training command."""

    config: Path | None = Field(
        default=None,
        description="YAML file containing argument overrides.",
    )
    dataset_args: DatasetArgs = Field(
        default_factory=DatasetArgs,
        description="Dataset profile and generated split sizes.",
    )
    model_args: ModelArgs = Field(
        default_factory=ModelArgs,
        description="MoLAG architecture and scaled-conjunction loss arguments.",
    )
    training_args: TrainingArgs = Field(
        default_factory=TrainingArgs,
        description="Optimisation, checkpointing, and data-loading arguments.",
    )
