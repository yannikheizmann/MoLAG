from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveInt

from molag.utils.argparsing import PydanticArgsBase


class EvaluationArgs(PydanticArgsBase):
    """Configuration for evaluating a finetuned MoLAG checkpoint."""

    config: Path | None = Field(
        default=None,
        description="YAML file containing evaluation argument overrides.",
    )
    run_directory: Path = Field(
        description="Finetuning output containing config.yaml and model weights.",
    )
    dataset: Path = Field(
        description="Frozen EvalDataset YAML file.",
    )
    output: Path = Field(
        default=Path("evaluation_results.json"),
        description="JSON file receiving metrics and evaluation provenance.",
    )
    batch_size: PositiveInt = Field(
        default=128,
        description="Number of frozen scenes evaluated per batch.",
    )
    dataloader_num_workers: int = Field(
        default=0,
        ge=0,
        description="Worker processes used to load frozen scenes.",
    )
    device: str = Field(
        default="cpu",
        min_length=1,
        description="Torch device used for model inference.",
    )
    threshold: float = Field(
        default=0.5,
        gt=0,
        lt=1,
        description="Affinity probability threshold used by evaluation metrics.",
    )
    metrics: list[Literal["Affinity", "Partition"]] = Field(
        default_factory=lambda: ["Affinity", "Partition"],
        min_length=1,
        description="Registered streaming metrics evaluated in one pass.",
    )
