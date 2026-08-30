from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveInt, model_validator

from molag.utils.argparsing import AdditionalArgsBase


class EvaluationArgs(AdditionalArgsBase):
    """Configuration for evaluating a finetuned MoLAG checkpoint."""

    run_directory: Path = Field(
        default=Path("results"),
        description="Finetuning output containing config.yaml and model weights.",
    )
    dataset: Path = Field(
        default=Path("evaluation/evaluation.yaml"),
        description="Frozen EvalDataset YAML file.",
    )
    calibration_dataset: Path = Field(
        default=Path("evaluation/calibration.yaml"),
        description="Frozen dataset used to select the affinity threshold.",
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
    threshold: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description=(
            "Explicit affinity threshold. When set, the calibration phase is skipped."
        ),
    )
    metrics: list[Literal["Affinity", "Partition"]] = Field(
        default_factory=lambda: ["Affinity", "Partition"],
        min_length=1,
        description="Registered streaming metrics evaluated in one pass.",
    )
    threshold_min: float = Field(default=0.05, gt=0, lt=1)
    threshold_max: float = Field(default=0.95, gt=0, lt=1)
    threshold_step: float = Field(default=0.01, gt=0, lt=1)
    objective: str = Field(
        default="partition_accuracy",
        min_length=1,
        description="Calibration metric maximized to select the threshold.",
    )

    @model_validator(mode="after")
    def validate_threshold_grid(self) -> EvaluationArgs:
        if self.threshold_max <= self.threshold_min:
            raise ValueError("threshold_max must be greater than threshold_min")
        steps = (self.threshold_max - self.threshold_min) / self.threshold_step
        if abs(steps - round(steps)) > 1e-8:
            raise ValueError("threshold range must be divisible by threshold_step")
        return self
