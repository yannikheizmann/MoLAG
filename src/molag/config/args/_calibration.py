from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveInt, model_validator

from molag.utils.argparsing import AdditionalArgsBase


class CalibrationArgs(AdditionalArgsBase):
    """Configuration for affinity-threshold calibration."""

    run_directory: Path = Field(
        default=Path("results"),
        description="Finetuning output containing configuration and model weights.",
    )
    dataset: Path = Field(
        default=Path("evaluation/calibration.yaml"),
        description="Frozen calibration dataset YAML file.",
    )
    output: Path = Field(
        default=Path("evaluation/calibration.json"),
        description="JSON file receiving threshold scores and provenance.",
    )
    batch_size: PositiveInt = Field(
        default=128,
        description="Number of calibration scenes processed per batch.",
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
    threshold_min: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Lowest affinity probability threshold evaluated.",
    )
    threshold_max: float = Field(
        default=0.95,
        gt=0,
        lt=1,
        description="Highest affinity probability threshold evaluated.",
    )
    threshold_step: float = Field(
        default=0.01,
        gt=0,
        lt=1,
        description="Spacing of the inclusive threshold grid.",
    )
    metric: Literal["Affinity", "Partition"] = Field(
        default="Partition",
        description="Registered metric implementation evaluated at each threshold.",
    )
    objective: str = Field(
        default="partition_accuracy",
        min_length=1,
        description="Scalar result returned by the metric that is maximized.",
    )

    @model_validator(mode="after")
    def validate_threshold_grid(self) -> CalibrationArgs:
        if self.threshold_max <= self.threshold_min:
            raise ValueError("threshold_max must be greater than threshold_min")
        steps = (self.threshold_max - self.threshold_min) / self.threshold_step
        if abs(steps - round(steps)) > 1e-8:
            raise ValueError("threshold range must be divisible by threshold_step")
        return self
