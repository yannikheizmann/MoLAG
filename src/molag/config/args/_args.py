from __future__ import annotations

from pathlib import Path

from pydantic import Field

from molag.utils.argparsing import PydanticArgsBase

from ._calibration import CalibrationArgs
from ._dataset import DatasetArgs
from ._eval_dataset import EvalDatasetGenerationArgs
from ._evaluation import EvaluationArgs
from ._loss import LossArgs
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
        description="MoLAG architecture arguments.",
    )
    loss_args: LossArgs = Field(
        default_factory=LossArgs,
        description="Scaled-conjunction affinity-loss arguments.",
    )
    training_args: TrainingArgs = Field(
        default_factory=TrainingArgs,
        description="Optimisation, checkpointing, and data-loading arguments.",
    )
    evaluation_args: EvaluationArgs = Field(
        default_factory=EvaluationArgs,
        description="Frozen-dataset evaluation arguments.",
    )
    eval_dataset_generation_args: EvalDatasetGenerationArgs = Field(
        default_factory=EvalDatasetGenerationArgs,
        description="Frozen evaluation-dataset generation arguments.",
    )
    calibration_args: CalibrationArgs = Field(
        default_factory=CalibrationArgs,
        description="Affinity-threshold calibration arguments.",
    )
