from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from molag.utils.argparsing import AdditionalArgsBase

class TrainingArgs(AdditionalArgsBase):
    """MoLAG training configuration."""

    output_dir: Path = Field(
        default=Path("results"),
        description="Directory receiving checkpoints and run metadata.",
    )
    per_device_train_batch_size: PositiveInt = Field(
        default=256,
        description="Number of scenes in one physical training batch per device.",
    )
    gradient_accumulation_steps: PositiveInt = Field(
        default=4,
        description="Physical batches accumulated before one optimiser update.",
    )
    per_device_eval_batch_size: PositiveInt = Field(
        default=128,
        description="Number of scenes evaluated per device and step.",
    )
    learning_rate: PositiveFloat = Field(
        default=1e-3,
        description="Peak optimiser learning rate.",
    )
    weight_decay: NonNegativeFloat = Field(
        default=0.0,
        description="Weight-decay coefficient.",
    )
    num_train_epochs: PositiveInt = Field(
        default=10,
        description="Number of complete passes over the generated training dataset.",
    )
    lr_scheduler_type: Literal["linear", "cosine"] = Field(
        default="cosine",
        description="Learning-rate schedule applied after warm-up.",
    )
    warmup_ratio: float = Field(
        default=0.01,
        ge=0,
        lt=1,
        description="Fraction of optimiser steps used for learning-rate warm-up.",
    )
    logging_steps: PositiveInt = Field(
        default=100,
        description="Optimiser steps between training log entries.",
    )
    eval_steps: PositiveInt = Field(
        default=5_000,
        description="Optimiser steps between in-training evaluations.",
    )
    save_steps: PositiveInt = Field(
        default=5_000,
        description="Optimiser steps between checkpoints.",
    )
    save_total_limit: PositiveInt = Field(
        default=3,
        description="Maximum number of intermediate checkpoints retained.",
    )
    dataloader_num_workers: NonNegativeInt = Field(
        default=8,
        description="Worker processes used by each training data loader.",
    )
    dataloader_persistent_workers: bool = Field(
        default=True,
        description="Keep data-loader workers alive between epochs.",
    )
    fp16: bool = Field(
        default=False,
        description="Use IEEE float16 mixed precision.",
    )
    bf16: bool = Field(
        default=True,
        description="Use bfloat16 mixed precision.",
    )
    seed: int = Field(
        default=42,
        description="Seed for model initialisation and training operations.",
    )
    training_metrics: list[str] = Field(
        default_factory=lambda: ["Affinity"],
        min_length=1,
        description="Registered streaming metrics used during training evaluation.",
    )
    metric_for_best_model: str = Field(
        default="loss",
        min_length=1,
        description="Evaluation metric used to select the retained checkpoint.",
    )
    load_best_model_at_end: bool = Field(
        default=True,
        description="Restore the best retained checkpoint after training.",
    )
    resume_from_checkpoint: Path | None = Field(
        default=None,
        description="Checkpoint directory from which training should resume.",
    )
    report_to: list[Literal["wandb"]] = Field(
        default_factory=list,
        description="Training integrations receiving metrics.",
    )
    run_name: str | None = Field(
        default=None,
        description="Optional run name forwarded to reporting integrations.",
    )
    push_to_hub: bool = Field(
        default=False,
        description="Push the model and configured checkpoints to the Hub.",
    )
    hub_model_id: str | None = Field(
        default=None,
        description="Target Hugging Face repository in namespace/name form.",
    )
    hub_private_repo: bool = Field(
        default=False,
        description="Create a private repository when it does not exist.",
    )
    hub_strategy: Literal[
        "end", "every_save", "checkpoint", "all_checkpoints"
    ] = Field(
        default="every_save",
        description="Hugging Face Trainer checkpoint upload strategy.",
    )

    def model_post_init(self, __context: object) -> None:
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 cannot both be enabled")
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError(
                "hub_model_id is required when push_to_hub is enabled"
            )

    @model_validator(mode="after")
    def validate_best_model_metric(self) -> TrainingArgs:
        required_metric = {
            "edge_accuracy": "Affinity",
            "edge_precision": "Affinity",
            "edge_recall": "Affinity",
            "edge_f1": "Affinity",
            "partition_accuracy": "Partition",
        }.get(self.metric_for_best_model)
        if required_metric is not None and required_metric not in self.training_metrics:
            raise ValueError(
                f"metric_for_best_model={self.metric_for_best_model!r} requires "
                f"{required_metric!r} in training_metrics"
            )
        return self
