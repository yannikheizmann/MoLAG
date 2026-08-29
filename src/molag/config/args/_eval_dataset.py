from __future__ import annotations

from pathlib import Path

from pydantic import Field, PositiveInt, model_validator

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
    samples_per_tracker_count: PositiveInt | None = Field(
        default=None,
        description=(
            "Scenes retained for every tracker-count stratum. When omitted, "
            "generate one unstratified dataset of the configured size."
        ),
    )
    min_trackers: PositiveInt = Field(
        default=1,
        description="Smallest visible tracker count in a stratified dataset.",
    )
    max_trackers: PositiveInt = Field(
        default=10,
        description="Largest visible tracker count in a stratified dataset.",
    )
    max_attempts_per_tracker_count: PositiveInt | None = Field(
        default=None,
        description=(
            "Maximum candidate scenes considered for each stratum. The default "
            "is 100 times samples_per_tracker_count."
        ),
    )

    @model_validator(mode="after")
    def validate_stratification(self) -> EvalDatasetGenerationArgs:
        if self.max_trackers < self.min_trackers:
            raise ValueError("max_trackers must not be below min_trackers")
        if (
            self.samples_per_tracker_count is not None
            and self.max_attempts_per_tracker_count is not None
            and self.max_attempts_per_tracker_count
            < self.samples_per_tracker_count
        ):
            raise ValueError(
                "max_attempts_per_tracker_count must be at least "
                "samples_per_tracker_count"
            )
        return self
