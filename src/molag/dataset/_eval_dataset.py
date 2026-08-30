"""Materialise deterministic scenes for calibration and evaluation."""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import Dataset

from ._config import DatasetConfig
from ._dataset import TrackingDataset

LOGGER = logging.getLogger(__name__)


class EvalSample(BaseModel):
    """Coordinates and labels for one frozen evaluation scene."""

    model_config = ConfigDict(extra="forbid")

    x: list[list[float]]
    y: list[list[int]]

    @model_validator(mode="after")
    def validate_shapes(self) -> EvalSample:
        """Validate coordinate and label shapes and finite coordinates."""
        if not self.x:
            raise ValueError("evaluation samples must contain at least one point")
        if len(self.x) != len(self.y):
            raise ValueError("x and y must contain the same number of points")
        if any(len(row) != 2 for row in self.x):
            raise ValueError("x must have shape (num_points, 2)")
        if any(len(row) != 2 for row in self.y):
            raise ValueError("y must have shape (num_points, 2)")
        if not np.isfinite(np.asarray(self.x)).all():
            raise ValueError("x must contain only finite coordinates")
        return self

    @classmethod
    def from_tensors(cls, x: torch.Tensor, y: torch.Tensor) -> EvalSample:
        """Convert coordinate and label tensors into a serialisable sample."""
        return cls(x=x.tolist(), y=y.tolist())

    def to_tensors(self) -> dict[str, torch.Tensor]:
        """Convert the stored values to model input tensors."""
        return {
            "x": torch.from_numpy(np.asarray(self.x, dtype=np.float32)),
            "y": torch.from_numpy(np.asarray(self.y, dtype=np.int64)),
        }


class EvalDataset(BaseModel, Dataset):
    """Fixed evaluation scenes persisted independently of the generator."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    profile: str = Field(min_length=1)
    size: int = Field(ge=1)
    seed: int
    created_at: str = Field(min_length=1)
    candidate_seed_ranges: list[list[int]] = Field(min_length=1)
    samples: list[EvalSample]

    @model_validator(mode="after")
    def validate_metadata(self) -> EvalDataset:
        """Validate sample counts and candidate-seed ranges."""
        if self.size != len(self.samples):
            raise ValueError("size must equal the number of frozen samples")
        for seed_range in self.candidate_seed_ranges:
            if len(seed_range) != 2 or seed_range[1] < seed_range[0]:
                raise ValueError(
                    "candidate seed ranges must contain [minimum, maximum]"
                )
        return self

    @classmethod
    def generate(
        cls,
        name: str,
        profile_path: str | Path,
        size: int,
        seed: int,
        description: str = "",
    ) -> EvalDataset:
        """Generate a fixed sequence of scenes from consecutive candidate seeds."""
        profile_source = Path(profile_path)
        config = DatasetConfig.from_yaml(profile_source).model_copy(
            update={"size": size, "seed": seed}
        )
        generated = TrackingDataset.from_config(config)
        samples: list[EvalSample] = []
        for index in range(size):
            item = generated[index]
            samples.append(EvalSample.from_tensors(item["x"], item["y"]))
            if (index + 1) % 100 == 0:
                LOGGER.info("Generated %d of %d evaluation scenes.", index + 1, size)

        return cls(
            name=name,
            description=description,
            profile=str(profile_source),
            size=size,
            seed=seed,
            created_at=datetime.now(timezone.utc).isoformat(),
            candidate_seed_ranges=[[seed, seed + size - 1]],
            samples=samples,
        )

    @classmethod
    def generate_stratified(
        cls,
        name: str,
        profile_path: str | Path,
        samples_per_tracker_count: int,
        min_trackers: int,
        max_trackers: int,
        seed: int,
        max_attempts_per_tracker_count: int | None = None,
        description: str = "",
    ) -> EvalDataset:
        """Generate equal-sized strata by visible tracker count."""
        if samples_per_tracker_count < 1:
            raise ValueError("samples_per_tracker_count must be positive")
        if min_trackers < 1:
            raise ValueError("min_trackers must be positive")
        if max_trackers < min_trackers:
            raise ValueError("max_trackers must not be below min_trackers")

        attempts = (
            max_attempts_per_tracker_count
            if max_attempts_per_tracker_count is not None
            else 100 * samples_per_tracker_count
        )
        if attempts < samples_per_tracker_count:
            raise ValueError(
                "max_attempts_per_tracker_count must be at least "
                "samples_per_tracker_count"
            )

        profile_source = Path(profile_path)
        profile = DatasetConfig.from_yaml(profile_source)
        samples: list[EvalSample] = []
        candidate_seed_ranges: list[list[int]] = []

        for offset, tracker_count in enumerate(
            range(min_trackers, max_trackers + 1)
        ):
            stratum_seed = seed + offset * attempts
            config = profile.model_copy(
                update={
                    "size": attempts,
                    "seed": stratum_seed,
                    "num_trackers": tracker_count,
                }
            )
            generated = TrackingDataset.from_config(config)
            accepted = 0
            for index in range(attempts):
                item = generated[index]
                tracker_labels = item["y"][:, 0]
                visible_trackers = int(
                    tracker_labels[tracker_labels >= 0].unique().numel()
                )
                if visible_trackers != tracker_count:
                    continue
                samples.append(EvalSample.from_tensors(item["x"], item["y"]))
                accepted += 1
                if accepted == samples_per_tracker_count:
                    break

            if accepted != samples_per_tracker_count:
                raise RuntimeError(
                    f"accepted only {accepted} of {samples_per_tracker_count} "
                    f"scenes with {tracker_count} visible trackers after "
                    f"{attempts} candidates"
                )
            candidate_seed_ranges.append(
                [stratum_seed, stratum_seed + index]
            )
            LOGGER.info(
                "Generated %d scenes with %d visible trackers from %d candidates.",
                accepted,
                tracker_count,
                index + 1,
            )

        random.Random(seed).shuffle(samples)
        return cls(
            name=name,
            description=description,
            profile=str(profile_source),
            size=len(samples),
            seed=seed,
            created_at=datetime.now(timezone.utc).isoformat(),
            candidate_seed_ranges=candidate_seed_ranges,
            samples=samples,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalDataset:
        """Load and validate a frozen dataset from YAML."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"evaluation dataset not found: {source}")
        with source.open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError(f"evaluation dataset must contain a mapping: {source}")
        return cls.model_validate(values)

    def to_yaml(self, path: str | Path) -> Path:
        """Write the frozen dataset to YAML and return its path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                self.model_dump(mode="json"),
                stream,
                sort_keys=False,
                default_flow_style=False,
            )
        return destination

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index].to_tensors()
