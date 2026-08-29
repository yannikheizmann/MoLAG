from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import Dataset
import yaml

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
        return cls(x=x.tolist(), y=y.tolist())

    def to_tensors(self) -> dict[str, torch.Tensor]:
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
    def from_yaml(cls, path: str | Path) -> EvalDataset:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"evaluation dataset not found: {source}")
        with source.open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError(f"evaluation dataset must contain a mapping: {source}")
        return cls.model_validate(values)

    def to_yaml(self, path: str | Path) -> Path:
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
