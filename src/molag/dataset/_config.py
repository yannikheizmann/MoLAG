from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from molag.utils.registry import Registry

from .modifiers import AnyModifier


class PoseConfig(BaseModel):
    """Position bounds and maximum tilt for tracker pose sampling."""

    model_config = ConfigDict(extra="forbid")

    x_min: float = Field(default=-100.0, description="Minimum X position in mm.")
    x_max: float = Field(default=100.0, description="Maximum X position in mm.")
    y_min: float = Field(default=-100.0, description="Minimum Y position in mm.")
    y_max: float = Field(default=100.0, description="Maximum Y position in mm.")
    z_min: float = Field(default=150.0, description="Minimum depth in mm.")
    z_max: float = Field(default=200.0, description="Maximum depth in mm.")
    max_tilt_deg: float = Field(
        default=85.0,
        description="Maximum tilt from the camera-facing orientation in degrees.",
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> PoseConfig:
        for axis in ("x", "y", "z"):
            lower = getattr(self, f"{axis}_min")
            upper = getattr(self, f"{axis}_max")
            if upper <= lower:
                raise ValueError(f"{axis}_max must be greater than {axis}_min")
        if self.z_min <= 0:
            raise ValueError("z_min must be positive")
        if not 0 <= self.max_tilt_deg <= 90:
            raise ValueError("max_tilt_deg must lie in [0, 90]")
        return self


class DatasetConfig(BaseModel):
    """Specification of a generated coordinate-set dataset."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    size: int = Field(default=50_000, ge=1)
    num_trackers: int | list[int] = Field(default=3)
    tracker: str = Field(default="Triangular")
    seed: int = Field(default=0)
    modifiers: list[AnyModifier] = Field(default_factory=list)
    pose: PoseConfig = Field(default_factory=PoseConfig)

    @field_validator("num_trackers")
    @classmethod
    def validate_num_trackers(cls, value: int | list[int]) -> int | list[int]:
        if isinstance(value, bool):
            raise ValueError("num_trackers must be an integer or [minimum, maximum]")
        if isinstance(value, int):
            if value < 1:
                raise ValueError("num_trackers must be at least 1")
            return value
        if len(value) != 2:
            raise ValueError("num_trackers range must contain exactly two values")
        minimum, maximum = value
        if minimum < 1:
            raise ValueError("minimum tracker count must be at least 1")
        if maximum < minimum:
            raise ValueError("maximum tracker count must not be below the minimum")
        return value

    @field_validator("tracker")
    @classmethod
    def validate_tracker(cls, value: str) -> str:
        Registry.get("TrackerBase", value)
        return value

    @property
    def num_trackers_range(self) -> tuple[int, int]:
        if isinstance(self.num_trackers, int):
            return self.num_trackers, self.num_trackers
        return self.num_trackers[0], self.num_trackers[1]

    @property
    def tracker_class(self) -> type:
        """Return the registered tracker implementation selected by the profile."""

        return Registry.get("TrackerBase", self.tracker)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetConfig:
        source = Path(path)
        with source.open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError(f"dataset profile must contain a mapping: {source}")
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
