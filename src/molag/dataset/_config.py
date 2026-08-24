from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class PoseConfig(BaseModel):
    """Position bounds and maximum tilt for tracker pose sampling."""

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
