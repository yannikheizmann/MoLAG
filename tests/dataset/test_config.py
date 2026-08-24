import pytest
from pydantic import ValidationError

from molag.dataset import PoseConfig


def test_pose_defaults() -> None:
    config = PoseConfig()
    assert config.x_min == -100.0
    assert config.x_max == 100.0
    assert config.z_min == 150.0
    assert config.z_max == 200.0
    assert config.max_tilt_deg == 85.0


@pytest.mark.parametrize(
    "values",
    [
        {"x_min": 10.0, "x_max": 10.0},
        {"z_min": 0.0},
        {"max_tilt_deg": 91.0},
    ],
)
def test_invalid_pose_configuration(values: dict) -> None:
    with pytest.raises(ValidationError):
        PoseConfig.model_validate(values)
