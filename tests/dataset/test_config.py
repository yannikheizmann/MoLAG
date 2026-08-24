from pathlib import Path

import pytest
from pydantic import ValidationError

from molag.dataset import DatasetConfig, PoseConfig
from molag.dataset.modifiers import (
    DropoutModifier,
    PixelNoiseModifier,
    SpuriousBlobsModifier,
)
from molag.dataset.tracker import TriangularTracker

PROFILE = (
    Path(__file__).parents[2]
    / "src/molag/dataset/profiles/molag_standard.yaml"
)


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


def test_dataset_profile_loads_complete_specification() -> None:
    config = DatasetConfig.from_yaml(PROFILE)

    assert config.size == 50_000
    assert config.num_trackers_range == (1, 10)
    assert config.tracker == "Triangular"
    assert config.tracker_class is TriangularTracker
    assert config.seed == 0
    assert config.pose.z_min == 100.0
    assert config.pose.z_max == 600.0
    assert [type(modifier) for modifier in config.modifiers] == [
        DropoutModifier,
        SpuriousBlobsModifier,
        PixelNoiseModifier,
    ]


def test_dataset_profile_round_trip(tmp_path: Path) -> None:
    original = DatasetConfig.from_yaml(PROFILE)
    destination = original.to_yaml(tmp_path / "copied-profile.yaml")

    loaded = DatasetConfig.from_yaml(destination)

    assert loaded == original


@pytest.mark.parametrize("num_trackers", [0, [0, 3], [3, 2], [1, 2, 3]])
def test_invalid_tracker_count(num_trackers: int | list[int]) -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(num_trackers=num_trackers)


def test_unknown_tracker_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Missing"):
        DatasetConfig(tracker="Missing")


def test_unknown_profile_fields_are_rejected(tmp_path: Path) -> None:
    profile = tmp_path / "invalid.yaml"
    profile.write_text("size: 10\nnum_trakcers: 3\n")

    with pytest.raises(ValidationError, match="num_trakcers"):
        DatasetConfig.from_yaml(profile)


def test_unknown_pose_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="max_tilt_degrees"):
        DatasetConfig(pose={"max_tilt_degrees": 85.0})
