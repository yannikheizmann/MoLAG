import numpy as np
import pytest
import torch

from molag.dataset import DatasetConfig, PoseConfig, TrackingDataset
from molag.dataset.modifiers import (
    DropoutModifier,
    PixelNoiseModifier,
    SpuriousBlobsModifier,
)
from molag.dataset.tracker import TriangularTracker


def centred_pose() -> PoseConfig:
    return PoseConfig(
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        z_min=300.0,
        z_max=301.0,
        max_tilt_deg=5.0,
    )


def test_initialisation_and_factory() -> None:
    config = DatasetConfig(
        size=5,
        num_trackers=[1, 3],
        pose=centred_pose(),
    )

    dataset = TrackingDataset.from_config(config)

    assert len(dataset) == 5
    assert dataset.num_trackers_range == (1, 3)


def test_item_tensor_contract() -> None:
    dataset = TrackingDataset(
        size=1,
        num_trackers=3,
        TrackerClass=TriangularTracker,
        pose_config=centred_pose(),
    )

    item = dataset[0]

    assert set(item) == {"x", "y"}
    assert item["x"].dtype == torch.float32
    assert item["y"].dtype == torch.int64
    assert item["x"].ndim == item["y"].ndim == 2
    assert item["x"].shape == item["y"].shape
    assert item["x"].shape[1] == 2


def test_coordinates_are_mean_centred_and_max_norm_scaled() -> None:
    dataset = TrackingDataset(
        size=3,
        num_trackers=3,
        TrackerClass=TriangularTracker,
        pose_config=centred_pose(),
    )

    for index in range(len(dataset)):
        coordinates = dataset[index]["x"].numpy()
        np.testing.assert_allclose(coordinates.mean(axis=0), 0.0, atol=1e-6)
        assert np.linalg.norm(coordinates, axis=1).max() == pytest.approx(1.0)


def test_generation_is_deterministic_by_seed_and_index() -> None:
    config = DatasetConfig(size=3, num_trackers=3, seed=42, pose=centred_pose())
    first = TrackingDataset.from_config(config)
    second = TrackingDataset.from_config(config)

    for index in range(3):
        assert torch.equal(first[index]["x"], second[index]["x"])
        assert torch.equal(first[index]["y"], second[index]["y"])


def test_exhausted_occlusion_filter_returns_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("molag.dataset._dataset.MAX_SCENE_GENERATION_ATTEMPTS", 1)
    monkeypatch.setattr(TrackingDataset, "_has_occlusion", lambda self, sample: True)
    dataset = TrackingDataset(
        size=1,
        num_trackers=1,
        TrackerClass=TriangularTracker,
        pose_config=centred_pose(),
    )

    item = dataset[0]

    assert set(item) == {"x", "y", "warning"}
    assert "occlusion filter" in item["warning"]


def test_spurious_modifier_adds_negative_tracker_ids() -> None:
    config = DatasetConfig(
        size=2,
        num_trackers=2,
        pose=centred_pose(),
        modifiers=[
            SpuriousBlobsModifier(probability=1.0, min_blobs=3, max_blobs=3)
        ],
    )

    dataset = TrackingDataset.from_config(config)

    for index in range(2):
        tracker_ids = dataset[index]["y"][:, 0]
        assert torch.count_nonzero(tracker_ids < 0) == 3


def test_dropout_and_noise_are_applied() -> None:
    base = DatasetConfig(size=1, num_trackers=2, pose=centred_pose())
    modified = base.model_copy(
        update={
            "modifiers": [
                DropoutModifier(
                    probability=1.0,
                    drop_probability=1.0,
                    min_leds_per_tracker=3,
                ),
                PixelNoiseModifier(probability=1.0, std=0.1),
            ]
        }
    )

    clean_item = TrackingDataset.from_config(base)[0]
    modified_item = TrackingDataset.from_config(modified)[0]

    assert len(modified_item["x"]) < len(clean_item["x"])
    assert not torch.equal(modified_item["x"], clean_item["x"][: len(modified_item["x"])])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": 0, "num_trackers": 1},
        {"size": 1, "num_trackers": 0},
        {"size": 1, "num_trackers": (3, 2)},
        {"size": 1, "num_trackers": 28},
    ],
)
def test_invalid_dataset_arguments(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        TrackingDataset(TrackerClass=TriangularTracker, **kwargs)
