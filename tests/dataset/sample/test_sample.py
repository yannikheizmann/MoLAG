import numpy as np
import pytest

from molag.dataset import PoseConfig, Sample
from molag.dataset.tracker import TriangularTracker
from molag.utils.registry import Registry


def test_sample_uses_registered_tracker_implementation() -> None:
    TrackerClass = Registry.get("TrackerBase", "Triangular")

    sample = Sample(3, TrackerClass, np.random.default_rng(42))

    assert all(isinstance(tracker, TriangularTracker) for tracker in sample.get_trackers())


def test_tracker_codes_are_unique_within_scene() -> None:
    sample = Sample(10, TriangularTracker, np.random.default_rng(42))

    identifiers = [tracker.id for tracker in sample.get_trackers()]

    assert len(identifiers) == len(set(identifiers))


def test_sample_generation_is_deterministic_for_seed() -> None:
    first = Sample(3, TriangularTracker, np.random.default_rng(42))
    second = Sample(3, TriangularTracker, np.random.default_rng(42))

    np.testing.assert_array_equal(first.get_world_coords(), second.get_world_coords())


def test_custom_pose_bounds_are_applied() -> None:
    pose_cfg = PoseConfig(
        x_min=-1.0,
        x_max=1.0,
        y_min=-2.0,
        y_max=2.0,
        z_min=300.0,
        z_max=301.0,
        max_tilt_deg=0.0,
    )

    sample = Sample(
        5,
        TriangularTracker,
        np.random.default_rng(42),
        pose_cfg=pose_cfg,
    )

    translations = np.stack([tracker.pose.t for tracker in sample.get_trackers()])
    assert np.all((-1.0 <= translations[:, 0]) & (translations[:, 0] <= 1.0))
    assert np.all((-2.0 <= translations[:, 1]) & (translations[:, 1] <= 2.0))
    assert np.all((300.0 <= translations[:, 2]) & (translations[:, 2] <= 301.0))


def test_get_data_returns_coordinates_and_labels() -> None:
    sample = Sample(3, TriangularTracker, np.random.default_rng(42))

    coordinates, labels = sample.get_data()

    assert coordinates.ndim == 2
    assert coordinates.shape[1] == 2
    assert coordinates.dtype == np.float32
    assert labels.shape == coordinates.shape
    assert labels.dtype == np.int64
    assert set(labels[:, 0]).issubset(
        {tracker.id for tracker in sample.get_trackers()}
    )
    assert np.all((0 <= labels[:, 1]) & (labels[:, 1] < 7))


@pytest.mark.parametrize("num_trackers", [0, -1, 28])
def test_invalid_tracker_count_is_rejected(num_trackers: int) -> None:
    with pytest.raises(ValueError):
        Sample(num_trackers, TriangularTracker, np.random.default_rng(42))
