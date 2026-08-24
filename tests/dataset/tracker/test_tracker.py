import numpy as np
import pytest
from pydantic import ValidationError

from molag.dataset.tracker import (
    TrackerBase,
    TrackerPose,
    TriangularTracker,
    TriangularTrackerCode,
    TriangularTrackerGeometry,
)
from molag.utils.registry import Registry


def test_triangular_tracker_is_registered() -> None:
    assert Registry.get("TrackerBase", "Triangular") is TriangularTracker
    assert issubclass(TriangularTracker, TrackerBase)


def test_all_configuration_ids_round_trip() -> None:
    for identifier in range(TriangularTrackerCode.num_unique_ids()):
        code = TriangularTrackerCode.from_id(identifier)
        assert code.to_id() == identifier


@pytest.mark.parametrize("identifier", [-1, 27])
def test_configuration_identifier_bounds(identifier: int) -> None:
    with pytest.raises(ValueError):
        TriangularTrackerCode.from_id(identifier)


def test_code_digits_are_limited_to_base_three() -> None:
    with pytest.raises(ValidationError):
        TriangularTrackerCode(c0=0, c1=1, c2=3)


def test_complete_geometry_has_seven_planar_leds() -> None:
    geometry = TriangularTrackerGeometry.from_code(
        TriangularTrackerCode.from_id(0)
    )
    points = geometry.as_array()
    assert points.shape == (7, 3)
    assert points.dtype == np.float64
    np.testing.assert_array_equal(points[:, 2], np.zeros(7))
    np.testing.assert_allclose(
        geometry.center,
        np.array([32.0, 18.475208614068, 0.0]),
    )


def test_tracker_combines_code_pose_and_geometry() -> None:
    pose = TrackerPose(R=np.eye(3), t=np.array([10.0, 20.0, 30.0]))
    code = TriangularTracker.CodeClass.from_id(5)
    tracker = TriangularTracker(
        code=code,
        pose=pose,
        geometry=TriangularTracker.GeometryClass.from_code(code),
    )

    assert tracker.id == 5
    assert tracker.geometry.code == tracker.code
    assert tracker.num_leds() == 7
    assert tracker.num_unique_ids() == 27
    world_coordinates = tracker.get_leds_world_coords()
    np.testing.assert_allclose(world_coordinates[:3].mean(axis=0), pose.t)


def test_pose_rejects_non_rotation_matrix() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        TrackerPose(R=np.ones((3, 3)), t=np.zeros(3))


def test_base_factory_constructs_registered_tracker_deterministically() -> None:
    first = TriangularTracker.from_id(5, np.random.default_rng(42))
    second = TriangularTracker.from_id(5, np.random.default_rng(42))

    assert first.id == 5
    np.testing.assert_array_equal(first.pose.R, second.pose.R)
    np.testing.assert_array_equal(first.pose.t, second.pose.t)
