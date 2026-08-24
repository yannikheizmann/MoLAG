import numpy as np
import pytest

from molag.dataset.tracker import CameraIntrinsics


def test_camera_intrinsics() -> None:
    assert CameraIntrinsics.aspect() == pytest.approx(4 / 3)
    assert CameraIntrinsics.hfov_deg() == pytest.approx(90.0)
    assert CameraIntrinsics.vfov_deg() == pytest.approx(73.7397952917)
    assert CameraIntrinsics.fx() == pytest.approx(720.0)
    assert CameraIntrinsics.fy() == pytest.approx(720.0)
    assert CameraIntrinsics.cx() == pytest.approx(720.0)
    assert CameraIntrinsics.cy() == pytest.approx(540.0)


def test_optical_axis_projects_to_principal_point() -> None:
    pixels, valid = CameraIntrinsics._project(np.array([[0.0, 0.0, 100.0]]))

    np.testing.assert_allclose(pixels, [[720.0, 540.0]])
    np.testing.assert_array_equal(valid, [True])


def test_image_y_axis_points_down() -> None:
    pixels, valid = CameraIntrinsics._project(
        np.array([[0.0, 10.0, 100.0], [0.0, -10.0, 100.0]])
    )

    assert pixels[0, 1] < CameraIntrinsics.cy()
    assert pixels[1, 1] > CameraIntrinsics.cy()
    np.testing.assert_array_equal(valid, [True, True])


def test_points_behind_camera_or_outside_frame_are_invalid() -> None:
    _, valid = CameraIntrinsics._project(
        np.array(
            [
                [0.0, 0.0, -1.0],
                [1_000.0, 0.0, 100.0],
                [0.0, 0.0, 100.0],
            ]
        )
    )

    np.testing.assert_array_equal(valid, [False, False, True])


def test_project_sample_preserves_tracker_and_led_axes() -> None:
    coordinates = np.zeros((2, 7, 3), dtype=np.float64)
    coordinates[:, :, 2] = 100.0

    pixels, valid = CameraIntrinsics.project_sample(coordinates, L=7)

    assert pixels.shape == (2, 7, 2)
    assert valid.shape == (2, 7)
    assert valid.all()


def test_project_sample_rejects_mismatched_led_count() -> None:
    with pytest.raises(ValueError, match="L=7"):
        CameraIntrinsics.project_sample(np.zeros((2, 6, 3)), L=7)
