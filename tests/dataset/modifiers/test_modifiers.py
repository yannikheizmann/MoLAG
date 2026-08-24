import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from molag.dataset.modifiers import (
    AnyModifier,
    DropoutModifier,
    PixelNoiseModifier,
    SpuriousBlobsModifier,
)


def make_coordinates(
    num_trackers: int = 3,
    num_leds: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 640, (num_trackers * num_leds, 2)).astype(np.float32)
    tracker_ids = np.repeat(np.arange(num_trackers, dtype=np.int64), num_leds)
    led_ids = np.tile(np.arange(num_leds, dtype=np.int64), num_trackers)
    return x, np.stack((tracker_ids, led_ids), axis=1)


def test_dropout_respects_minimum_per_tracker() -> None:
    x, y = make_coordinates()
    modifier = DropoutModifier(
        drop_probability=1.0,
        min_leds_per_tracker=3,
    )

    modified_x, modified_y = modifier.apply(x, y, np.random.default_rng(0))

    assert len(modified_x) == len(modified_y) == 9
    for tracker_id in np.unique(modified_y[:, 0]):
        assert np.count_nonzero(modified_y[:, 0] == tracker_id) == 3
    assert modified_x.dtype == np.float32
    assert modified_y.dtype == np.int64


def test_dropout_does_not_remove_spurious_points() -> None:
    x, y = make_coordinates()
    x = np.concatenate((x, np.zeros((2, 2), dtype=np.float32)))
    y = np.concatenate((y, np.array([[-1, -1], [-2, -1]], dtype=np.int64)))

    _, modified_y = DropoutModifier(drop_probability=1.0).apply(
        x, y, np.random.default_rng(0)
    )

    np.testing.assert_array_equal(np.sort(modified_y[modified_y[:, 0] < 0, 0]), [-2, -1])


def test_spurious_points_have_unique_negative_ids_inside_bounding_box() -> None:
    x, y = make_coordinates()
    modifier = SpuriousBlobsModifier(min_blobs=5, max_blobs=5)

    modified_x, modified_y = modifier.apply(x, y, np.random.default_rng(0))

    spurious_x = modified_x[len(x) :]
    spurious_y = modified_y[len(y) :]
    assert len(spurious_x) == 5
    assert np.all(spurious_x >= x.min(axis=0))
    assert np.all(spurious_x <= x.max(axis=0))
    assert np.all(spurious_y[:, 0] < 0)
    assert len(np.unique(spurious_y[:, 0])) == 5
    np.testing.assert_array_equal(spurious_y[:, 1], np.full(5, -1))


def test_spurious_blob_range_is_validated() -> None:
    with pytest.raises(ValidationError):
        SpuriousBlobsModifier(min_blobs=5, max_blobs=2)


def test_pixel_noise_changes_only_coordinates() -> None:
    x, y = make_coordinates()

    modified_x, modified_y = PixelNoiseModifier(std=0.1).apply(
        x, y, np.random.default_rng(0)
    )

    assert not np.array_equal(modified_x, x)
    np.testing.assert_array_equal(modified_y, y)
    assert modified_x.dtype == np.float32


def test_modifier_stages() -> None:
    assert DropoutModifier().stage == "pre_norm"
    assert SpuriousBlobsModifier().stage == "pre_norm"
    assert PixelNoiseModifier().stage == "post_norm"


@pytest.mark.parametrize(
    ("values", "expected_type"),
    [
        ({"type": "Dropout"}, DropoutModifier),
        ({"type": "SpuriousBlobs"}, SpuriousBlobsModifier),
        ({"type": "PixelNoise"}, PixelNoiseModifier),
    ],
)
def test_modifier_union(values: dict, expected_type: type) -> None:
    modifier = TypeAdapter(AnyModifier).validate_python(values)
    assert isinstance(modifier, expected_type)


def test_unknown_modifier_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(AnyModifier).validate_python({"type": "Unknown"})
