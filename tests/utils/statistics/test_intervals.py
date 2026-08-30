import pytest

from molag.utils.statistics import clustered_ratio_interval, wilson_interval


def test_wilson_interval_contains_observed_proportion() -> None:
    lower, upper = wilson_interval(8, 10)

    assert lower < 0.8 < upper


def test_clustered_ratio_interval_uses_scene_level_counts() -> None:
    lower, upper = clustered_ratio_interval([2, 0, 1], [2, 2, 1])

    assert lower < 0.6 < upper


def test_clustered_ratio_interval_rejects_unpaired_counts() -> None:
    with pytest.raises(ValueError, match="same length"):
        clustered_ratio_interval([1], [1, 2])
