from pathlib import Path

import numpy as np
import pytest

from molag.evaluation import AffinityMetrics, Evaluator
from molag.inference import PredictionCache, ScenePrediction


def make_scene(offset: float = 0.0) -> ScenePrediction:
    return ScenePrediction(
        coordinates=np.array(
            [[offset, 0.0], [offset + 1.0, 0.0]], dtype=np.float32
        ),
        point_labels=np.array([[0, 0], [0, 1]], dtype=np.int64),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        edge_logits=np.array([0.75], dtype=np.float32),
        edge_labels=np.array([1], dtype=np.int64),
    )


def test_prediction_cache_round_trip_is_pickle_free(tmp_path: Path) -> None:
    cache = PredictionCache([make_scene(), make_scene(2.0)])

    output = cache.to_npz(tmp_path / "predictions.npz")
    restored = PredictionCache.from_npz(output)

    assert len(restored) == 2
    np.testing.assert_array_equal(restored[0].coordinates, cache[0].coordinates)
    np.testing.assert_array_equal(restored[1].point_labels, cache[1].point_labels)
    np.testing.assert_array_equal(restored[1].edge_index, cache[1].edge_index)
    np.testing.assert_array_equal(restored[1].edge_logits, cache[1].edge_logits)
    np.testing.assert_array_equal(restored[1].edge_labels, cache[1].edge_labels)


def test_prediction_cache_validates_scene_shapes() -> None:
    with pytest.raises(ValueError, match="point_labels"):
        ScenePrediction(
            coordinates=np.zeros((2, 2), dtype=np.float32),
            point_labels=np.zeros((2, 1), dtype=np.int64),
            edge_index=np.array([[0], [1]], dtype=np.int64),
            edge_logits=np.array([0.0], dtype=np.float32),
            edge_labels=np.array([0], dtype=np.int64),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("coordinates", np.array([[np.nan, 0.0], [1.0, 0.0]]), "finite"),
        ("edge_logits", np.array([np.inf]), "finite"),
        ("point_labels", np.zeros((2, 2), dtype=np.float32), "integer"),
    ],
)
def test_scene_prediction_rejects_invalid_values(
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    values = make_scene().__dict__ | {field: value}

    with pytest.raises(ValueError, match=message):
        ScenePrediction(**values)


def test_prediction_cache_rejects_missing_arrays(tmp_path: Path) -> None:
    path = tmp_path / "missing.npz"
    np.savez(path, format_version=np.asarray(1))

    with pytest.raises(ValueError, match="missing arrays"):
        PredictionCache.from_npz(path)


@pytest.mark.parametrize(
    ("array", "value", "message"),
    [
        ("node_offsets", np.array([1, 2]), "start at zero"),
        ("node_offsets", np.array([0, 2, 1]), "nondecreasing"),
        ("edge_offsets", np.array([0, 1, 1]), "equal scenes"),
        ("coordinates", np.zeros((1, 2)), "node_offsets"),
        ("edge_logits", np.zeros(2), "edge_offsets"),
    ],
)
def test_prediction_cache_rejects_inconsistent_archive(
    tmp_path: Path,
    array: str,
    value: np.ndarray,
    message: str,
) -> None:
    valid_path = PredictionCache([make_scene()]).to_npz(tmp_path / "valid.npz")
    with np.load(valid_path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values[array] = value
    invalid_path = tmp_path / f"invalid_{array}.npz"
    np.savez(invalid_path, **values)

    with pytest.raises(ValueError, match=message):
        PredictionCache.from_npz(invalid_path)


def test_cached_predictions_can_be_evaluated_without_model() -> None:
    metrics = Evaluator.evaluate_predictions(
        PredictionCache([make_scene()]),
        AffinityMetrics(threshold=0.5),
    )

    assert metrics == {
        "edge_accuracy": 1.0,
        "edge_precision": 1.0,
        "edge_recall": 1.0,
        "edge_f1": 1.0,
    }
