import json

import numpy as np
import pytest
from torch import nn

from molag.inference import MoLAGPredictor
from molag.model.gnn.blocks import upper_tri_mask


class PositiveEdgeModel(nn.Module):
    def forward(self, data):
        pair_count = int(upper_tri_mask(data.edge_index).sum())
        return {"edge_logits": data.x.new_full((pair_count,), 100.0)}


class RecordingModel(PositiveEdgeModel):
    def forward(self, data):
        self.coordinates = data.x.detach().cpu()
        return super().forward(data)


def test_predictor_returns_affinities_and_groups() -> None:
    predictor = MoLAGPredictor(PositiveEdgeModel(), threshold=0.5)

    result = predictor.predict(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    )

    assert result.edge_index.shape == (2, 3)
    np.testing.assert_allclose(result.affinities, 1.0)
    assert len(result.groups) == 1
    np.testing.assert_array_equal(result.groups[0], [0, 1, 2])


def test_predictor_loads_threshold_from_run(monkeypatch, tmp_path) -> None:
    (tmp_path / "calibration.json").write_text(
        json.dumps({"threshold": 0.73})
    )
    monkeypatch.setattr(
        "molag.evaluation.ModelLoader.from_run_directory",
        lambda *args: PositiveEdgeModel(),
    )

    predictor = MoLAGPredictor.from_run_directory(tmp_path)
    result = predictor.predict([[0.0, 0.0], [1.0, 0.0]])

    assert len(result.groups) == 1


def test_predictor_loads_model_and_calibration_from_hub(
    monkeypatch,
    tmp_path,
) -> None:
    (tmp_path / "calibration.json").write_text(json.dumps({"threshold": 0.73}))
    monkeypatch.setattr(
        "molag.evaluation.ModelLoader.from_hub",
        lambda *args, **kwargs: tmp_path,
    )
    monkeypatch.setattr(
        "molag.evaluation.ModelLoader.from_run_directory",
        lambda *args, **kwargs: PositiveEdgeModel(),
    )

    predictor = MoLAGPredictor.from_hub(
        "organisation/model",
        revision="paper",
        device="cpu",
    )

    assert len(predictor.predict([[0.0, 0.0], [1.0, 0.0]]).groups) == 1


def test_predictor_normalizes_raw_coordinates() -> None:
    model = RecordingModel()
    predictor = MoLAGPredictor(model, threshold=0.5)

    predictor.predict([[10.0, 20.0], [14.0, 20.0], [10.0, 23.0]])

    np.testing.assert_allclose(model.coordinates.mean(dim=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(
        model.coordinates.norm(dim=1).max(),
        1.0,
        atol=1e-7,
    )


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_predictor_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        MoLAGPredictor(PositiveEdgeModel(), threshold)
