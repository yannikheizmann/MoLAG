import json
from pathlib import Path

from torch import nn

from molag.config import CalibrationArgs
from molag.dataset import EvalDataset, EvalSample
from molag.main import Main
from molag.model.gnn.blocks import upper_tri_mask


class PositiveEdgeModel(nn.Module):
    def forward(self, data):
        pair_count = int(upper_tri_mask(data.edge_index).sum())
        return {"edge_logits": data.x.new_full((pair_count,), 100.0)}


def test_calibrate_saves_selected_threshold(monkeypatch, tmp_path: Path) -> None:
    dataset_path = EvalDataset(
        name="calibration",
        profile="profile.yaml",
        size=1,
        seed=10,
        created_at="2026-01-01T00:00:00+00:00",
        candidate_seed_ranges=[[10, 10]],
        samples=[
            EvalSample(
                x=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                y=[[0, 0], [0, 1], [0, 2]],
            )
        ],
    ).to_yaml(tmp_path / "calibration.yaml")
    output = tmp_path / "calibration.json"
    monkeypatch.setattr(
        "molag.main.ModelLoader.from_run_directory",
        lambda *args: PositiveEdgeModel(),
    )

    payload = Main._calibrate(
        CalibrationArgs(
            run_directory=tmp_path / "run",
            dataset=dataset_path,
            output=output,
            threshold_min=0.5,
            threshold_max=0.7,
            threshold_step=0.1,
        )
    )

    assert payload["threshold"] == 0.7
    assert payload["metric"] == "Partition"
    assert payload["objective"] == "partition_accuracy"
    assert payload["objective_value"] == 1.0
    assert json.loads(output.read_text()) == payload


def test_calibrate_entrypoint_routes_through_run(monkeypatch) -> None:
    modes: list[str] = []
    monkeypatch.setattr(Main, "run", lambda mode: modes.append(mode))

    Main.calibrate()

    assert modes == ["calibrate"]


def test_run_uses_calibration_argument_group(monkeypatch, tmp_path: Path) -> None:
    captured: list[CalibrationArgs] = []
    monkeypatch.setattr(Main, "_calibrate", lambda args: captured.append(args))

    Main.run(
        "calibrate",
        [
            "--calibration_args",
            f"run_directory={tmp_path / 'run'}",
            f"dataset={tmp_path / 'calibration.yaml'}",
            "threshold_min=0.1",
            "threshold_max=0.9",
            "threshold_step=0.1",
            "metric=Affinity",
            "objective=edge_f1",
        ],
    )

    assert captured[0].run_directory == tmp_path / "run"
    assert captured[0].threshold_min == 0.1
    assert captured[0].threshold_max == 0.9
    assert captured[0].metric == "Affinity"
    assert captured[0].objective == "edge_f1"
