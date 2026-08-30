import json
from pathlib import Path

from torch import nn

from molag.config import (
    Args,
    EvaluationArgs,
    LossArgs,
    ModelArgs,
    TrainingArgs,
)
from molag.dataset import EvalDataset, EvalSample
from molag.inference import PredictionCache
from molag.main import Main
from molag.model import MoLAGModel
from molag.model.gnn.blocks import upper_tri_mask

PROFILE = Path("src/molag/dataset/profiles/molag_standard.yaml")


class PositiveEdgeModel(nn.Module):
    def forward(self, data):
        pair_count = int(upper_tri_mask(data.edge_index).sum())
        return {"edge_logits": data.x.new_full((pair_count,), 100.0)}


def test_evaluate_runs_model_and_saves_results(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    model_args = ModelArgs(hidden_dims=[4], edge_head_dims=[4])
    loss_args = LossArgs(supcon_weight=0)
    Args(
        model_args=model_args,
        loss_args=loss_args,
        training_args=TrainingArgs(output_dir=run_directory, bf16=False),
    ).save(run_directory, format="yaml")
    MoLAGModel(model_args, loss_args).save_local(
        run_directory / "model.safetensors"
    )
    dataset_path = EvalDataset.generate(
        "evaluation",
        PROFILE,
        size=3,
        seed=100,
    ).to_yaml(tmp_path / "evaluation.yaml")
    output = run_directory / "evaluation.json"

    metrics = Main._evaluate(
        EvaluationArgs(
            run_directory=run_directory,
            dataset=dataset_path,
            batch_size=2,
            threshold=0.5,
        )
    )

    saved = json.loads(output.read_text())
    assert "edge_accuracy" in metrics
    assert "partition_accuracy" in metrics
    assert saved["metrics"] == metrics
    assert "by_n_trackers" in saved["breakdown"]
    assert "by_visible_leds" in saved["breakdown"]
    assert (run_directory / "samples.csv").is_file()
    assert (run_directory / "tracker_samples.csv").is_file()
    predictions = PredictionCache.from_npz(run_directory / "predictions.npz")
    assert len(predictions) == 3
    assert saved["predictions"] == str(run_directory / "predictions.npz")
    assert saved["dataset_name"] == "evaluation"
    assert saved["candidate_seed_ranges"] == [[100, 102]]
    assert saved["threshold"] == 0.5
    assert saved["calibration"] is None


def test_evaluate_calibrates_before_test_evaluation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "config.yaml").write_text("test: provenance\n")
    (run_directory / "model.safetensors").write_bytes(b"test checkpoint")
    calibration_dataset = EvalDataset(
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
    test_dataset = EvalDataset.from_yaml(calibration_dataset).model_copy(
        update={"name": "test", "seed": 20, "candidate_seed_ranges": [[20, 20]]}
    ).to_yaml(tmp_path / "test.yaml")
    monkeypatch.setattr(
        "molag.main.ModelLoader.from_run_directory",
        lambda *args: PositiveEdgeModel(),
    )

    metrics = Main._evaluate(
        EvaluationArgs(
            run_directory=run_directory,
            calibration_dataset=calibration_dataset,
            dataset=test_dataset,
            threshold_min=0.5,
            threshold_max=0.7,
            threshold_step=0.1,
        )
    )

    calibration = json.loads((run_directory / "calibration.json").read_text())
    evaluation = json.loads((run_directory / "evaluation.json").read_text())
    assert calibration["threshold"] == 0.7
    assert len(calibration["results"]) == 3
    calibration_predictions = PredictionCache.from_npz(
        run_directory / "calibration_predictions.npz"
    )
    assert len(calibration_predictions) == 1
    assert calibration["predictions"] == str(
        run_directory / "calibration_predictions.npz"
    )
    assert evaluation["threshold"] == 0.7
    assert evaluation["calibration"] == str(run_directory / "calibration.json")
    assert evaluation["metrics"] == metrics


def test_run_uses_evaluation_argument_schema(monkeypatch, tmp_path: Path) -> None:
    captured: list[EvaluationArgs] = []
    monkeypatch.setattr(Main, "_evaluate", lambda args: captured.append(args))

    Main.run(
        "evaluate",
        [
            "--evaluation_args",
            f"run_directory={tmp_path / 'run'}",
            f"dataset={tmp_path / 'dataset.yaml'}",
            "batch_size=16",
        ],
    )

    assert captured[0].run_directory == tmp_path / "run"
    assert captured[0].dataset == tmp_path / "dataset.yaml"
    assert captured[0].batch_size == 16


def test_evaluate_entrypoint_routes_through_run(monkeypatch) -> None:
    modes: list[str] = []
    monkeypatch.setattr(Main, "run", lambda mode: modes.append(mode))

    Main.evaluate()

    assert modes == ["evaluate"]
