import json
from pathlib import Path

from molag.config import (
    Args,
    EvaluationArgs,
    LossArgs,
    ModelArgs,
    TrainingArgs,
)
from molag.dataset import EvalDataset
from molag.main import Main
from molag.model import MoLAGModel

PROFILE = Path("src/molag/dataset/profiles/molag_standard.yaml")


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
    output = tmp_path / "results" / "metrics.json"

    metrics = Main._evaluate(
        EvaluationArgs(
            run_directory=run_directory,
            dataset=dataset_path,
            output=output,
            batch_size=2,
        )
    )

    saved = json.loads(output.read_text())
    assert "edge_accuracy" in metrics
    assert "partition_accuracy" in metrics
    assert saved["metrics"] == metrics
    assert saved["dataset_name"] == "evaluation"
    assert saved["candidate_seed_ranges"] == [[100, 102]]


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
