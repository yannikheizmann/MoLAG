from pathlib import Path

import yaml

from molag.config import Args, DatasetArgs, LossArgs, ModelArgs, TrainingArgs
from molag.main import Main
from molag.evaluation import CombinedMetrics


class TrainerStub:
    instance: "TrainerStub | None" = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        TrainerStub.instance = self

    def train(self) -> dict[str, float]:
        return {"train_loss": 1.0}


def test_execute_builds_disjoint_splits_and_saves_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump({"size": 1, "num_trackers": 2, "seed": 17})
    )
    output_dir = tmp_path / "results"
    args = Args(
        dataset_args=DatasetArgs(
            dataset_profile=profile_path,
            train_size=8,
            eval_size=3,
        ),
        model_args=ModelArgs(hidden_dims=[4], edge_head_dims=[4]),
        loss_args=LossArgs(),
        training_args=TrainingArgs(output_dir=output_dir, bf16=False),
    )
    monkeypatch.setattr(
        "molag.main.Trainer", TrainerStub
    )

    metrics = Main._finetune(args)

    trainer = TrainerStub.instance
    assert trainer is not None
    assert len(trainer.kwargs["train_dataset"]) == 8
    assert trainer.kwargs["train_dataset"]._seed == 17
    assert len(trainer.kwargs["eval_dataset"]) == 3
    assert trainer.kwargs["eval_dataset"]._seed == 25
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "dataset_profile.yaml").exists()
    assert isinstance(trainer.kwargs["metrics"], CombinedMetrics)
    assert metrics == {"train_loss": 1.0}


def test_run_parses_arguments(monkeypatch) -> None:
    captured: list[Args] = []

    def finetune_stub(args: Args) -> dict[str, float]:
        captured.append(args)
        return {}

    monkeypatch.setattr(Main, "_finetune", finetune_stub)

    result = Main.run(
        "finetune",
        [
            "--dataset_args",
            "train_size=12",
            "eval_size=4",
            "--training_args",
            "bf16=False",
        ]
    )

    assert captured[0].dataset_args.train_size == 12
    assert captured[0].dataset_args.eval_size == 4
    assert captured[0].training_args.bf16 is False
    assert result is None


def test_help_lists_configuration_groups(capsys) -> None:
    try:
        Main.run("finetune", ["--help"])
    except SystemExit as error:
        assert error.code == 0

    help_text = capsys.readouterr().out
    assert "--config" in help_text
    assert "--dataset_args" in help_text
    assert "--training_args" in help_text


def test_finetune_entrypoint_routes_through_run(monkeypatch) -> None:
    modes: list[str] = []
    monkeypatch.setattr(Main, "run", lambda mode: modes.append(mode))

    Main.finetune()

    assert modes == ["finetune"]
