from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from molag.config import Args, LossArgs, ModelArgs, TrainingArgs
from molag.utils.argparsing import ArgsParser


def test_molag_defaults() -> None:
    args = Args()

    assert args.dataset_args.dataset_profile == Path(
        "src/molag/dataset/profiles/molag_standard.yaml"
    )
    assert args.config is None
    assert args.dataset_args.train_size == 5_000_000
    assert args.dataset_args.eval_size == 10_000
    assert args.model_args.hidden_dims == [
        128,
        256,
        512,
        1024,
        2048,
        2048,
        1024,
    ]
    assert args.model_args.message_alignment == 8
    assert args.model_args.edge_head_dims == [128]
    assert args.loss_args.separation_weight == 0.46
    assert args.loss_args.aggregation_beta == 1.0
    assert args.loss_args.delta_nontree == 3.0
    assert args.loss_args.eps_spur == 0.01
    assert args.loss_args.conjunct_scaling_power == 0.5
    assert args.loss_args.separation_scaling_power is None
    assert args.loss_args.eligible_scene_mean is True
    assert args.loss_args.supcon_weight == 0.0
    assert args.training_args.per_device_train_batch_size == 256
    assert args.training_args.gradient_accumulation_steps == 4
    assert args.training_args.learning_rate == 1e-3
    assert args.training_args.num_train_epochs == 10
    assert args.training_args.lr_scheduler_type == "cosine"
    assert args.training_args.warmup_ratio == 0.01
    assert args.training_args.fp16 is False
    assert args.training_args.bf16 is True
    assert args.training_args.report_to == []
    assert args.training_args.push_to_hub is False
    assert args.training_args.training_metrics == ["Affinity"]
    assert args.evaluation_args.hub_model_id is None
    assert args.evaluation_args.hub_revision is None


def test_enabled_hub_requires_repository() -> None:
    with pytest.raises(ValueError, match="hub_model_id is required"):
        TrainingArgs(push_to_hub=True)


def test_best_model_metric_requires_its_training_metric() -> None:
    with pytest.raises(ValueError, match="requires 'Partition'"):
        TrainingArgs(metric_for_best_model="partition_accuracy")

    args = TrainingArgs(
        training_metrics=["Affinity", "Partition"],
        metric_for_best_model="partition_accuracy",
    )
    assert args.metric_for_best_model == "partition_accuracy"


def test_nested_cli_syntax() -> None:
    args = ArgsParser(Args).parse(
        [
            "--dataset_args",
            "train_size=1000",
            "--model_args",
            "hidden_dims=[32,64]",
            "--loss_args",
            "separation_weight=0.7",
            "--training_args",
            "learning_rate=0.0001",
            "bf16=False",
            "report_to=['wandb']",
            "run_name=paper-run",
            "push_to_hub=True",
            "hub_model_id=example/molag",
        ]
    )

    assert args.dataset_args.train_size == 1_000
    assert args.model_args.hidden_dims == [32, 64]
    assert args.loss_args.separation_weight == 0.7
    assert args.training_args.learning_rate == 1e-4
    assert args.training_args.bf16 is False
    assert args.training_args.report_to == ["wandb"]
    assert args.training_args.run_name == "paper-run"
    assert args.training_args.push_to_hub is True
    assert args.training_args.hub_model_id == "example/molag"


def test_dataset_yaml_values_can_be_overridden_from_its_cli_group(
    tmp_path: Path,
) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset_args": {
                    "dataset_profile": "profiles/custom.yaml",
                    "train_size": 2_000,
                    "eval_size": 200,
                }
            }
        )
    )

    args = ArgsParser(Args).parse(
        [
            "--config",
            str(config),
            "--dataset_args",
            "eval_size=300",
        ]
    )

    assert args.dataset_args.dataset_profile == Path("profiles/custom.yaml")
    assert args.dataset_args.train_size == 2_000
    assert args.dataset_args.eval_size == 300
    assert args.config == config


def test_top_level_args_only_composes_argument_groups() -> None:
    assert set(Args.model_fields) == {
        "config",
        "dataset_args",
        "model_args",
        "loss_args",
        "training_args",
        "evaluation_args",
        "eval_dataset_generation_args",
    }


def test_molag_finetune_experiment_matches_defaults() -> None:
    args = ArgsParser(Args).parse(
        ["--config", "experiments/molag_finetune.yaml"]
    )

    assert args.model_copy(update={"config": None}) == Args()


@pytest.mark.parametrize(
    ("config", "samples_per_count", "seed", "attempts"),
    [
        (
            "experiments/molag_calibration_dataset.yaml",
            500,
            7_000_000,
            50_000,
        ),
        (
            "experiments/molag_test_dataset.yaml",
            5_000,
            10_000_000,
            1_000_000,
        ),
    ],
)
def test_held_out_dataset_experiment(
    config: str,
    samples_per_count: int,
    seed: int,
    attempts: int,
) -> None:
    args = ArgsParser(Args).parse(["--config", config])
    generation = args.eval_dataset_generation_args

    assert generation.samples_per_tracker_count == samples_per_count
    assert generation.seed == seed
    assert generation.max_attempts_per_tracker_count == attempts
    assert generation.min_trackers == 1
    assert generation.max_trackers == 10


def test_evaluation_experiment_uses_calibration_artifact() -> None:
    args = ArgsParser(Args).parse(
        ["--config", "experiments/molag_evaluate.yaml"]
    )
    evaluation = args.evaluation_args

    assert evaluation.dataset == Path(
        "evaluation/molag_test_1_to_10_50k_seed10000000.yaml"
    )
    assert evaluation.calibration_dataset == Path(
        "evaluation/molag_calibration_1_to_10_5k_seed7000000.yaml"
    )
    assert evaluation.threshold is None
    assert evaluation.metrics == ["Affinity", "Partition"]
    assert evaluation.threshold_min == 0.05
    assert evaluation.threshold_max == 0.95
    assert evaluation.threshold_step == 0.01
    assert evaluation.objective == "partition_accuracy"


@pytest.mark.parametrize(
    ("model_args", "field"),
    [
        ({"hidden_dims": []}, "hidden_dims"),
        ({"aggregation_beta": 0}, "aggregation_beta"),
        ({"conjunct_scaling_power": 1.1}, "conjunct_scaling_power"),
        ({"eps_spur": -0.1}, "eps_spur"),
    ],
)
def test_invalid_model_configuration_is_rejected(
    model_args: dict, field: str
) -> None:
    args_type = ModelArgs if "hidden_dims" in model_args else LossArgs
    with pytest.raises(ValidationError) as error:
        args_type.model_validate(model_args)

    assert field in str(error.value)


def test_training_configuration_rejects_incompatible_values() -> None:
    with pytest.raises(ValidationError):
        TrainingArgs(fp16=True, bf16=True)
