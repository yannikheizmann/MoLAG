from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from molag.config import Args, ModelArgs, TrainingArgs
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
    assert args.model_args.edge_head_dims == [128]
    assert args.model_args.separation_weight == 0.46
    assert args.model_args.aggregation_beta == 1.0
    assert args.model_args.delta_nontree == 3.0
    assert args.model_args.eps_spur == 0.01
    assert args.model_args.conjunct_scaling_power == 0.5
    assert args.model_args.separation_scaling_power is None
    assert args.model_args.eligible_scene_mean is True
    assert args.model_args.supcon_weight == 0.0
    assert args.training_args.per_device_train_batch_size == 256
    assert args.training_args.gradient_accumulation_steps == 4
    assert args.training_args.learning_rate == 1e-3
    assert args.training_args.num_train_epochs == 10
    assert args.training_args.lr_scheduler_type == "cosine"
    assert args.training_args.warmup_ratio == 0.01
    assert args.training_args.fp16 is False
    assert args.training_args.bf16 is True


def test_nested_cli_syntax() -> None:
    args = ArgsParser(Args).parse(
        [
            "--dataset_args",
            "train_size=1000",
            "--model_args",
            "hidden_dims=[32,64]",
            "separation_weight=0.7",
            "--training_args",
            "learning_rate=0.0001",
            "bf16=False",
        ]
    )

    assert args.dataset_args.train_size == 1_000
    assert args.model_args.hidden_dims == [32, 64]
    assert args.model_args.separation_weight == 0.7
    assert args.training_args.learning_rate == 1e-4
    assert args.training_args.bf16 is False


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
        "training_args",
    }


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
    with pytest.raises(ValidationError) as error:
        ModelArgs.model_validate(model_args)

    assert field in str(error.value)


def test_training_configuration_rejects_incompatible_values() -> None:
    with pytest.raises(ValidationError):
        TrainingArgs(fp16=True, bf16=True)
