from pathlib import Path

import pytest
import yaml
from pydantic import Field

from molag.utils.argparsing import (
    AdditionalArgsBase,
    ArgsParser,
    ConfigKeyError,
    PydanticArgsBase,
)


class TrainingArgs(AdditionalArgsBase):
    learning_rate: float = 1e-3
    num_train_epochs: int = 10


class Args(PydanticArgsBase):
    train_size: int = 100
    training_args: TrainingArgs = Field(default_factory=TrainingArgs)


def write_yaml(path: Path, values: dict) -> Path:
    path.write_text(yaml.safe_dump(values))
    return path


def test_defaults_are_used_without_overrides() -> None:
    args = ArgsParser(Args).parse([])

    assert args == Args()


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    config = write_yaml(
        tmp_path / "experiment.yaml",
        {"train_size": 200, "training_args": {"learning_rate": 5e-4}},
    )

    args = ArgsParser(Args).parse(["--config", str(config)])

    assert args.train_size == 200
    assert args.training_args.learning_rate == 5e-4
    assert args.training_args.num_train_epochs == 10


def test_explicit_cli_values_override_yaml(tmp_path: Path) -> None:
    config = write_yaml(
        tmp_path / "experiment.yaml",
        {"training_args": {"learning_rate": 5e-4, "num_train_epochs": 8}},
    )

    args = ArgsParser(Args).parse(
        [
            "--config",
            str(config),
            "--training_args",
            "learning_rate=0.0001",
        ]
    )

    assert args.training_args.learning_rate == 1e-4
    assert args.training_args.num_train_epochs == 8


def test_unknown_yaml_keys_are_reported_together(tmp_path: Path) -> None:
    config = write_yaml(
        tmp_path / "experiment.yaml",
        {
            "trian_size": 200,
            "training_args": {"learning_rae": 5e-4},
        },
    )

    with pytest.raises(ConfigKeyError) as error:
        ArgsParser(Args).parse(["--config", str(config)])

    message = str(error.value)
    assert "trian_size" in message
    assert "train_size" in message
    assert "training_args.learning_rae" in message
    assert "training_args.learning_rate" in message


def test_report_identifies_yaml_and_cli_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_yaml(tmp_path / "experiment.yaml", {"train_size": 200})

    ArgsParser(Args).parse(
        [
            "--config",
            str(config),
            "--training_args",
            "num_train_epochs=3",
        ]
    )

    output = capsys.readouterr().out
    assert f"Loaded configuration: {config}" in output
    assert "train_size" in output
    assert "training_args.num_train_epochs" in output


def test_required_field_can_be_supplied_by_yaml(tmp_path: Path) -> None:
    class RequiredArgs(PydanticArgsBase):
        output_name: str

    config = write_yaml(tmp_path / "experiment.yaml", {"output_name": "paper-run"})

    args = ArgsParser(RequiredArgs).parse(["--config", str(config)])

    assert args.output_name == "paper-run"
