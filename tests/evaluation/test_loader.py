from pathlib import Path

import pytest
import torch

from molag.config import Args, LossArgs, ModelArgs, TrainingArgs
from molag.evaluation import ModelLoader
from molag.model import MoLAGModel


def test_model_is_reconstructed_from_run_directory(tmp_path: Path) -> None:
    args = Args(
        model_args=ModelArgs(hidden_dims=[4], edge_head_dims=[4]),
        loss_args=LossArgs(supcon_weight=0),
        training_args=TrainingArgs(output_dir=tmp_path, bf16=False),
    )
    args.save(tmp_path, format="yaml")
    source = MoLAGModel(args.model_args, args.loss_args)
    source.save_local(tmp_path / "model.safetensors")

    restored = ModelLoader.from_run_directory(tmp_path)

    assert restored.training is False
    assert restored.model_args == args.model_args
    assert restored.loss_args == args.loss_args
    for original, loaded in zip(source.parameters(), restored.parameters()):
        torch.testing.assert_close(original, loaded)


def test_missing_run_configuration_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run configuration"):
        ModelLoader.from_run_directory(tmp_path)


def test_missing_checkpoint_is_reported(tmp_path: Path) -> None:
    Args(training_args=TrainingArgs(bf16=False)).save(tmp_path, format="yaml")

    with pytest.raises(FileNotFoundError, match="no model checkpoint"):
        ModelLoader.from_run_directory(tmp_path)
