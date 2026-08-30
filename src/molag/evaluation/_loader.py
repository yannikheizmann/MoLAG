"""Reconstruct trained MoLAG models from local or Hub run artefacts."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download

from molag.config import Args
from molag.model import MoLAGModel
from molag.utils import resolve_device


class ModelLoader:
    """MoLAG model loader for the standard finetuning run layout."""

    CHECKPOINT_FILENAMES = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.pt",
    )

    @classmethod
    def from_hub(
        cls,
        model_id: str,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
    ) -> Path:
        """Download and validate a model snapshot in the local run layout."""
        if not model_id.strip():
            raise ValueError("model_id must not be empty")
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                allow_patterns=[
                    "config.yaml",
                    "model.safetensors",
                    "pytorch_model.bin",
                    "model.pt",
                    "calibration.json",
                    "dataset_profile.yaml",
                ],
            )
        )
        cls._load_args(snapshot / "config.yaml")
        cls.find_checkpoint(snapshot)
        return snapshot

    @classmethod
    def from_run_directory(
        cls,
        run_directory: str | Path,
        device: str | torch.device | None = "auto",
    ) -> MoLAGModel:
        """Reconstruct a model from its configuration and selected checkpoint."""
        run_path = Path(run_directory)
        args = cls._load_args(run_path / "config.yaml")
        checkpoint = cls.find_checkpoint(run_path)

        model = MoLAGModel(args.model_args, args.loss_args)
        resolved_device = resolve_device(device)
        model.load_local(checkpoint, map_location="cpu")
        model.to(resolved_device)
        model.eval()
        return model

    @staticmethod
    def _load_args(path: Path) -> Args:
        if not path.is_file():
            raise FileNotFoundError(f"run configuration not found: {path}")
        with path.open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError(f"run configuration must contain a mapping: {path}")
        return Args.model_validate(values)

    @classmethod
    def find_checkpoint(cls, run_directory: str | Path) -> Path:
        """Return the model checkpoint selected for a run directory."""
        run_path = Path(run_directory)
        for filename in cls.CHECKPOINT_FILENAMES:
            checkpoint = run_path / filename
            if checkpoint.is_file():
                return checkpoint
        expected = ", ".join(cls.CHECKPOINT_FILENAMES)
        raise FileNotFoundError(
            f"no model checkpoint found in {run_path}; expected one of: {expected}"
        )
