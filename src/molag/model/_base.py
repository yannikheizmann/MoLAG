"""Registered model interface and local checkpoint persistence."""

from __future__ import annotations

from abc import ABC
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from molag.utils.registry import RegistryMeta


class ModelBase(nn.Module, ABC, metaclass=RegistryMeta["ModelBase"]):
    """Registered neural-network model with local checkpoint persistence."""

    def save_local(self, path: str | Path) -> None:
        """Save model parameters as a PyTorch or Safetensors checkpoint."""
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint.suffix == ".safetensors":
            save_file(self.state_dict(), checkpoint)
        else:
            torch.save(self.state_dict(), checkpoint)

    def load_local(
        self,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> None:
        """Load model parameters from a PyTorch or Safetensors checkpoint."""
        checkpoint = Path(path)
        if checkpoint.suffix == ".safetensors":
            state = load_file(checkpoint, device=str(map_location))
        else:
            state = torch.load(
                checkpoint,
                map_location=map_location,
                weights_only=True,
            )
        self.load_state_dict(state)
