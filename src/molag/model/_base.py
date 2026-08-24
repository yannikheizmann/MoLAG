from __future__ import annotations

from abc import ABC
from pathlib import Path

import torch
from torch import nn

from molag.utils.registry import RegistryMeta


class ModelBase(nn.Module, ABC, metaclass=RegistryMeta["ModelBase"]):
    """Common interface for registered MoLAG model implementations."""

    def save_local(self, path: str | Path) -> None:
        """Save the model parameters to a local checkpoint."""
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), checkpoint)

    def load_local(
        self,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> None:
        """Load model parameters from a local checkpoint."""
        state = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state)
