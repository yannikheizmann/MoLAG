"""Resolve explicit and automatically selected PyTorch devices."""

from __future__ import annotations

import torch


def preferred_device() -> torch.device:
    """Return CUDA, MPS, or CPU in descending order of preference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(
    device: str | torch.device | None = "auto",
) -> torch.device:
    """Resolve an explicit device or select the preferred local backend."""
    if device is None or str(device).strip().lower() == "auto":
        return preferred_device()
    return torch.device(device)
