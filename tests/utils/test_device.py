from unittest.mock import patch

import torch

from molag.utils import preferred_device, resolve_device


def test_preferred_device_prioritizes_cuda() -> None:
    with patch("torch.cuda.is_available", return_value=True):
        assert preferred_device() == torch.device("cuda")


def test_preferred_device_uses_mps_when_cuda_is_unavailable() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert preferred_device() == torch.device("mps")


def test_preferred_device_falls_back_to_cpu() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        assert preferred_device() == torch.device("cpu")


def test_resolve_device_preserves_explicit_selection() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cuda:1")) == torch.device("cuda:1")


def test_resolve_device_uses_preference_for_auto() -> None:
    with patch("molag.utils._device.preferred_device") as preferred:
        preferred.return_value = torch.device("cpu")

        assert resolve_device("auto") == torch.device("cpu")
        assert resolve_device(None) == torch.device("cpu")
        assert preferred.call_count == 2
