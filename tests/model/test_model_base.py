from pathlib import Path

import pytest
import torch
from torch import nn

from molag.model import ModelBase
from molag.utils.registry import Registry


class ExampleModel(ModelBase):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 1)


def test_model_implementation_is_registered() -> None:
    assert Registry.get("ModelBase", "Example") is ExampleModel


@pytest.mark.parametrize("suffix", [".pt", ".safetensors"])
def test_local_checkpoint_round_trip(tmp_path: Path, suffix: str) -> None:
    source = ExampleModel()
    checkpoint = tmp_path / "nested" / f"model{suffix}"
    source.save_local(checkpoint)

    restored = ExampleModel()
    restored.load_local(checkpoint)

    for original, loaded in zip(source.parameters(), restored.parameters()):
        torch.testing.assert_close(original, loaded)
