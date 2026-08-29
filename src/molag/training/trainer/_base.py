from abc import ABC, abstractmethod
from typing import Any

from molag.utils.registry import RegistryMeta


class TrainerBase(ABC, metaclass=RegistryMeta["TrainerBase"]):
    """Interface implemented by MoLAG training backends."""

    @abstractmethod
    def train(self) -> dict[str, float]: ...

    @abstractmethod
    def evaluate(self) -> dict[str, float]: ...

    @abstractmethod
    def predict(self, dataset: Any) -> Any: ...

    @property
    @abstractmethod
    def output_dir(self) -> str: ...
