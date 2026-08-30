from abc import ABC, abstractmethod
from typing import Any

from molag.utils.registry import RegistryMeta


class MetricsBase(ABC, metaclass=RegistryMeta["MetricsBase"]):
    """Interface for streaming evaluation accumulators."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(self, **values: Any) -> None: ...

    @abstractmethod
    def compute(self) -> dict[str, float]: ...

    def breakdown(self) -> dict[str, Any]:
        """Return optional structured diagnostics accumulated with the metrics."""
        return {}
