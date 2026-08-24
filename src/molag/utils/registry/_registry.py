from __future__ import annotations

from threading import RLock
from typing import Any


class Registry:
    """Thread-safe mapping from interface names to implementations."""

    _registries: dict[str, dict[str, type[Any]]] = {}
    _lock = RLock()

    @classmethod
    def register(
        cls,
        interface_name: str,
        name: str,
        implementation: type[Any],
    ) -> None:
        with cls._lock:
            implementations = cls._registries.setdefault(interface_name, {})
            if name in implementations:
                raise ValueError(
                    f"{name!r} is already registered under {interface_name!r}."
                )
            implementations[name] = implementation

    @classmethod
    def get(cls, interface_name: str, name: str) -> type[Any]:
        with cls._lock:
            try:
                implementations = cls._registries[interface_name]
            except KeyError as error:
                raise ValueError(
                    f"No implementations are registered for {interface_name!r}."
                ) from error
            try:
                return implementations[name]
            except KeyError as error:
                available = ", ".join(sorted(implementations)) or "none"
                raise ValueError(
                    f"{name!r} is not registered under {interface_name!r}. "
                    f"Available implementations: {available}."
                ) from error

    @classmethod
    def get_all(cls, interface_name: str) -> dict[str, type[Any]]:
        with cls._lock:
            if interface_name not in cls._registries:
                raise ValueError(
                    f"No implementations are registered for {interface_name!r}."
                )
            return dict(cls._registries[interface_name])

    @classmethod
    def unregister(cls, interface_name: str, name: str) -> None:
        with cls._lock:
            implementations = cls._registries.get(interface_name)
            if implementations is not None:
                implementations.pop(name, None)

    @classmethod
    def clear(cls, interface_name: str | None = None) -> None:
        with cls._lock:
            if interface_name is None:
                cls._registries.clear()
            else:
                cls._registries.pop(interface_name, None)

