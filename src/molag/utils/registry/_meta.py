"""Register concrete subclasses automatically from their class names."""

from __future__ import annotations

from abc import ABCMeta
from typing import Any

from ._registry import Registry


class RegistryMeta(ABCMeta):
    """Register implementations using the project's interface naming convention.

    ``ExampleBase`` implementations must end in ``Example``. For instance,
    ``FastExample`` is registered under the key ``Fast``.
    """

    _interface_name = ""
    _implementation_suffix = ""

    def __class_getitem__(cls, interface_name: str) -> type[RegistryMeta]:
        if not interface_name.endswith("Base"):
            raise ValueError("registered interface names must end in 'Base'")

        class ConfiguredRegistryMeta(cls):
            """Registry metaclass bound to one interface name."""

            _interface_name = interface_name
            _implementation_suffix = interface_name.removesuffix("Base")

        ConfiguredRegistryMeta.__name__ = f"RegistryMeta[{interface_name}]"
        return ConfiguredRegistryMeta

    def __init__(
        cls,
        name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
    ) -> None:
        super().__init__(name, bases, namespace)
        interface_name = cls._interface_name
        suffix = cls._implementation_suffix
        if not interface_name or name == interface_name or name.endswith("Base"):
            return
        if not name.endswith(suffix):
            raise ValueError(
                f"{name!r} must end in {suffix!r} because it implements "
                f"{interface_name!r}."
            )
        Registry.register(interface_name, name.removesuffix(suffix), cls)
