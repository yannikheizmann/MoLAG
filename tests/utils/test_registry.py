from abc import ABC

import pytest

from molag.utils.registry import Registry, RegistryMeta


def test_subclass_is_registered_by_naming_convention() -> None:
    class ExampleBase(ABC, metaclass=RegistryMeta["ExampleBase"]):
        pass

    class FastExample(ExampleBase):
        pass

    try:
        assert Registry.get("ExampleBase", "Fast") is FastExample
    finally:
        Registry.clear("ExampleBase")


def test_implementation_name_must_match_interface() -> None:
    class ExampleBase(ABC, metaclass=RegistryMeta["ExampleBase"]):
        pass

    with pytest.raises(ValueError, match="must end"):

        class InvalidName(ExampleBase):
            pass

    Registry.clear("ExampleBase")


def test_registry_errors_list_available_implementations() -> None:
    class ExampleBase(ABC, metaclass=RegistryMeta["ExampleBase"]):
        pass

    class FastExample(ExampleBase):
        pass

    try:
        with pytest.raises(ValueError, match="Fast"):
            Registry.get("ExampleBase", "Missing")
    finally:
        Registry.clear("ExampleBase")

