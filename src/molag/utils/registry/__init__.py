"""Automatic registration of interface implementations."""

from ._meta import RegistryMeta
from ._registry import Registry

__all__ = ["Registry", "RegistryMeta"]

