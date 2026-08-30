"""Shared utilities used throughout MoLAG."""

from ._geometry import GeometryUtils

__all__ = ["GeometryUtils"]
from ._device import preferred_device, resolve_device

__all__ = ["preferred_device", "resolve_device"]
