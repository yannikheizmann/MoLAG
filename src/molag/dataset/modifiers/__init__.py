"""Configurable transformations of generated coordinate sets."""

from typing import Annotated

from pydantic import Field

from ._base import ModifierBase
from ._dropout import DropoutModifier
from ._noise import PixelNoiseModifier
from ._spurious import SpuriousBlobsModifier

AnyModifier = Annotated[
    DropoutModifier | SpuriousBlobsModifier | PixelNoiseModifier,
    Field(discriminator="type"),
]

__all__ = [
    "AnyModifier",
    "DropoutModifier",
    "ModifierBase",
    "PixelNoiseModifier",
    "SpuriousBlobsModifier",
]

