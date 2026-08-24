"""Synthetic coordinate-set generation for MoLAG."""

from ._config import DatasetConfig, PoseConfig
from ._collator import PyGTrackingAffinityCollator
from ._dataset import TrackingDataset
from .modifiers import (
    AnyModifier,
    DropoutModifier,
    ModifierBase,
    PixelNoiseModifier,
    SpuriousBlobsModifier,
)
from .sample import Sample
from .tracker import (
    CameraIntrinsics,
    TrackerBase,
    TrackerCodeBase,
    TrackerGeometryBase,
    TrackerPose,
    TriangularTracker,
    TriangularTrackerCode,
    TriangularTrackerGeometry,
)

__all__ = [
    "PoseConfig",
    "DatasetConfig",
    "PyGTrackingAffinityCollator",
    "TrackingDataset",
    "Sample",
    "CameraIntrinsics",
    "AnyModifier",
    "DropoutModifier",
    "ModifierBase",
    "PixelNoiseModifier",
    "SpuriousBlobsModifier",
    "TrackerBase",
    "TrackerCodeBase",
    "TrackerGeometryBase",
    "TrackerPose",
    "TriangularTracker",
    "TriangularTrackerCode",
    "TriangularTrackerGeometry",
]
