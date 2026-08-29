"""Composable components of affinity-learning objectives."""

from ._base import AffinityLossComponentBase
from ._connectivity import ConnectivityLossComponent
from ._separation import SeparationLossComponent
from ._spurious_attachment import SpuriousAttachmentLossComponent
from ._spurious_bridge import SpuriousBridgeLossComponent
from ._supervised_contrastive import SupervisedContrastiveLossComponent

__all__ = [
    "AffinityLossComponentBase",
    "ConnectivityLossComponent",
    "SeparationLossComponent",
    "SpuriousAttachmentLossComponent",
    "SpuriousBridgeLossComponent",
    "SupervisedContrastiveLossComponent",
]
