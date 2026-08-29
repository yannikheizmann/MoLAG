"""Composable affinity-learning objectives."""

from ._affinity import ScaledConjunctionAffinityLoss
from .components import AffinityLossComponentBase
from .context import AffinityLossContextBase, FullAffinityLossContext

__all__ = [
    "AffinityLossComponentBase",
    "AffinityLossContextBase",
    "FullAffinityLossContext",
    "ScaledConjunctionAffinityLoss",
]
