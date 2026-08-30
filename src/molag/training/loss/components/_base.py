"""Abstract interface and scene reduction for affinity-loss components."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ..context import AffinityLossContextBase


class AffinityLossComponentBase(ABC):
    """Weighted affinity condition with configurable scene reduction."""

    def __init__(
        self,
        weight: float,
        scaling_power: float = 0.0,
        eligible_scene_mean: bool = False,
    ) -> None:
        if weight < 0:
            raise ValueError("component weight must not be negative")
        if not 0 <= scaling_power <= 1:
            raise ValueError("scaling_power must lie in [0, 1]")
        self.weight = weight
        self.scaling_power = scaling_power
        self.eligible_scene_mean = eligible_scene_mean

    def _reduce_scenes(self, totals: Tensor, counts: Tensor) -> Tensor:
        """Reduce condition totals as ``mean * count**scaling_power`` per scene."""
        safe_counts = counts.clamp(min=1.0)
        per_scene = (totals / safe_counts) * safe_counts.pow(self.scaling_power)
        if not self.eligible_scene_mean:
            return per_scene.mean()
        eligible = counts > 0
        return (
            per_scene[eligible].mean()
            if bool(eligible.any())
            else per_scene.sum() * 0
        )

    def _mean_by_scene(
        self, values: Tensor, scene_ids: Tensor, n_scenes: int
    ) -> Tensor:
        """Aggregate condition values by scene and apply the scene reduction."""
        totals = torch.zeros(n_scenes, dtype=values.dtype, device=values.device)
        counts = torch.zeros_like(totals)
        totals.scatter_add_(0, scene_ids, values)
        counts.scatter_add_(0, scene_ids, torch.ones_like(values))
        return self._reduce_scenes(totals, counts)

    @abstractmethod
    def __call__(self, context: AffinityLossContextBase) -> Tensor: ...
