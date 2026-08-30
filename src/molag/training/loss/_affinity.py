from __future__ import annotations

import math

import torch
from torch import Tensor

from .components import (
    AffinityLossComponentBase,
    ConnectivityLossComponent,
    SeparationLossComponent,
    SpuriousAttachmentLossComponent,
    SpuriousBridgeLossComponent,
    SupervisedContrastiveLossComponent,
)
from .context import FullAffinityLossContext


class ScaledConjunctionAffinityLoss:
    """Combine independently configurable affinity-loss components."""

    def __init__(
        self,
        supcon_weight: float = 0.03,
        supcon_temperature: float = 0.2,
        connectivity_weight: float = 1.0,
        connectivity_margin: float = 1.0,
        separation_weight: float = 2.0,
        separation_margin: float = 1.0,
        spurious_bridge_weight: float = 0.25,
        spurious_margin: float = 0.0,
        max_tracker_nodes: int = 7,
        aggregation_beta: float = float("inf"),
        delta_nontree: float = float("inf"),
        eps_spur: float = 0.0,
        conjunct_scaling_power: float = 0.0,
        separation_scaling_power: float | None = None,
        eligible_scene_mean: bool = False,
    ) -> None:
        if max_tracker_nodes < 1:
            raise ValueError("max_tracker_nodes must be positive")
        if connectivity_margin < 0:
            raise ValueError("connectivity_margin must not be negative")
        if separation_margin < 0:
            raise ValueError("separation_margin must not be negative")
        if spurious_margin < 0:
            raise ValueError("spurious_margin must not be negative")
        if math.isnan(aggregation_beta) or aggregation_beta <= 0:
            raise ValueError("aggregation_beta must be positive")
        if math.isnan(delta_nontree) or delta_nontree < 0:
            raise ValueError("delta_nontree must not be negative")
        if eps_spur < 0:
            raise ValueError("eps_spur must not be negative")
        separation_power = (
            conjunct_scaling_power
            if separation_scaling_power is None
            else separation_scaling_power
        )
        self.max_tracker_nodes = max_tracker_nodes
        self.components: tuple[AffinityLossComponentBase, ...] = (
            ConnectivityLossComponent(
                connectivity_weight, connectivity_margin, aggregation_beta,
                delta_nontree, conjunct_scaling_power, eligible_scene_mean,
            ),
            SeparationLossComponent(
                separation_weight, separation_margin, aggregation_beta,
                separation_power, eligible_scene_mean,
            ),
            SpuriousAttachmentLossComponent(
                separation_weight * eps_spur, separation_margin,
                aggregation_beta, eligible_scene_mean,
            ),
            SpuriousBridgeLossComponent(
                spurious_bridge_weight, spurious_margin,
                conjunct_scaling_power, eligible_scene_mean,
            ),
            SupervisedContrastiveLossComponent(
                supcon_weight, supcon_temperature,
            ),
        )

    def __call__(
        self,
        edge_logits: Tensor,
        edge_labels: Tensor,
        node_embeddings: Tensor,
        tracker_labels: Tensor,
        batch_vec: Tensor,
        edge_index: Tensor,
        n_scenes: int | None = None,
    ) -> Tensor:
        context = FullAffinityLossContext(
            edge_logits=edge_logits,
            edge_labels=edge_labels,
            node_embeddings=node_embeddings,
            tracker_labels=tracker_labels,
            batch_vec=batch_vec,
            edge_index=edge_index,
            n_scenes=n_scenes,
            max_tracker_nodes=self.max_tracker_nodes,
        )
        values = [
            component.weight * component(context)
            for component in self.components
            if component.weight > 0
        ]
        return torch.stack(values).sum() if values else context.zero
