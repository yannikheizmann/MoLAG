import math

import torch
import torch.nn.functional as F
from torch import Tensor

from molag.utils.loss import grouped_soft_maximum

from ..context import AffinityLossContextBase
from ._base import AffinityLossComponentBase


class ConnectivityLossComponent(AffinityLossComponentBase):
    """Penalize weak links in each real tracker's spanning tree."""

    def __init__(self, weight: float, margin: float, aggregation_beta: float,
                 delta_nontree: float, scaling_power: float,
                 eligible_scene_mean: bool) -> None:
        super().__init__(weight, scaling_power, eligible_scene_mean)
        self.margin = margin
        self.aggregation_beta = aggregation_beta
        self.delta_nontree = delta_nontree

    def __call__(self, context: AffinityLossContextBase) -> Tensor:
        forest = context.spanning_forest
        if forest.selected_edge_indices.numel() == 0:
            return context.zero
        if math.isinf(self.delta_nontree):
            edge_ids = forest.selected_edge_indices
            group_ids = forest.selected_group_indices
            handicap = None
        else:
            edge_ids = context.edge_categories.same_real.nonzero(as_tuple=True)[0]
            group_ids = forest.node_group[context.edge_index[0, edge_ids]]
            tree_edges = torch.zeros_like(context.edge_logits, dtype=torch.bool)
            tree_edges[forest.selected_edge_indices] = True
            handicap = torch.full(
                (edge_ids.numel(),), self.delta_nontree,
                dtype=torch.float32, device=context.edge_logits.device,
            )
            handicap[tree_edges[edge_ids]] = 0.0
        penalties = F.softplus(self.margin - context.edge_logits[edge_ids].float())
        worst = grouped_soft_maximum(
            penalties, group_ids, forest.group_sizes.numel(),
            self.aggregation_beta, handicap,
        )
        valid = worst.isfinite()
        return self._mean_by_scene(
            worst[valid], forest.group_scene[valid], context.n_scenes
        )
