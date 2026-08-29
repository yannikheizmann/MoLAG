import torch
import torch.nn.functional as F

from molag.utils.loss import grouped_soft_maximum

from ..context import AffinityLossContextBase
from ._base import AffinityLossComponentBase


class SeparationLossComponent(AffinityLossComponentBase):
    """Penalize affinities between different real trackers."""

    def __init__(self, weight: float, margin: float, aggregation_beta: float,
                 scaling_power: float, eligible_scene_mean: bool) -> None:
        super().__init__(weight, scaling_power, eligible_scene_mean)
        self.margin = margin
        self.aggregation_beta = aggregation_beta

    def __call__(self, context: AffinityLossContextBase):
        forest = context.spanning_forest
        n_groups = forest.group_scene.numel()
        edge_ids = context.edge_categories.different_real.nonzero(as_tuple=True)[0]
        if n_groups == 0 or edge_ids.numel() == 0:
            return context.zero
        row, col = context.edge_index[:, edge_ids]
        first = forest.node_group[row]
        second = forest.node_group[col]
        keys = torch.minimum(first, second) * n_groups + torch.maximum(first, second)
        unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        worst = grouped_soft_maximum(
            context.edge_logits[edge_ids].float(), inverse,
            unique_keys.numel(), self.aggregation_beta,
        )
        penalties = F.softplus(self.margin + worst)
        scenes = forest.group_scene[unique_keys.div(n_groups, rounding_mode="floor")]
        return self._mean_by_scene(penalties, scenes, context.n_scenes)
