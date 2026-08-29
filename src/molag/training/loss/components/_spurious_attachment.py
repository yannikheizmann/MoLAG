import torch
import torch.nn.functional as F

from molag.utils.loss import grouped_soft_maximum

from ..context import AffinityLossContextBase
from ._base import AffinityLossComponentBase


class SpuriousAttachmentLossComponent(AffinityLossComponentBase):
    """Penalize attachment of spurious points to real trackers."""

    def __init__(self, weight: float, margin: float, aggregation_beta: float,
                 eligible_scene_mean: bool) -> None:
        super().__init__(weight, 0.0, eligible_scene_mean)
        self.margin = margin
        self.aggregation_beta = aggregation_beta

    def __call__(self, context: AffinityLossContextBase):
        edge_ids = context.edge_categories.spurious_real.nonzero(as_tuple=True)[0]
        if edge_ids.numel() == 0:
            return context.zero
        forest = context.spanning_forest
        row, col = context.edge_index[:, edge_ids]
        row_spurious = context.tracker_labels[row] < 0
        spurious_node = torch.where(row_spurious, row, col)
        real_node = torch.where(row_spurious, col, row)
        groups = forest.node_group[real_node]
        keys = groups * context.tracker_labels.numel() + spurious_node
        unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        worst = grouped_soft_maximum(
            context.edge_logits[edge_ids].float(), inverse,
            unique_keys.numel(), self.aggregation_beta,
        )
        penalties = F.softplus(self.margin + worst)
        scenes = forest.group_scene[
            unique_keys.div(context.tracker_labels.numel(), rounding_mode="floor")
        ]
        return self._mean_by_scene(penalties, scenes, context.n_scenes)
