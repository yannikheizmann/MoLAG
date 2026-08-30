"""Spurious-path bridge-risk component."""

import torch
import torch.nn.functional as F

from ..context import AffinityLossContextBase
from ._base import AffinityLossComponentBase


class SpuriousBridgeLossComponent(AffinityLossComponentBase):
    """Penalty for paths joining real trackers through spurious points."""

    def __init__(self, weight: float, margin: float, scaling_power: float,
                 eligible_scene_mean: bool) -> None:
        super().__init__(weight, scaling_power, eligible_scene_mean)
        self.margin = margin

    def __call__(self, context: AffinityLossContextBase):
        """Evaluate widest-path bridge risk for every real tracker pair.

        Tracker groups act as terminals and spurious points as intermediate nodes.
        A Floyd--Warshall max--min closure yields the path whose weakest affinity is
        strongest. Pairs without a spurious path contribute zero while remaining in
        the scene denominator.
        """
        forest = context.spanning_forest
        n_groups = forest.group_scene.numel()
        spurious_nodes = (context.tracker_labels < 0).nonzero(as_tuple=True)[0]
        real_spurious_edges = context.edge_categories.spurious_real.nonzero(
            as_tuple=True
        )[0]
        if (
            real_spurious_edges.numel() == 0
            or spurious_nodes.numel() == 0
            or n_groups < 2
        ):
            return context.zero

        group_counts = torch.bincount(
            forest.group_scene, minlength=context.n_scenes
        )
        group_starts = torch.cumsum(group_counts, 0) - group_counts
        group_order = torch.argsort(forest.group_scene, stable=True)
        group_local = torch.empty(
            n_groups, dtype=torch.long, device=context.edge_logits.device
        )
        group_local[group_order] = torch.arange(
            n_groups, device=context.edge_logits.device
        ) - torch.repeat_interleave(group_starts, group_counts)

        spurious_scenes = context.batch_vec[spurious_nodes]
        spurious_order = torch.argsort(spurious_scenes, stable=True)
        sorted_spurious = spurious_nodes[spurious_order]
        spurious_scenes = context.batch_vec[sorted_spurious]
        spurious_counts = torch.bincount(
            spurious_scenes, minlength=context.n_scenes
        )
        spurious_starts = torch.cumsum(spurious_counts, 0) - spurious_counts
        spurious_local = torch.full_like(context.tracker_labels, -1)
        spurious_local[sorted_spurious] = torch.arange(
            spurious_nodes.numel(), device=context.edge_logits.device
        ) - torch.repeat_interleave(spurious_starts, spurious_counts)

        max_groups = int(group_counts.max().item())
        max_spurious = int(spurious_counts.max().item())
        width = max_groups + max_spurious
        adjacency = torch.full(
            (context.n_scenes, width, width), float("-inf"),
            dtype=torch.float32, device=context.edge_logits.device,
        )
        row, col = context.edge_index[:, real_spurious_edges]
        row_spurious = context.tracker_labels[row] < 0
        spurious_node = torch.where(row_spurious, row, col)
        real_node = torch.where(row_spurious, col, row)
        groups = forest.node_group[real_node]
        scenes = context.batch_vec[spurious_node]
        group_position = group_local[groups]
        spurious_position = max_groups + spurious_local[spurious_node]
        stride = width * width
        indices = [
            scenes * stride + group_position * width + spurious_position,
            scenes * stride + spurious_position * width + group_position,
        ]
        values = [
            context.edge_logits[real_spurious_edges].float(),
            context.edge_logits[real_spurious_edges].float(),
        ]

        spurious_edges = context.edge_categories.spurious_spurious.nonzero(
            as_tuple=True
        )[0]
        if spurious_edges.numel():
            source, destination = context.edge_index[:, spurious_edges]
            scenes = context.batch_vec[source]
            source_position = max_groups + spurious_local[source]
            destination_position = max_groups + spurious_local[destination]
            indices.extend([
                scenes * stride + source_position * width + destination_position,
                scenes * stride + destination_position * width + source_position,
            ])
            logits = context.edge_logits[spurious_edges].float()
            values.extend([logits, logits])

        adjacency.flatten().scatter_reduce_(
            0, torch.cat(indices), torch.cat(values), reduce="amax", include_self=True
        )
        widest = adjacency
        for position in range(max_groups, width):
            via = torch.minimum(
                widest[:, :, position].unsqueeze(2),
                widest[:, position, :].unsqueeze(1),
            )
            widest = torch.maximum(widest, via)

        paths = widest[:, :max_groups, :max_groups]
        valid_groups = torch.arange(
            max_groups, device=context.edge_logits.device
        ).unsqueeze(0) < group_counts.unsqueeze(1)
        upper_triangle = torch.triu(
            torch.ones(
                max_groups, max_groups, dtype=torch.bool,
                device=context.edge_logits.device,
            ), diagonal=1,
        ).unsqueeze(0)
        pair_mask = (
            valid_groups.unsqueeze(2) & valid_groups.unsqueeze(1) & upper_triangle
        )
        bridged = pair_mask & torch.isfinite(paths)
        risks = paths.where(bridged, torch.zeros_like(paths))
        losses = F.softplus(self.margin + risks) * bridged.float()
        return self._reduce_scenes(
            losses.flatten(1).sum(1), pair_mask.flatten(1).sum(1).float()
        )
