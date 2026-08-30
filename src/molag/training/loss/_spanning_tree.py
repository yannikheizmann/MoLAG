"""Batched maximum-spanning-tree selection for small ground-truth trackers."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SpanningForest:
    selected_edge_indices: Tensor
    selected_group_indices: Tensor
    group_scene: Tensor
    group_sizes: Tensor
    node_group: Tensor


def maximum_spanning_forest(
    edge_logits: Tensor,
    edge_index: Tensor,
    tracker_labels: Tensor,
    batch_vec: Tensor,
    same_real_mask: Tensor,
    *,
    max_tracker_nodes: int,
) -> SpanningForest:
    """Select one maximum spanning tree per ``(scene, tracker_id)`` group.

    Prim selection uses detached logits. The returned integer indices select the
    original logits, so gradients flow through chosen tree-edge values but not
    through the discrete choice itself.
    """
    device = edge_logits.device
    n_nodes = tracker_labels.numel()
    real_nodes = (tracker_labels >= 0).nonzero(as_tuple=True)[0]

    if real_nodes.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return SpanningForest(
            selected_edge_indices=empty,
            selected_group_indices=empty,
            group_scene=empty,
            group_sizes=empty,
            node_group=torch.full((n_nodes,), -1, dtype=torch.long, device=device),
        )

    # Encode (tracker_id, scene_id) as a collision-free 1-D key. PyTorch MPS
    # does not implement torch.unique(..., dim=0), while 1-D unique is native.
    # Real labels and scene IDs are nonnegative here.
    scene_stride = batch_vec.max() + 1
    group_keys = tracker_labels[real_nodes] * scene_stride + batch_vec[real_nodes]
    unique_groups, inverse = torch.unique(
        group_keys, sorted=True, return_inverse=True
    )
    n_groups = unique_groups.numel()
    group_scene = unique_groups.remainder(scene_stride)
    group_sizes = torch.bincount(inverse, minlength=n_groups)

    if bool((group_sizes > max_tracker_nodes).any()):
        largest = int(group_sizes.max().item())
        raise ValueError(
            f"Tracker group contains {largest} nodes, exceeding max_tracker_nodes="
            f"{max_tracker_nodes}."
        )

    order = torch.argsort(inverse, stable=True)
    sorted_nodes = real_nodes[order]
    sorted_groups = inverse[order]
    starts = torch.cumsum(group_sizes, dim=0) - group_sizes
    local_positions = torch.arange(
        real_nodes.numel(), device=device
    ) - torch.repeat_interleave(starts, group_sizes)

    node_group = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
    node_local = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
    node_group[sorted_nodes] = sorted_groups
    node_local[sorted_nodes] = local_positions

    scores = edge_logits.new_full(
        (n_groups, max_tracker_nodes, max_tracker_nodes), float("-inf")
    )
    source_edges = torch.full(
        (n_groups, max_tracker_nodes, max_tracker_nodes),
        -1,
        dtype=torch.long,
        device=device,
    )

    all_edge_ids = torch.arange(edge_logits.numel(), device=device)
    same_ids = all_edge_ids[same_real_mask]
    row, col = edge_index[:, same_real_mask]
    group_ids = node_group[row]
    local_row = node_local[row]
    local_col = node_local[col]
    detached = edge_logits[same_real_mask].detach()

    scores[group_ids, local_row, local_col] = detached
    scores[group_ids, local_col, local_row] = detached
    source_edges[group_ids, local_row, local_col] = same_ids
    source_edges[group_ids, local_col, local_row] = same_ids

    valid_nodes = (
        torch.arange(max_tracker_nodes, device=device).unsqueeze(0)
        < group_sizes.unsqueeze(1)
    )
    visited = torch.zeros_like(valid_nodes)
    visited[:, 0] = group_sizes > 0
    flat_scores = scores.view(n_groups, -1)
    flat_sources = source_edges.view(n_groups, -1)
    group_range = torch.arange(n_groups, device=device)

    selected_edges: list[Tensor] = []
    selected_groups: list[Tensor] = []
    for _ in range(max_tracker_nodes - 1):
        active = visited.sum(1) < group_sizes
        candidates = (
            visited.unsqueeze(2)
            & (~visited).unsqueeze(1)
            & valid_nodes.unsqueeze(1)
            & (source_edges >= 0)
        )
        choices = flat_scores.masked_fill(
            ~candidates.view(n_groups, -1), float("-inf")
        ).argmax(1)
        chosen_edges = flat_sources[group_range, choices]
        selected_edges.append(torch.where(active, chosen_edges, -1))
        selected_groups.append(torch.where(active, group_range, -1))

        destination = choices.remainder(max_tracker_nodes)
        additions = torch.zeros_like(visited)
        additions.scatter_(1, destination.unsqueeze(1), active.unsqueeze(1))
        visited = visited | additions

    selected_edge_indices = torch.stack(selected_edges, dim=1).flatten()
    selected_group_indices = torch.stack(selected_groups, dim=1).flatten()
    valid_selection = selected_edge_indices >= 0
    selected_edge_indices = selected_edge_indices[valid_selection]
    selected_group_indices = selected_group_indices[valid_selection]

    expected = (group_sizes - 1).clamp(min=0).sum()
    if int(selected_edge_indices.numel()) != int(expected.item()):
        raise RuntimeError(
            "Could not construct a spanning tree for every real tracker group; "
            "the affinity edge graph must contain every unordered within-group pair."
        )

    return SpanningForest(
        selected_edge_indices=selected_edge_indices,
        selected_group_indices=selected_group_indices,
        group_scene=group_scene,
        group_sizes=group_sizes,
        node_group=node_group,
    )

