import torch
import torch.nn.functional as F
from torch import Tensor

from ..context import AffinityLossContextBase
from ._base import AffinityLossComponentBase


class SupervisedContrastiveLossComponent(AffinityLossComponentBase):
    """Encourage scene-local embeddings of the same tracker to agree."""

    def __init__(self, weight: float, temperature: float) -> None:
        super().__init__(weight)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def __call__(self, context: AffinityLossContextBase) -> Tensor:
        real_indices = (context.tracker_labels >= 0).nonzero(as_tuple=True)[0]
        if real_indices.numel() < 2:
            return context.zero

        scenes = context.batch_vec[real_indices]
        order = torch.argsort(scenes, stable=True)
        scenes = scenes[order]
        labels = context.tracker_labels[real_indices][order]
        embeddings = F.normalize(
            context.node_embeddings[real_indices][order].float(), dim=1
        )
        scene_sizes = torch.bincount(scenes, minlength=context.n_scenes)
        max_nodes = int(scene_sizes.max().item())
        starts = torch.cumsum(scene_sizes, dim=0) - scene_sizes
        positions = torch.arange(real_indices.numel(), device=embeddings.device)
        positions -= torch.repeat_interleave(starts, scene_sizes)

        padded_embeddings = embeddings.new_zeros(
            (context.n_scenes, max_nodes, embeddings.shape[1])
        )
        padded_labels = torch.full(
            (context.n_scenes, max_nodes), -1, dtype=labels.dtype, device=labels.device
        )
        padded_embeddings[scenes, positions] = embeddings
        padded_labels[scenes, positions] = labels
        valid_nodes = torch.arange(max_nodes, device=labels.device).unsqueeze(0)
        valid_nodes = valid_nodes < scene_sizes.unsqueeze(1)
        similarities = torch.bmm(
            padded_embeddings, padded_embeddings.transpose(1, 2)
        ) / self.temperature
        diagonal = torch.eye(
            max_nodes, dtype=torch.bool, device=labels.device
        ).unsqueeze(0)
        valid_pairs = valid_nodes.unsqueeze(2) & valid_nodes.unsqueeze(1) & ~diagonal
        positive_pairs = valid_pairs & (
            padded_labels.unsqueeze(2) == padded_labels.unsqueeze(1)
        )
        log_denominator = similarities.masked_fill(
            ~valid_pairs, float("-inf")
        ).logsumexp(dim=2)
        positive_counts = positive_pairs.sum(dim=2)
        valid_anchors = valid_nodes & (positive_counts > 0)
        mean_positive = (
            (positive_pairs.to(similarities.dtype) * similarities).sum(dim=2)
            / positive_counts.clamp(min=1).to(similarities.dtype)
        )
        anchor_loss = torch.where(
            valid_anchors,
            (log_denominator - mean_positive).float(),
            torch.zeros_like(log_denominator).float(),
        )
        scene_loss = anchor_loss.sum(dim=1) / valid_anchors.sum(dim=1).clamp(min=1)
        valid_scenes = valid_anchors.any(dim=1)
        return scene_loss[valid_scenes].sum() / valid_scenes.sum().clamp(min=1)
