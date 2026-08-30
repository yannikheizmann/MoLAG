"""Complete-graph construction and unordered-edge selection."""

from functools import lru_cache

import torch
from torch import Tensor


@lru_cache(maxsize=128)
def full_edge_index(num_nodes: int) -> Tensor:
    """Construct both directed edges for every distinct node pair.

    The cached tensor remains on the CPU; callers may rely on PyTorch Geometric
    batching or an explicit transfer to move it to another device.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must not be negative")
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    nodes = torch.arange(num_nodes, dtype=torch.long)
    sources = nodes.repeat_interleave(num_nodes - 1)
    destinations = torch.cat(
        [torch.cat((nodes[:index], nodes[index + 1 :])) for index in range(num_nodes)]
    )
    return torch.stack((sources, destinations))


def upper_tri_mask(edge_index: Tensor) -> Tensor:
    """Select the ``source < destination`` representative of each node pair."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")
    source, destination = edge_index
    return source < destination
