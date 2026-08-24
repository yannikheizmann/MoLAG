from functools import lru_cache

import torch
from torch import Tensor


@lru_cache(maxsize=128)
def full_edge_index(num_nodes: int) -> Tensor:
    """Return both directed edges for every distinct pair of nodes."""
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
    """Select one directed edge for each unordered node pair."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")
    source, destination = edge_index
    return source < destination
