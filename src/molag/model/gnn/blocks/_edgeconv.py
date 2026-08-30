"""Edge-conditioned message-passing block."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing


class EdgeConvBlock(MessagePassing):
    """Apply an MLP to edge messages and max-aggregate them per node.

    The message width can be padded to a multiple of ``alignment``. This avoids
    inefficient matrix-multiplication kernels for reduced-precision tensors on
    common GPUs. Padding values are zero, so the represented function is unchanged,
    although the first linear layer and its checkpoint shape include the padding.

    A tile multiple is used instead of the next power of two because matrix kernels
    require divisibility, not power-of-two dimensions. Power-of-two rounding could
    add almost as many unused features as the unpadded message already contains.
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        out_dim: int,
        alignment: int = 8,
    ) -> None:
        if min(in_dim, edge_dim, out_dim) < 1:
            raise ValueError("feature dimensions must be positive")
        if alignment < 1:
            raise ValueError("alignment must be positive")
        super().__init__(aggr="max")

        message_dim = 2 * in_dim + edge_dim
        self._padding = -message_dim % alignment
        self.mlp = nn.Sequential(
            nn.Linear(message_dim + self._padding, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """Propagate messages along the supplied directed edges."""
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """Construct and transform messages for individual directed edges."""
        parts = [x_i, x_j, edge_attr]
        if self._padding:
            parts.append(x_i.new_zeros((x_i.shape[0], self._padding)))
        return self.mlp(torch.cat(parts, dim=-1))
