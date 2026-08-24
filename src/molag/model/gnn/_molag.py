from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from molag.config import ModelArgs

from ._base import GraphNeuralNetworkBase
from .blocks import EdgeConvBlock, upper_tri_mask


class MoLAGModel(GraphNeuralNetworkBase):
    """Predict same-tracker affinities for every unordered pair of points."""

    def __init__(self, args: ModelArgs | None = None) -> None:
        resolved_args = args or ModelArgs()
        super().__init__(in_dim=resolved_args.in_dim)
        self.args = resolved_args

        blocks: list[EdgeConvBlock] = []
        input_dim = self.in_dim
        for output_dim in self.args.hidden_dims:
            blocks.append(
                EdgeConvBlock(
                    in_dim=input_dim,
                    edge_dim=self.args.edge_feature_dim,
                    out_dim=output_dim,
                    alignment=self.args.message_alignment,
                )
            )
            input_dim = output_dim
        self.convs = nn.ModuleList(blocks)

        self.embedding_dim = sum(self.args.hidden_dims)
        affinity_layers: list[nn.Module] = []
        input_dim = 2 * self.embedding_dim
        for output_dim in self.args.edge_head_dims:
            affinity_layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
            input_dim = output_dim
        affinity_layers.append(nn.Linear(input_dim, 1))
        self.edge_mlp = nn.Sequential(*affinity_layers)

    def forward_gnn(self, data: Data | Batch) -> Tensor:
        """Return the concatenated output of all message-passing blocks."""
        batch = self.ensure_batch(data)
        self._validate_graph(batch)

        block_outputs: list[Tensor] = []
        node_features = batch.x
        for block in self.convs:
            node_features = block(
                node_features,
                batch.edge_index,
                batch.edge_attr,
            )
            block_outputs.append(node_features)
        return torch.cat(block_outputs, dim=-1)

    def forward(self, data: Data | Batch, **kwargs: Any) -> dict[str, Any]:
        """Return node embeddings and one symmetric logit per unordered pair."""
        batch = self.ensure_batch(data)
        node_embeddings = self.forward_gnn(batch)

        source, destination = batch.edge_index[:, upper_tri_mask(batch.edge_index)]
        source_features = node_embeddings[source]
        destination_features = node_embeddings[destination]
        pair_features = torch.cat(
            (
                source_features * destination_features,
                (source_features - destination_features).abs(),
            ),
            dim=-1,
        )
        edge_logits = self.edge_mlp(pair_features).squeeze(-1)

        return {
            "edge_logits": edge_logits,
            "node_embeddings": node_embeddings,
        }

    def _validate_graph(self, batch: Batch) -> None:
        if batch.x is None or batch.x.ndim != 2 or batch.x.shape[1] != self.in_dim:
            raise ValueError(f"data.x must have shape (num_nodes, {self.in_dim})")
        if batch.edge_index is None:
            raise ValueError("data.edge_index is required")
        if batch.edge_attr is None:
            raise ValueError("data.edge_attr is required")
        if (
            batch.edge_attr.ndim != 2
            or batch.edge_attr.shape[0] != batch.edge_index.shape[1]
            or batch.edge_attr.shape[1] != self.args.edge_feature_dim
        ):
            raise ValueError(
                "data.edge_attr must have shape "
                f"(num_edges, {self.args.edge_feature_dim})"
            )
