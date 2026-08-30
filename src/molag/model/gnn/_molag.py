"""Graph neural network for tracker-affinity prediction."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from molag.config import LossArgs, ModelArgs
from molag.training import ScaledConjunctionAffinityLoss

from ._base import GraphNeuralNetworkBase
from .blocks import EdgeConvBlock, upper_tri_mask


class MoLAGModel(GraphNeuralNetworkBase):
    """Predict same-tracker affinities for every unordered pair of points."""

    def __init__(
        self,
        model_args: ModelArgs | None = None,
        loss_args: LossArgs | None = None,
    ) -> None:
        resolved_model_args = model_args or ModelArgs()
        super().__init__(in_dim=resolved_model_args.in_dim)
        self.model_args = resolved_model_args
        self.loss_args = loss_args or LossArgs()

        blocks: list[EdgeConvBlock] = []
        input_dim = self.in_dim
        for output_dim in self.model_args.hidden_dims:
            blocks.append(
                EdgeConvBlock(
                    in_dim=input_dim,
                    edge_dim=self.model_args.edge_feature_dim,
                    out_dim=output_dim,
                    alignment=self.model_args.message_alignment,
                )
            )
            input_dim = output_dim
        self.convs = nn.ModuleList(blocks)

        self.embedding_dim = sum(self.model_args.hidden_dims)
        affinity_layers: list[nn.Module] = []
        input_dim = 2 * self.embedding_dim
        for output_dim in self.model_args.edge_head_dims:
            affinity_layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
            input_dim = output_dim
        affinity_layers.append(nn.Linear(input_dim, 1))
        self.edge_mlp = nn.Sequential(*affinity_layers)
        self.loss_fn = ScaledConjunctionAffinityLoss(
            **self.loss_args.model_dump(),
        )

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

    def forward(
        self,
        data: Data | Batch,
        edge_labels: Tensor | None = None,
        tracker_labels: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return node embeddings and one symmetric logit per unordered pair."""
        batch = self.ensure_batch(data)
        node_embeddings = self.forward_gnn(batch)

        pair_mask = upper_tri_mask(batch.edge_index)
        pair_edge_index = batch.edge_index[:, pair_mask]
        source, destination = pair_edge_index
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

        targets = edge_labels if edge_labels is not None else labels
        loss = None
        if targets is not None:
            if tracker_labels is None:
                raise ValueError(
                    "tracker_labels are required when affinity labels are provided"
                )
            loss = self.loss_fn(
                edge_logits=edge_logits,
                edge_labels=targets,
                node_embeddings=node_embeddings,
                tracker_labels=tracker_labels,
                batch_vec=batch.batch,
                edge_index=pair_edge_index,
                n_scenes=batch.num_graphs,
            )

        return {
            "loss": loss,
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
            or batch.edge_attr.shape[1] != self.model_args.edge_feature_dim
        ):
            raise ValueError(
                "data.edge_attr must have shape "
                f"(num_edges, {self.model_args.edge_feature_dim})"
            )
