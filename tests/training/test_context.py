from unittest.mock import patch

import pytest
import torch

from molag.training import AffinityLossContextBase, FullAffinityLossContext


def context() -> FullAffinityLossContext:
    return FullAffinityLossContext(
        edge_logits=torch.tensor([1.0, -1.0, 0.5], requires_grad=True),
        edge_labels=torch.tensor([1, 0, 0]),
        node_embeddings=torch.randn(3, 4, requires_grad=True),
        tracker_labels=torch.tensor([0, 0, 1]),
        batch_vec=torch.tensor([0, 0, 0]),
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
        max_tracker_nodes=7,
    )


def test_context_base_is_abstract() -> None:
    with pytest.raises(TypeError):
        AffinityLossContextBase(
            edge_logits=torch.empty(0),
            edge_labels=torch.empty(0),
            node_embeddings=torch.empty(0, 1),
            tracker_labels=torch.empty(0, dtype=torch.long),
            batch_vec=torch.empty(0, dtype=torch.long),
            edge_index=torch.empty(2, 0, dtype=torch.long),
        )


def test_full_context_computes_structures_lazily_once() -> None:
    with patch(
        "molag.training.loss.context._full.EdgeCategories.from_graph"
    ) as categorize:
        categorize.return_value.same_real = torch.tensor([True, False, False])
        loss_context = context()

        assert categorize.call_count == 0
        _ = loss_context.edge_categories
        _ = loss_context.edge_categories
        assert categorize.call_count == 1


def test_context_derives_scene_count() -> None:
    assert context().n_scenes == 1


def test_zero_remains_connected_to_autograd() -> None:
    loss_context = context()

    loss_context.zero.backward()

    assert loss_context.edge_logits.grad is not None
    assert loss_context.node_embeddings.grad is not None
