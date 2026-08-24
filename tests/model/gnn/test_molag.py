import pytest
import torch
from torch_geometric.data import Data

from molag.config import ModelArgs
from molag.dataset import PyGTrackingAffinityCollator
from molag.model import MoLAGModel
from molag.utils.registry import Registry


def small_model() -> MoLAGModel:
    return MoLAGModel(
        ModelArgs(
            hidden_dims=[4, 8, 4],
            edge_head_dims=[6],
            message_alignment=8,
        )
    )


def sample_batch() -> dict:
    sample = {
        "x": torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "y": torch.tensor([[0, 0], [0, 1], [1, 0]]),
    }
    return PyGTrackingAffinityCollator()([sample])


def test_model_is_registered() -> None:
    assert Registry.get("ModelBase", "MoLAG") is MoLAGModel


def test_architecture_uses_configured_dimensions() -> None:
    model = small_model()

    assert len(model.convs) == 3
    assert model.embedding_dim == 16
    assert model.edge_mlp[0].in_features == 32
    assert model.edge_mlp[0].out_features == 6


def test_forward_returns_one_logit_per_unordered_pair() -> None:
    model = small_model()
    batch = sample_batch()

    output = model(batch["data"])

    assert output["node_embeddings"].shape == (3, 16)
    assert output["edge_logits"].shape == (3,)


def test_forward_supports_gradient_computation() -> None:
    model = small_model()

    output = model(sample_batch()["data"])
    output["edge_logits"].sum().backward()

    assert all(parameter.grad is not None for parameter in model.parameters())


def test_missing_edge_attributes_are_rejected() -> None:
    model = small_model()
    graph = Data(
        x=torch.ones(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )

    with pytest.raises(ValueError, match="edge_attr"):
        model(graph)
