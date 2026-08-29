import torch

from molag.training.loss.components import (
    ConnectivityLossComponent,
    SeparationLossComponent,
    SpuriousAttachmentLossComponent,
    SpuriousBridgeLossComponent,
    SupervisedContrastiveLossComponent,
)
from molag.training.loss.context import FullAffinityLossContext


def loss_context() -> FullAffinityLossContext:
    edge_index = torch.triu_indices(4, 4, offset=1)
    return FullAffinityLossContext(
        edge_logits=torch.tensor(
            [1.0, -1.0, 0.2, -0.8, 0.1, 0.3], requires_grad=True
        ),
        edge_labels=torch.tensor([1, 0, 0, 0, 0, 0]),
        node_embeddings=torch.randn(4, 5, requires_grad=True),
        tracker_labels=torch.tensor([0, 0, 1, -1]),
        batch_vec=torch.zeros(4, dtype=torch.long),
        edge_index=edge_index,
        max_tracker_nodes=7,
    )


def components():
    return (
        ConnectivityLossComponent(1.0, 1.0, 1.0, 3.0, 0.5, True),
        SeparationLossComponent(1.0, 1.0, 1.0, 0.5, True),
        SpuriousAttachmentLossComponent(1.0, 1.0, 1.0, True),
        SpuriousBridgeLossComponent(1.0, 0.0, 0.5, True),
        SupervisedContrastiveLossComponent(1.0, 0.2),
    )


def test_each_component_returns_a_finite_scalar() -> None:
    context = loss_context()

    values = [component(context) for component in components()]

    assert all(value.ndim == 0 for value in values)
    assert all(torch.isfinite(value) for value in values)


def test_components_remain_connected_to_autograd() -> None:
    context = loss_context()

    torch.stack([component(context) for component in components()]).sum().backward()

    assert context.edge_logits.grad is not None
    assert context.node_embeddings.grad is not None
