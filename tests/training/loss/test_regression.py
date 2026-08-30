import math

import pytest
import torch

from molag.training import ScaledConjunctionAffinityLoss
from molag.utils.loss import grouped_soft_maximum


def complete_edges(n_nodes: int) -> torch.Tensor:
    return torch.triu_indices(n_nodes, n_nodes, offset=1)


def evaluate_loss(
    loss_fn: ScaledConjunctionAffinityLoss,
    logits: torch.Tensor,
    tracker_labels: torch.Tensor,
    node_embeddings: torch.Tensor | None = None,
    batch_vec: torch.Tensor | None = None,
) -> torch.Tensor:
    if batch_vec is None:
        batch_vec = torch.zeros(len(tracker_labels), dtype=torch.long)
    edge_index = complete_edges(len(tracker_labels))
    same_scene = batch_vec[edge_index[0]] == batch_vec[edge_index[1]]
    edge_index = edge_index[:, same_scene]
    row, col = edge_index
    edge_labels = (
        (tracker_labels[row] == tracker_labels[col]) & (tracker_labels[row] >= 0)
    ).long()
    if node_embeddings is None:
        node_embeddings = torch.zeros(len(tracker_labels), 4)
    return loss_fn(
        edge_logits=logits,
        edge_labels=edge_labels,
        node_embeddings=node_embeddings,
        tracker_labels=tracker_labels,
        batch_vec=batch_vec,
        edge_index=edge_index,
        n_scenes=int(batch_vec.max()) + 1,
    )


def connectivity_only(**kwargs) -> ScaledConjunctionAffinityLoss:
    return ScaledConjunctionAffinityLoss(
        separation_weight=0,
        spurious_bridge_weight=0,
        supcon_weight=0,
        **kwargs,
    )


def separation_only(**kwargs) -> ScaledConjunctionAffinityLoss:
    return ScaledConjunctionAffinityLoss(
        connectivity_weight=0,
        spurious_bridge_weight=0,
        supcon_weight=0,
        **kwargs,
    )


def gradient(
    loss_fn: ScaledConjunctionAffinityLoss,
    logits: torch.Tensor,
    tracker_labels: torch.Tensor,
    batch_vec: torch.Tensor | None = None,
) -> torch.Tensor:
    values = logits.clone().detach().requires_grad_(True)
    evaluate_loss(loss_fn, values, tracker_labels, batch_vec=batch_vec).backward()
    assert values.grad is not None
    return values.grad


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_tracker_nodes": 0}, "max_tracker_nodes"),
        ({"aggregation_beta": 0}, "aggregation_beta"),
        ({"delta_nontree": -1}, "delta_nontree"),
        ({"eps_spur": -0.1}, "eps_spur"),
        ({"conjunct_scaling_power": 1.1}, "scaling_power"),
        ({"connectivity_margin": -1}, "connectivity_margin"),
        ({"separation_margin": -1}, "separation_margin"),
        ({"spurious_margin": -1}, "spurious_margin"),
        ({"supcon_temperature": 0}, "temperature"),
    ],
)
def test_invalid_loss_configuration_is_rejected(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        ScaledConjunctionAffinityLoss(**kwargs)


def test_soft_maximum_gradient_is_a_normalized_softmax() -> None:
    values = torch.tensor([0.5, 2.0, 1.25, 4.0, 0.75], requires_grad=True)
    groups = torch.tensor([0, 0, 0, 1, 1])

    grouped_soft_maximum(values, groups, 2, beta=3).sum().backward()

    expected = torch.cat(
        (
            torch.softmax(3 * values.detach()[:3], dim=0),
            torch.softmax(3 * values.detach()[3:], dim=0),
        )
    )
    torch.testing.assert_close(values.grad, expected)
    torch.testing.assert_close(values.grad[:3].sum(), torch.tensor(1.0))
    torch.testing.assert_close(values.grad[3:].sum(), torch.tensor(1.0))


def test_finite_aggregation_spreads_connectivity_gradient_over_tree() -> None:
    labels = torch.tensor([0, 0, 0, 0])
    logits = torch.tensor([2.0, 1.0, 0.5, 3.0, 0.25, 1.5])

    hard = gradient(connectivity_only(), logits, labels)
    soft = gradient(connectivity_only(aggregation_beta=5), logits, labels)

    assert int((hard.abs() > 0).sum()) == 1
    assert int((soft.abs() > 0).sum()) == 3


def test_finite_nontree_handicap_reaches_every_same_tracker_edge() -> None:
    labels = torch.tensor([0, 0, 0, 0])
    logits = torch.tensor([2.0, 1.0, 0.5, 3.0, 0.25, 1.5])

    result = gradient(
        connectivity_only(aggregation_beta=2, delta_nontree=1),
        logits,
        labels,
    )

    assert int((result.abs() > 0).sum()) == 6


def test_spurious_attachment_weight_is_inert_without_spurious_nodes() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    logits = torch.tensor([4.0, -1.0, -1.0, 3.0, -2.0, 2.0])

    disabled = evaluate_loss(separation_only(aggregation_beta=5), logits, labels)
    enabled = evaluate_loss(
        separation_only(aggregation_beta=5, eps_spur=0.02), logits, labels
    )

    assert torch.equal(disabled, enabled)


def test_spurious_attachment_weight_adds_gradient_only_when_enabled() -> None:
    labels = torch.tensor([0, 0, 1, 1, -1])
    logits = torch.tensor(
        [4.0, -1.0, -1.0, 3.0, -2.0, 2.0, -1.0, 5.0, 1.0, 0.5]
    )
    edges = complete_edges(5)
    spurious = (labels[edges[0]] < 0) | (labels[edges[1]] < 0)

    disabled = gradient(separation_only(aggregation_beta=5), logits, labels)
    enabled = gradient(
        separation_only(aggregation_beta=5, eps_spur=0.01), logits, labels
    )

    assert float(disabled[spurious].abs().sum()) == 0
    assert float(enabled[spurious].abs().sum()) > 0
    assert float(enabled[~spurious].abs().sum()) > 0


def test_scaling_power_converts_conjunct_mean_to_sum() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    logits = torch.randn(15, generator=torch.Generator().manual_seed(3))

    mean = evaluate_loss(separation_only(), logits, labels)
    total = evaluate_loss(
        separation_only(conjunct_scaling_power=1), logits, labels
    )
    half = evaluate_loss(
        separation_only(conjunct_scaling_power=0.5), logits, labels
    )

    torch.testing.assert_close(total, mean * 3)
    torch.testing.assert_close(half, mean * math.sqrt(3))


def test_eligible_scene_mean_excludes_scenes_without_conjuncts() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    edge_count = int(
        (batch[complete_edges(7)[0]] == batch[complete_edges(7)[1]]).sum()
    )
    logits = torch.randn(edge_count, generator=torch.Generator().manual_seed(11))

    diluted = evaluate_loss(separation_only(), logits, labels, batch_vec=batch)
    eligible = evaluate_loss(
        separation_only(eligible_scene_mean=True),
        logits,
        labels,
        batch_vec=batch,
    )

    torch.testing.assert_close(eligible, diluted * 2)


def test_empty_eligible_term_returns_differentiable_zero() -> None:
    labels = torch.tensor([0, 0, 0])
    logits = torch.randn(3, requires_grad=True)

    value = evaluate_loss(
        separation_only(eligible_scene_mean=True), logits, labels
    )
    value.backward()

    assert float(value.detach()) == 0
    assert logits.grad is not None
    assert not logits.grad.isnan().any()


def test_supervised_contrastive_term_is_scene_local() -> None:
    labels = torch.tensor([0, 0, 0, 0])
    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]
    )
    logits = torch.zeros(2)
    loss = ScaledConjunctionAffinityLoss(
        connectivity_weight=0,
        separation_weight=0,
        spurious_bridge_weight=0,
        supcon_weight=1,
    )

    per_scene = evaluate_loss(
        loss,
        logits,
        labels,
        node_embeddings=embeddings,
        batch_vec=torch.tensor([0, 0, 1, 1]),
    )
    one_scene = evaluate_loss(
        loss,
        torch.zeros(6),
        labels,
        node_embeddings=embeddings,
    )

    assert per_scene < one_scene


def test_spurious_bridge_component_penalizes_only_cross_tracker_paths() -> None:
    labels = torch.tensor([0, 1, -1, -1])
    disconnected = torch.full((6,), -5.0)
    disconnected[1] = 5.0
    disconnected[4] = 5.0
    connected = disconnected.clone()
    connected[5] = 5.0
    bridge_only = ScaledConjunctionAffinityLoss(
        connectivity_weight=0,
        separation_weight=0,
        spurious_bridge_weight=1,
        supcon_weight=0,
    )

    disconnected_loss = evaluate_loss(bridge_only, disconnected, labels)
    connected_loss = evaluate_loss(bridge_only, connected, labels)

    assert connected_loss > disconnected_loss


def test_spurious_bridge_gradient_targets_path_bottleneck() -> None:
    labels = torch.tensor([0, 1, -1, -1])
    logits = torch.full((6,), -5.0)
    logits[1] = 5.0
    logits[4] = 4.0
    logits[5] = 2.0
    logits.requires_grad_(True)
    bridge_only = ScaledConjunctionAffinityLoss(
        connectivity_weight=0,
        separation_weight=0,
        spurious_bridge_weight=1,
        supcon_weight=0,
    )

    evaluate_loss(bridge_only, logits, labels).backward()

    assert logits.grad is not None
    assert logits.grad[5] > 0
    assert logits.grad[1] == pytest.approx(0)
    assert logits.grad[4] == pytest.approx(0)


def test_all_spurious_scene_is_finite_and_differentiable() -> None:
    labels = torch.full((4,), -1, dtype=torch.long)
    logits = torch.zeros(6, requires_grad=True)

    value = evaluate_loss(
        ScaledConjunctionAffinityLoss(supcon_weight=0), logits, labels
    )
    value.backward()

    assert torch.isfinite(value)
    assert value.item() == pytest.approx(0)
    assert logits.grad is not None


def test_tracker_larger_than_configured_bound_fails_loudly() -> None:
    with pytest.raises(ValueError, match="exceeding max_tracker_nodes"):
        evaluate_loss(
            connectivity_only(max_tracker_nodes=3),
            torch.zeros(6),
            torch.zeros(4, dtype=torch.long),
        )
