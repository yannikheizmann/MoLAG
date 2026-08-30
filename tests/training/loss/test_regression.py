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


def test_paper_loss_matches_reference_value_and_gradients() -> None:
    torch.manual_seed(3)
    tracker_labels = torch.tensor(
        [0, 0, 0, 1, 1, -1, -1, 0, 0, 1, 1, 1, -1]
    )
    batch_vec = torch.tensor([0] * 7 + [1] * 6)
    edges = []
    for scene in range(2):
        indices = (batch_vec == scene).nonzero().flatten().tolist()
        edges.extend(
            (left, right)
            for position, left in enumerate(indices)
            for right in indices[position + 1 :]
        )
    edge_index = torch.tensor(edges).T
    logits = torch.randn(len(edges), requires_grad=True)
    embeddings = torch.randn(len(tracker_labels), 11, requires_grad=True)
    loss_fn = ScaledConjunctionAffinityLoss(
        supcon_weight=0.03,
        supcon_temperature=0.2,
        connectivity_weight=1.0,
        connectivity_margin=1.0,
        separation_weight=0.46,
        separation_margin=1.0,
        spurious_bridge_weight=0.25,
        spurious_margin=0.0,
        max_tracker_nodes=7,
        aggregation_beta=1.0,
        delta_nontree=3.0,
        eps_spur=0.01,
        conjunct_scaling_power=0.5,
        separation_scaling_power=None,
        eligible_scene_mean=True,
    )

    loss = loss_fn(
        edge_logits=logits,
        edge_labels=torch.zeros_like(logits),
        node_embeddings=embeddings,
        tracker_labels=tracker_labels,
        batch_vec=batch_vec,
        edge_index=edge_index,
        n_scenes=2,
    )
    loss.backward()

    assert float(loss.detach()) == pytest.approx(2.3297815322875977)
    torch.testing.assert_close(
        logits.grad,
        torch.tensor(
            [
                -0.011156992986798286,
                -0.14482709765434265,
                0.012857330963015556,
                0.03018842823803425,
                0.00022309250198304653,
                8.787897240836173e-05,
                -0.04105331003665924,
                0.005692916922271252,
                0.026688961312174797,
                8.450166933471337e-05,
                0.00015009116032160819,
                0.02249673195183277,
                0.07004114240407944,
                0.0001407644886057824,
                0.08016809076070786,
                -0.22694985568523407,
                0.0002837100182659924,
                0.0004210810293443501,
                0.00013058044714853168,
                3.181809370289557e-05,
                0.0,
                -0.2630990743637085,
                0.05229347199201584,
                0.014292276464402676,
                0.010293564759194851,
                0.0006992859416641295,
                0.036588337272405624,
                0.028314167633652687,
                0.01305979024618864,
                0.00026666343910619617,
                -0.10300479829311371,
                -0.08772704005241394,
                0.0004030045820400119,
                -0.012702981941401958,
                7.805336645105854e-05,
                0.08088243752717972,
            ]
        ),
    )
    assert embeddings.grad is not None
    assert float(embeddings.grad.abs().sum()) == pytest.approx(
        0.21547284722328186
    )
    assert float(torch.linalg.vector_norm(embeddings.grad)) == pytest.approx(
        0.027936430647969246
    )
    assert float(embeddings.grad.abs().max()) == pytest.approx(
        0.009786822833120823
    )
    assert torch.count_nonzero(embeddings.grad[[5, 6, 12]]) == 0


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
