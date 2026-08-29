import numpy as np

from molag.evaluation import PartitionAssessment


def complete_edges(n_nodes: int) -> np.ndarray:
    return np.stack(np.triu_indices(n_nodes, k=1))


def test_correct_partition_accepts_spurious_attachment() -> None:
    labels = np.array([0, 0, 1, 1, -1])
    edges = complete_edges(5)
    active_pairs = {(0, 1), (2, 3), (0, 4)}
    positive = np.array([tuple(pair) in active_pairs for pair in edges.T])

    result = PartitionAssessment.from_graph(labels, edges, positive)

    assert result.correct
    assert not result.has_real_merge
    assert not result.has_real_split


def test_real_tracker_split_is_reported() -> None:
    result = PartitionAssessment.from_graph(
        np.array([0, 0]), complete_edges(2), np.array([False])
    )

    assert not result.correct
    assert result.has_real_split


def test_spurious_bridge_is_distinguished_from_direct_merge() -> None:
    result = PartitionAssessment.from_graph(
        np.array([0, 1, -1]),
        complete_edges(3),
        np.array([False, True, True]),
    )

    assert result.has_real_merge
    assert result.spurious_bridge
    assert result.real_only_correct
