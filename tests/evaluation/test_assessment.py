import numpy as np

from molag.evaluation import GroupingFailureMode, PartitionAssessment


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
    assert result.failure_mode == GroupingFailureMode.CORRECT
    assert result.n_trackers == 2
    assert result.n_trackers_correct == 2
    assert result.n_spurious == 1


def test_real_tracker_split_is_reported() -> None:
    result = PartitionAssessment.from_graph(
        np.array([0, 0]), complete_edges(2), np.array([False])
    )

    assert not result.correct
    assert result.has_real_split
    assert result.trackers[0].has_split
    assert result.trackers[0].component_id is None
    assert result.trackers[0].failure_mode == GroupingFailureMode.FALSE_SPLIT


def test_spurious_bridge_is_distinguished_from_direct_merge() -> None:
    result = PartitionAssessment.from_graph(
        np.array([0, 1, -1]),
        complete_edges(3),
        np.array([False, True, True]),
    )

    assert result.has_real_merge
    assert result.spurious_bridge
    assert result.real_only_correct
    assert result.failure_mode == GroupingFailureMode.SPURIOUS_BRIDGE


def test_tracker_assessment_distinguishes_merge_and_split() -> None:
    result = PartitionAssessment.from_graph(
        np.array([0, 0, 1, 1]),
        complete_edges(4),
        np.array([False, True, False, False, False, False]),
    )

    assert result.failure_mode == GroupingFailureMode.MIXED
    assert result.trackers[0].has_merge
    assert result.trackers[0].has_split
    assert result.trackers[0].failure_mode == GroupingFailureMode.MIXED


def test_complete_tracker_counts_are_merge_tolerant_for_extractability() -> None:
    result = PartitionAssessment.from_graph(
        np.array([0, 0, 1, 1, 2]),
        complete_edges(5),
        np.array([True, True, False, False, False, True, False, False, False, False]),
    )

    assert result.complete_tracker_counts(num_leds=2) == (2, 0, 1)
