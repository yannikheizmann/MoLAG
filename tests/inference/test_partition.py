import numpy as np
import pytest

from molag.inference import AffinityPartition


def test_partition_returns_connected_groups_and_singletons() -> None:
    partition = AffinityPartition.from_graph(
        n_nodes=4,
        edge_index=np.array([[0, 0, 1, 2], [1, 2, 2, 3]]),
        positive_edges=np.array([True, False, False, False]),
    )

    assert [group.tolist() for group in partition.groups] == [[0, 1], [2], [3]]


def test_partition_does_not_require_ground_truth_labels() -> None:
    partition = AffinityPartition.from_graph(
        n_nodes=2,
        edge_index=np.array([[0], [1]]),
        positive_edges=np.array([True]),
    )

    assert partition.component_ids.tolist() == [0, 0]


def test_invalid_node_index_is_rejected() -> None:
    with pytest.raises(ValueError):
        AffinityPartition.from_graph(
            n_nodes=2,
            edge_index=np.array([[0], [2]]),
            positive_edges=np.array([True]),
        )
