import pytest
import torch

from molag.dataset import PyGTrackingAffinityCollator


def test_collator_builds_batched_graph_and_labels() -> None:
    samples = [
        {
            "x": torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            "y": torch.tensor([[0, 0], [0, 1], [1, 0]]),
        },
        {
            "x": torch.tensor([[0.0, 0.0], [0.0, 2.0]]),
            "y": torch.tensor([[2, 0], [-1, -1]]),
        },
    ]

    output = PyGTrackingAffinityCollator()(samples)

    assert output["data"].num_graphs == 2
    assert output["data"].x.shape == (5, 2)
    assert output["data"].edge_index.shape == (2, 8)
    assert output["data"].edge_attr.shape == (8, 3)
    assert output["tracker_labels"].tolist() == [0, 0, 1, 2, -1]
    assert output["led_labels"].tolist() == [0, 1, 0, 0, -1]
    assert output["edge_labels"].tolist() == [1, 0, 0, 0]
    assert output["labels"] is output["edge_labels"]


def test_edge_features_contain_distance_and_direction() -> None:
    sample = {
        "x": torch.tensor([[0.0, 0.0], [3.0, 4.0]]),
        "y": torch.tensor([[0, 0], [0, 1]]),
    }

    edge_attributes = PyGTrackingAffinityCollator()([sample])["data"].edge_attr

    torch.testing.assert_close(
        edge_attributes,
        torch.tensor([[25.0, 0.6, 0.8], [25.0, -0.6, -0.8]]),
    )


def test_spurious_points_never_form_positive_edges() -> None:
    sample = {
        "x": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        "y": torch.tensor([[-1, -1], [-1, -1]]),
    }

    output = PyGTrackingAffinityCollator()([sample])

    assert output["edge_labels"].tolist() == [0]


@pytest.mark.parametrize(
    "samples, error",
    [
        ([], ValueError),
        ([{"x": [], "y": []}], TypeError),
        (
            [{"x": torch.ones(2, 3), "y": torch.ones(2, 2, dtype=torch.long)}],
            ValueError,
        ),
        (
            [{"x": torch.ones(2, 2), "y": torch.ones(3, 2, dtype=torch.long)}],
            ValueError,
        ),
    ],
)
def test_invalid_batches_raise(samples: list[dict], error: type[Exception]) -> None:
    with pytest.raises(error):
        PyGTrackingAffinityCollator()(samples)
