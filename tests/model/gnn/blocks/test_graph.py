import pytest
import torch

from molag.model.gnn.blocks import full_edge_index, upper_tri_mask


def test_full_edge_index_contains_each_directed_pair() -> None:
    edge_index = full_edge_index(3)

    assert edge_index.shape == (2, 6)
    assert set(map(tuple, edge_index.t().tolist())) == {
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    }


def test_full_edge_index_handles_small_graphs() -> None:
    assert full_edge_index(0).shape == (2, 0)
    assert full_edge_index(1).shape == (2, 0)
    with pytest.raises(ValueError):
        full_edge_index(-1)


def test_upper_tri_mask_selects_each_pair_once() -> None:
    edge_index = full_edge_index(4)
    selected = edge_index[:, upper_tri_mask(edge_index)]

    assert selected.shape == (2, 6)
    assert torch.all(selected[0] < selected[1])
