import pytest
import torch

from molag.model.gnn.blocks import EdgeConvBlock, full_edge_index


def test_edgeconv_output_shape_and_gradient() -> None:
    block = EdgeConvBlock(in_dim=2, edge_dim=3, out_dim=5)
    coordinates = torch.randn(4, 2, requires_grad=True)
    edge_index = full_edge_index(4)
    edge_attributes = torch.randn(edge_index.shape[1], 3)

    output = block(coordinates, edge_index, edge_attributes)
    output.sum().backward()

    assert output.shape == (4, 5)
    assert coordinates.grad is not None


def test_message_width_is_aligned_without_changing_output_width() -> None:
    block = EdgeConvBlock(in_dim=2, edge_dim=3, out_dim=7)

    assert block.mlp[0].in_features == 8
    assert block.mlp[-2].out_features == 7


def test_message_alignment_is_configurable() -> None:
    block = EdgeConvBlock(in_dim=2, edge_dim=3, out_dim=7, alignment=16)

    assert block.mlp[0].in_features == 16


@pytest.mark.parametrize("dimensions", [(0, 3, 4), (2, 0, 4), (2, 3, 0)])
def test_dimensions_must_be_positive(dimensions: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        EdgeConvBlock(*dimensions)


def test_alignment_must_be_positive() -> None:
    with pytest.raises(ValueError):
        EdgeConvBlock(in_dim=2, edge_dim=3, out_dim=4, alignment=0)
