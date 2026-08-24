from typing import Any

import pytest
from torch_geometric.data import Batch, Data

from molag.model.gnn import GraphNeuralNetworkBase


class ExampleGraphNeuralNetworkModel(GraphNeuralNetworkBase):
    def forward(self, data: Data | Batch, **kwargs: Any) -> dict[str, Any]:
        return {"data": self.ensure_batch(data)}


def test_single_graph_is_wrapped_in_batch() -> None:
    model = ExampleGraphNeuralNetworkModel(in_dim=2)

    output = model(Data())

    assert isinstance(output["data"], Batch)
    assert output["data"].num_graphs == 1


def test_existing_batch_is_preserved() -> None:
    model = ExampleGraphNeuralNetworkModel(in_dim=2)
    batch = Batch.from_data_list([Data(), Data()])

    assert model.ensure_batch(batch) is batch


def test_input_dimension_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ExampleGraphNeuralNetworkModel(in_dim=0)
