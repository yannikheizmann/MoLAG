import pytest

from molag.evaluation import CombinedMetrics, MetricsBase


class CountingMetrics(MetricsBase):
    def reset(self) -> None:
        self.count = 0

    def update(self, **values) -> None:
        self.count += int(values["count"])

    def compute(self) -> dict[str, float]:
        return {"count": float(self.count)}


def test_collection_updates_all_metrics() -> None:
    first = CountingMetrics()
    second = CountingMetrics()
    collection = CombinedMetrics([first, second])

    collection.reset()
    collection.update(count=3)

    assert first.compute() == {"count": 3.0}
    assert second.compute() == {"count": 3.0}


def test_collection_rejects_duplicate_result_names() -> None:
    collection = CombinedMetrics([CountingMetrics(), CountingMetrics()])
    collection.reset()

    with pytest.raises(ValueError, match="duplicate metric names"):
        collection.compute()


def test_collection_requires_a_metric() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CombinedMetrics([])
