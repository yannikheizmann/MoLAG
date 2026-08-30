import pytest

from molag.evaluation import CombinedMetrics, MetricsBase


class CountingMetrics(MetricsBase):
    def reset(self) -> None:
        self.count = 0

    def update(self, **values) -> None:
        self.count += int(values["count"])

    def compute(self) -> dict[str, float]:
        return {"count": float(self.count)}

    def breakdown(self) -> dict[str, object]:
        return {"counts": {"total": self.count}}


class RecordingMetrics(CountingMetrics):
    def __init__(self, field: str, value: object) -> None:
        self._field = field
        self._value = value

    def sample_records(self) -> list[dict[str, object]]:
        return [{"sample_index": 0, self._field: self._value}]

    def tracker_records(self) -> list[dict[str, object]]:
        return [
            {
                "sample_index": 0,
                "tracker_id": 2,
                self._field: self._value,
            }
        ]


def test_collection_updates_all_metrics() -> None:
    first = CountingMetrics()
    second = CountingMetrics()
    collection = CombinedMetrics([first, second])

    collection.reset()
    collection.update(count=3)

    assert first.compute() == {"count": 3.0}
    assert second.compute() == {"count": 3.0}


def test_collection_combines_breakdowns() -> None:
    collection = CombinedMetrics([CountingMetrics()])
    collection.reset()
    collection.update(count=3)

    assert collection.breakdown() == {"counts": {"total": 3}}


def test_collection_rejects_duplicate_result_names() -> None:
    collection = CombinedMetrics([CountingMetrics(), CountingMetrics()])
    collection.reset()

    with pytest.raises(ValueError, match="duplicate metric names"):
        collection.compute()


def test_collection_requires_a_metric() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CombinedMetrics([])


def test_collection_merges_modular_records_by_identity() -> None:
    collection = CombinedMetrics(
        [RecordingMetrics("first", 1), RecordingMetrics("second", 2)]
    )

    assert collection.sample_records() == [
        {"sample_index": 0, "first": 1, "second": 2}
    ]
    assert collection.tracker_records() == [
        {"sample_index": 0, "tracker_id": 2, "first": 1, "second": 2}
    ]


def test_collection_rejects_duplicate_record_fields() -> None:
    collection = CombinedMetrics(
        [RecordingMetrics("value", 1), RecordingMetrics("value", 2)]
    )

    with pytest.raises(ValueError, match="duplicate sample records fields"):
        collection.sample_records()
