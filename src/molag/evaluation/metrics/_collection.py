"""Combine independent streaming metrics into one evaluation pass."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._base import MetricsBase


class CombinedMetrics(MetricsBase):
    """Composite that fans out one stream to independent metric instances."""

    def __init__(self, metrics: Sequence[MetricsBase]) -> None:
        if not metrics:
            raise ValueError("metrics must contain at least one implementation")
        self._metrics = list(metrics)

    def reset(self) -> None:
        """Reset every contained metric."""
        for metric in self._metrics:
            metric.reset()

    def update(self, **values: Any) -> None:
        """Forward one update to every contained metric."""
        for metric in self._metrics:
            metric.update(**values)

    def compute(self) -> dict[str, float]:
        """Merge scalar results while rejecting duplicate metric names."""
        result: dict[str, float] = {}
        for metric in self._metrics:
            values = metric.compute()
            duplicates = result.keys() & values.keys()
            if duplicates:
                names = ", ".join(sorted(duplicates))
                raise ValueError(f"duplicate metric names: {names}")
            result.update(values)
        return result

    def breakdown(self) -> dict[str, Any]:
        """Merge structured breakdowns while rejecting duplicate names."""
        result: dict[str, Any] = {}
        for metric in self._metrics:
            values = metric.breakdown()
            duplicates = result.keys() & values.keys()
            if duplicates:
                names = ", ".join(sorted(duplicates))
                raise ValueError(f"duplicate breakdown names: {names}")
            result.update(values)
        return result

    def sample_records(self) -> list[dict[str, Any]]:
        """Merge scene records by sample index."""
        return self._merge_records("sample records", ("sample_index",))

    def tracker_records(self) -> list[dict[str, Any]]:
        """Merge tracker records by sample and tracker identifiers."""
        return self._merge_records(
            "tracker records", ("sample_index", "tracker_id")
        )

    def _merge_records(
        self,
        record_type: str,
        keys: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[Any, ...], dict[str, Any]] = {}
        for metric in self._metrics:
            records = (
                metric.sample_records()
                if keys == ("sample_index",)
                else metric.tracker_records()
            )
            for record in records:
                missing = [key for key in keys if key not in record]
                if missing:
                    raise ValueError(
                        f"{record_type} must contain keys: {', '.join(keys)}"
                    )
                identity = tuple(record[key] for key in keys)
                target = merged.setdefault(
                    identity, {key: record[key] for key in keys}
                )
                values = {
                    key: value
                    for key, value in record.items()
                    if key not in keys
                }
                duplicates = target.keys() & values.keys()
                if duplicates:
                    names = ", ".join(sorted(duplicates))
                    raise ValueError(f"duplicate {record_type} fields: {names}")
                target.update(values)
        return list(merged.values())
