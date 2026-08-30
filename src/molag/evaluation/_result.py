from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate evaluation output and its detailed diagnostic records."""

    metrics: dict[str, float]
    breakdown: dict[str, Any]
    samples: list[dict[str, Any]]
    trackers: list[dict[str, Any]]

    def write(self, path: str | Path, metadata: dict[str, Any]) -> Path:
        """Write the JSON summary and detailed CSV records."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {**metadata, "metrics": self.metrics, "breakdown": self.breakdown}
        destination.write_text(json.dumps(payload, indent=2) + "\n")
        self._write_csv(destination.parent / "samples.csv", self.samples)
        self._write_csv(destination.parent / "tracker_samples.csv", self.trackers)
        return destination

    @staticmethod
    def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        fieldnames = list(records[0])
        if any(list(record) != fieldnames for record in records):
            raise ValueError("CSV records must have identical fields and ordering")
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
