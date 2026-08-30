import csv
import json
from pathlib import Path

from molag.evaluation import EvaluationResult


def test_evaluation_result_writes_summary_and_detailed_records(
    tmp_path: Path,
) -> None:
    result = EvaluationResult(
        metrics={"partition_accuracy": 1.0},
        breakdown={"by_failure_mode": {"correct": 1}},
        samples=[{"sample_index": 0, "correct": True}],
        trackers=[{"sample_index": 0, "tracker_id": 3, "correct": True}],
    )

    output = result.write(tmp_path / "evaluation.json", {"threshold": 0.5})

    summary = json.loads(output.read_text())
    assert summary["threshold"] == 0.5
    assert summary["metrics"] == {"partition_accuracy": 1.0}
    with (tmp_path / "samples.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {"sample_index": "0", "correct": "True"}
        ]
    with (tmp_path / "tracker_samples.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {"sample_index": "0", "tracker_id": "3", "correct": "True"}
        ]
