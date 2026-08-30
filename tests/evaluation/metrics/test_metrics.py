import numpy as np

from molag.evaluation import AffinityMetrics, PartitionMetrics
from molag.utils.registry import Registry


def test_metric_implementations_are_registered() -> None:
    assert Registry.get("MetricsBase", "Affinity") is AffinityMetrics
    assert Registry.get("MetricsBase", "Partition") is PartitionMetrics


def test_affinity_metrics_accumulate_across_updates() -> None:
    metrics = AffinityMetrics()
    metrics.update(logits=np.array([2.0, -2.0]), labels=np.array([1, 0]))
    metrics.update(logits=np.array([2.0, -2.0]), labels=np.array([0, 1]))

    result = metrics.compute()

    assert result["edge_accuracy"] == 0.5
    assert result["edge_precision"] == 0.5
    assert result["edge_recall"] == 0.5
    assert result["edge_f1"] == 0.5


def test_partition_metrics_accumulate_scene_diagnostics() -> None:
    metrics = PartitionMetrics(complete_tracker_leds=2)
    edges = np.array([[0, 0, 1], [1, 2, 2]])
    metrics.update(
        logits=np.array([2.0, -2.0, -2.0]),
        tracker_labels=np.array([0, 0, 1]),
        edge_index=edges,
    )
    metrics.update(
        logits=np.array([2.0, 2.0, 2.0]),
        tracker_labels=np.array([0, 0, 1]),
        edge_index=edges,
    )

    result = metrics.compute()

    assert result["partition_accuracy"] == 0.5
    assert result["real_merge_rate"] == 0.5
    assert result["partition_accuracy_real_only"] == 0.5
    assert result["tracker_recovery_rate"] == 0.5
    assert result["complete_tracker_share"] == 0.5
    assert result["complete_tracker_recovery"] == 0.5
    assert result["complete_tracker_extractable"] == 1.0


def test_partition_metrics_report_spurious_attachment_separately() -> None:
    metrics = PartitionMetrics(complete_tracker_leds=2)
    metrics.update(
        logits=np.array([2.0, 2.0, -2.0]),
        tracker_labels=np.array([0, 0, -1]),
        edge_index=np.array([[0, 0, 1], [1, 2, 2]]),
    )

    result = metrics.compute()

    assert result["partition_accuracy"] == 1.0
    assert result["spurious_attachment_rate"] == 1.0

    breakdown = metrics.breakdown()
    assert breakdown["by_n_trackers"]["1"]["partition_accuracy"] == 1.0
    assert breakdown["by_n_spurious"]["1-2"]["spurious_attachment_rate"] == 1.0
    assert breakdown["by_visible_leds"]["2"]["recovery_rate"] == 1.0
    assert breakdown["by_failure_mode"] == {"correct": 1}

    assert metrics.sample_records() == [
        {
            "sample_index": 0,
            "n_nodes": 3,
            "n_real": 2,
            "n_spurious": 1,
            "n_spurious_attached": 1,
            "n_trackers": 1,
            "n_trackers_correct": 1,
            "n_predicted_groups": 1,
            "partition_correct": True,
            "partition_real_only_correct": True,
            "has_real_merge": False,
            "has_real_split": False,
            "spurious_bridge": False,
            "failure_mode": "correct",
            "n_complete_trackers": 1,
            "n_complete_trackers_correct": 1,
            "n_complete_trackers_extractable": 1,
        }
    ]
    assert metrics.tracker_records()[0]["complete"] is True
    assert metrics.tracker_records()[0]["failure_mode"] == "correct"


def test_empty_metrics_return_empty_mapping() -> None:
    assert AffinityMetrics().compute() == {}
    assert PartitionMetrics().compute() == {}
