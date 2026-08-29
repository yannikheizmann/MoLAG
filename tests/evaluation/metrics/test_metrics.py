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
    metrics = PartitionMetrics()
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


def test_empty_metrics_return_empty_mapping() -> None:
    assert AffinityMetrics().compute() == {}
    assert PartitionMetrics().compute() == {}
