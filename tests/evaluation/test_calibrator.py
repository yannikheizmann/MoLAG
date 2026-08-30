import pytest
from torch import nn

from molag.dataset import EvalDataset, EvalSample, PyGTrackingAffinityCollator
from molag.evaluation import (
    AffinityMetrics,
    PartitionMetrics,
    ThresholdCalibrator,
)
from molag.inference import PredictionGenerator
from molag.model.gnn.blocks import upper_tri_mask


class PositiveEdgeModel(nn.Module):
    def forward(self, data):
        pair_count = int(upper_tri_mask(data.edge_index).sum())
        return {"edge_logits": data.x.new_full((pair_count,), 100.0)}


def predictions(dataset: EvalDataset):
    return PredictionGenerator(
        model=PositiveEdgeModel(),
        dataset=dataset,
        data_collator=PyGTrackingAffinityCollator(),
        batch_size=1,
        device="cpu",
    ).predict()


def test_calibrator_uses_higher_threshold_to_break_ties() -> None:
    dataset = EvalDataset(
        name="tie",
        profile="profile.yaml",
        size=1,
        seed=0,
        created_at="2026-01-01T00:00:00+00:00",
        candidate_seed_ranges=[[0, 0]],
        samples=[
            EvalSample(
                x=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                y=[[0, 0], [0, 1], [0, 2]],
            )
        ],
    )
    calibrator = ThresholdCalibrator(
        metric_factory=lambda threshold: PartitionMetrics(threshold),
        objective="partition_accuracy",
        thresholds=[0.2, 0.5, 0.8],
    )

    result = calibrator.calibrate(predictions(dataset))

    assert result.threshold == 0.8
    assert result.objective == "partition_accuracy"
    assert result.objective_value == 1.0
    assert set(result.metrics_by_threshold) == {0.2, 0.5, 0.8}
    for metrics in result.metrics_by_threshold.values():
        assert metrics["partition_accuracy"] == 1.0
        assert metrics["real_merge_rate"] == 0.0
        assert metrics["real_split_rate"] == 0.0
        assert metrics["spurious_bridge_rate"] == 0.0


def test_calibrator_supports_affinity_metric_objectives() -> None:
    dataset = EvalDataset(
        name="affinity",
        profile="profile.yaml",
        size=1,
        seed=0,
        created_at="2026-01-01T00:00:00+00:00",
        candidate_seed_ranges=[[0, 0]],
        samples=[
            EvalSample(
                x=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                y=[[0, 0], [0, 1], [0, 2]],
            )
        ],
    )
    calibrator = ThresholdCalibrator(
        metric_factory=lambda threshold: AffinityMetrics(threshold),
        objective="edge_f1",
        thresholds=[0.5],
    )

    result = calibrator.calibrate(predictions(dataset))

    assert result.threshold == 0.5
    assert result.objective == "edge_f1"
    assert result.objective_value == 1.0
    assert result.metrics_by_threshold[0.5] == {
        "edge_accuracy": 1.0,
        "edge_precision": 1.0,
        "edge_recall": 1.0,
        "edge_f1": 1.0,
    }


def test_calibrator_requires_thresholds() -> None:
    dataset = EvalDataset(
        name="empty",
        profile="profile.yaml",
        size=1,
        seed=0,
        created_at="2026-01-01T00:00:00+00:00",
        candidate_seed_ranges=[[0, 0]],
        samples=[EvalSample(x=[[0.0, 0.0]], y=[[0, 0]])],
    )

    with pytest.raises(ValueError, match="at least one"):
        ThresholdCalibrator(
            metric_factory=lambda threshold: PartitionMetrics(threshold),
            objective="partition_accuracy",
            thresholds=[],
        )


def test_calibrator_rejects_objective_not_provided_by_metric() -> None:
    dataset = EvalDataset(
        name="invalid-objective",
        profile="profile.yaml",
        size=1,
        seed=0,
        created_at="2026-01-01T00:00:00+00:00",
        candidate_seed_ranges=[[0, 0]],
        samples=[EvalSample(x=[[0.0, 0.0]], y=[[0, 0]])],
    )
    calibrator = ThresholdCalibrator(
        metric_factory=lambda threshold: PartitionMetrics(threshold),
        objective="edge_f1",
        thresholds=[0.5],
    )

    with pytest.raises(ValueError, match="does not provide objective 'edge_f1'"):
        calibrator.calibrate(predictions(dataset))
