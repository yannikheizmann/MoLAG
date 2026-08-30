import pytest
from torch import nn

from molag.dataset import EvalDataset, EvalSample, PyGTrackingAffinityCollator
from molag.evaluation import AffinityMetrics, PartitionMetrics, ThresholdCalibrator
from molag.model.gnn.blocks import upper_tri_mask


class PositiveEdgeModel(nn.Module):
    def forward(self, data):
        pair_count = int(upper_tri_mask(data.edge_index).sum())
        return {"edge_logits": data.x.new_full((pair_count,), 100.0)}


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
        model=PositiveEdgeModel(),
        dataset=dataset,
        data_collator=PyGTrackingAffinityCollator(),
        metric_factory=lambda threshold: PartitionMetrics(threshold),
        objective="partition_accuracy",
        thresholds=[0.2, 0.5, 0.8],
        batch_size=1,
        device="cpu",
    )

    result = calibrator.calibrate()

    assert result.threshold == 0.8
    assert result.objective == "partition_accuracy"
    assert result.objective_value == 1.0
    expected_metrics = {
        "partition_accuracy": 1.0,
        "real_merge_rate": 0.0,
        "real_split_rate": 0.0,
        "spurious_bridge_rate": 0.0,
    }
    assert result.metrics_by_threshold == {
        0.2: expected_metrics,
        0.5: expected_metrics,
        0.8: expected_metrics,
    }


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
        model=PositiveEdgeModel(),
        dataset=dataset,
        data_collator=PyGTrackingAffinityCollator(),
        metric_factory=lambda threshold: AffinityMetrics(threshold),
        objective="edge_f1",
        thresholds=[0.5],
        batch_size=1,
        device="cpu",
    )

    result = calibrator.calibrate()

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
            model=PositiveEdgeModel(),
            dataset=dataset,
            data_collator=PyGTrackingAffinityCollator(),
            metric_factory=lambda threshold: PartitionMetrics(threshold),
            objective="partition_accuracy",
            thresholds=[],
            batch_size=1,
            device="cpu",
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
        model=PositiveEdgeModel(),
        dataset=dataset,
        data_collator=PyGTrackingAffinityCollator(),
        metric_factory=lambda threshold: PartitionMetrics(threshold),
        objective="edge_f1",
        thresholds=[0.5],
        batch_size=1,
        device="cpu",
    )

    with pytest.raises(ValueError, match="does not provide objective 'edge_f1'"):
        calibrator.calibrate()
