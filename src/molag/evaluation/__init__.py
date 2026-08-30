"""Evaluation of MoLAG predictions."""

from ._assessment import (
    GroupingFailureMode,
    PartitionAssessment,
    TrackerAssessment,
)
from ._calibrator import CalibrationResult, ThresholdCalibrator
from ._evaluator import Evaluator
from ._loader import ModelLoader
from molag.inference import PredictionCache, PredictionGenerator, ScenePrediction
from ._provenance import EvaluationProvenance, FileFingerprint
from ._result import EvaluationResult
from .metrics import (
    AffinityMetrics,
    CombinedMetrics,
    MetricsBase,
    PartitionMetrics,
    RealAffinityMetrics,
)

__all__ = [
    "AffinityMetrics",
    "CalibrationResult",
    "CombinedMetrics",
    "Evaluator",
    "EvaluationResult",
    "EvaluationProvenance",
    "FileFingerprint",
    "GroupingFailureMode",
    "MetricsBase",
    "ModelLoader",
    "PartitionAssessment",
    "PartitionMetrics",
    "PredictionCache",
    "PredictionGenerator",
    "RealAffinityMetrics",
    "ScenePrediction",
    "ThresholdCalibrator",
    "TrackerAssessment",
]
