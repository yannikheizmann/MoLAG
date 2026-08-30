"""Inference from pairwise MoLAG affinity predictions."""

from ._partition import AffinityPartition
from ._predictor import InferenceResult, MoLAGPredictor

__all__ = ["AffinityPartition", "InferenceResult", "MoLAGPredictor"]
