import math

import numpy as np

from ._base import MetricsBase


class AffinityMetrics(MetricsBase):
    """Accumulate binary metrics for same-tracker affinity predictions."""

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must lie strictly between 0 and 1")
        self.threshold = threshold
        self._logit_threshold = math.log(threshold / (1 - threshold))
        self.reset()

    def reset(self) -> None:
        self._true_positive = 0
        self._true_negative = 0
        self._false_positive = 0
        self._false_negative = 0

    def update(self, **values) -> None:
        logits = np.asarray(values["logits"], dtype=np.float32).reshape(-1)
        labels = np.asarray(values["labels"], dtype=np.int64).reshape(-1)
        if logits.shape != labels.shape:
            raise ValueError("logits and labels must have the same shape")
        predicted = logits >= self._logit_threshold
        positive = labels == 1
        self._true_positive += int(np.sum(predicted & positive))
        self._true_negative += int(np.sum(~predicted & ~positive))
        self._false_positive += int(np.sum(predicted & ~positive))
        self._false_negative += int(np.sum(~predicted & positive))

    def compute(self) -> dict[str, float]:
        tp = self._true_positive
        tn = self._true_negative
        fp = self._false_positive
        fn = self._false_negative
        total = tp + tn + fp + fn
        if total == 0:
            return {}
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            "edge_accuracy": (tp + tn) / total,
            "edge_precision": precision,
            "edge_recall": recall,
            "edge_f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
        }
