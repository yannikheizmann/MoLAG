"""Semantic edge categories used by connectivity-aware affinity training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from torch import Tensor


@dataclass(frozen=True)
class EdgeCategories:
    same_real: Tensor
    different_real: Tensor
    spurious_real: Tensor
    spurious_spurious: Tensor

    @classmethod
    def from_graph(cls, edge_index: Tensor, tracker_labels: Tensor) -> Self:
        """Categorize edges from the tracker labels of their endpoints."""
        row, col = edge_index
        row_labels = tracker_labels[row]
        col_labels = tracker_labels[col]
        row_real = row_labels >= 0
        col_real = col_labels >= 0
        both_real = row_real & col_real

        return cls(
            same_real=both_real & (row_labels == col_labels),
            different_real=both_real & (row_labels != col_labels),
            spurious_real=row_real ^ col_real,
            spurious_spurious=~row_real & ~col_real,
        )

