"""Numerical utilities for constructing loss functions."""

from ._reduction import grouped_logsumexp, grouped_maximum, grouped_soft_maximum

__all__ = ["grouped_logsumexp", "grouped_maximum", "grouped_soft_maximum"]
