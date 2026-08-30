"""Compute confidence intervals used by evaluation summaries."""

from __future__ import annotations

from collections.abc import Sequence


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    z_squared = z**2
    denominator = 1 + z_squared / total
    centre = (proportion + z_squared / (2 * total)) / denominator
    radius = (
        z
        * (
            proportion * (1 - proportion) / total
            + z_squared / (4 * total**2)
        )
        ** 0.5
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def clustered_ratio_interval(
    numerators: Sequence[int],
    denominators: Sequence[int],
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a ratio interval treating paired entries as independent clusters."""
    if len(numerators) != len(denominators):
        raise ValueError("numerators and denominators must have the same length")
    total = sum(denominators)
    n_clusters = sum(int(value > 0) for value in denominators)
    if total <= 0 or n_clusters < 2:
        return 0.0, 0.0
    ratio = sum(numerators) / total
    squared_residuals = sum(
        (numerator - ratio * denominator) ** 2
        for numerator, denominator in zip(numerators, denominators, strict=True)
    )
    variance = (
        n_clusters
        / (n_clusters - 1)
        * squared_residuals
        / total**2
    )
    radius = z * variance**0.5
    return max(0.0, ratio - radius), min(1.0, ratio + radius)
