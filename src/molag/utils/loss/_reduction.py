import math

import torch
from torch import Tensor


def grouped_maximum(values: Tensor, group_ids: Tensor, n_groups: int) -> Tensor:
    """Return each group's maximum, using negative infinity for empty groups."""
    result = torch.full(
        (n_groups,),
        float("-inf"),
        dtype=values.dtype,
        device=values.device,
    )
    result.scatter_reduce_(
        0,
        group_ids,
        values,
        reduce="amax",
        include_self=True,
    )
    return result


def grouped_logsumexp(
    values: Tensor,
    group_ids: Tensor,
    n_groups: int,
) -> Tensor:
    """Return a numerically stable log-sum-exp for each group."""
    maxima = grouped_maximum(values.detach(), group_ids, n_groups)
    shift = torch.where(maxima.isfinite(), maxima, torch.zeros_like(maxima))
    sums = torch.zeros(n_groups, dtype=values.dtype, device=values.device)
    sums.scatter_add_(0, group_ids, torch.exp(values - shift[group_ids]))
    return shift + torch.log(sums)


def grouped_soft_maximum(
    values: Tensor,
    group_ids: Tensor,
    n_groups: int,
    beta: float,
    handicap: Tensor | None = None,
) -> Tensor:
    """Return a weighted soft maximum for each group.

    ``handicap`` is expressed in the same units as ``values``. Infinite handicap
    removes an entry. Empty groups return negative infinity.
    """
    if math.isinf(beta):
        adjusted = values if handicap is None else values - handicap
        return grouped_maximum(adjusted, group_ids, n_groups)

    scaled = beta * values
    if handicap is None:
        numerator = grouped_logsumexp(scaled, group_ids, n_groups)
        counts = torch.zeros(n_groups, dtype=values.dtype, device=values.device)
        counts.scatter_add_(0, group_ids, torch.ones_like(values))
        denominator = torch.log(counts)
    else:
        log_weights = -beta * handicap
        numerator = grouped_logsumexp(
            scaled + log_weights,
            group_ids,
            n_groups,
        )
        denominator = grouped_logsumexp(log_weights, group_ids, n_groups)

    result = (numerator - denominator) / beta
    return torch.where(
        numerator.isfinite(),
        result,
        torch.full_like(result, float("-inf")),
    )
