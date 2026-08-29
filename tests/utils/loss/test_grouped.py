import torch

from molag.utils.loss import grouped_maximum, grouped_soft_maximum


def test_maximum_uses_negative_infinity_for_empty_groups() -> None:
    result = grouped_maximum(
        torch.tensor([1.0, 3.0, 2.0]),
        torch.tensor([0, 0, 2]),
        n_groups=3,
    )

    torch.testing.assert_close(result[[0, 2]], torch.tensor([3.0, 2.0]))
    assert torch.isneginf(result[1])


def test_soft_maximum_distributes_gradient() -> None:
    values = torch.tensor([1.0, 2.0], requires_grad=True)

    result = grouped_soft_maximum(
        values,
        torch.tensor([0, 0]),
        n_groups=1,
        beta=1.0,
    )
    result.backward()

    assert values.grad is not None
    assert torch.all(values.grad > 0)
    torch.testing.assert_close(values.grad.sum(), torch.tensor(1.0))
