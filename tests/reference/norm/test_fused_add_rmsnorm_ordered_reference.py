# SPDX-License-Identifier: Apache-2.0
"""Fixed-path RMSNorm checks, with independent mathematics kept separate."""

import pytest
import torch

from rl_engine.reference.norm.fused_add_rmsnorm import (
    NativeFusedAddRMSNormOp,
    ordered_backward,
    ordered_forward,
    ordered_row_sum,
)
from tests.helpers.tensor_checks import assert_bitwise


def test_feature_sum_has_a_declared_order_not_an_exact_sum():
    """The first seven unit increments round away before the second group joins."""
    values = torch.ones((2, 16), dtype=torch.float32)
    values[:, 0] = 2**24
    assert_bitwise(ordered_row_sum(values), torch.full((2, 1), 2**24 + 8.0))
    assert_bitwise(ordered_row_sum(values[:, :7]), torch.full((2, 1), float(2**24)))
    assert values.double().sum(dim=-1)[0].item() == 2**24 + 15
    # Squaring belongs to the operator, not the shared reduction helper.
    x = values.sqrt()
    forward = ordered_forward(x, torch.zeros_like(x), torch.ones(16), eps=1e-5)
    assert_bitwise(forward.sum_squares, ordered_row_sum(values))


@pytest.mark.parametrize("shape", [(7,), (3, 7), (2, 3, 129), (2, 0, 7)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("branch", ["both", "y_only", "residual_only"])
def test_ordered_reference_against_independent_fp64(shape, dtype, branch):
    """Use ordinary FP64/autograd as an independent accuracy oracle, not for bits."""
    generator = torch.Generator().manual_seed(461)
    x = torch.randn(shape, generator=generator).to(dtype)
    residual = torch.randn(shape, generator=generator).to(dtype)
    weight = torch.randn(shape[-1], generator=generator)
    gy = torch.randn(shape, generator=generator) if branch != "residual_only" else None
    gr = torch.randn(shape, generator=generator) if branch != "y_only" else None
    fixed = ordered_forward(x, residual, weight, eps=1e-5)
    backward = ordered_backward(
        fixed.updated_residual,
        fixed.inverse_rms,
        weight,
        gy,
        gr,
        x_dtype=x.dtype,
        residual_dtype=residual.dtype,
    )
    leaves = tuple(value.double().requires_grad_(True) for value in (x, residual, weight))
    u = leaves[0] + leaves[1]
    y = (u / (u.square().mean(dim=-1, keepdim=True) + 1e-5).sqrt()) * leaves[2]
    outputs, upstream = [], []
    for output, grad in ((y, gy), (u, gr)):
        if grad is not None:
            outputs.append(output)
            upstream.append(grad.double())
    gradients = torch.autograd.grad(outputs, leaves, upstream, allow_unused=True)
    gradients = tuple(
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(leaves, gradients, strict=True)
    )
    for actual, expected in zip(
        (
            fixed.y,
            fixed.updated_residual,
            backward.grad_x,
            backward.grad_residual,
            backward.grad_weight,
        ),
        (y, u, *gradients),
        strict=True,
    ):
        tolerance = {torch.float16: 3e-3, torch.bfloat16: 2e-2, torch.float32: 2e-5}[actual.dtype]
        torch.testing.assert_close(
            actual, expected.to(actual.dtype), rtol=tolerance, atol=tolerance
        )
        assert not actual.requires_grad


def test_parameter_fold_retains_row_order_and_final_cast():
    u = torch.ones((33, 7))
    gy = torch.zeros_like(u)
    gy[0], gy[1], gy[32] = 2**24, -(2**24), 1
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        result = ordered_backward(
            u,
            torch.ones(33),
            torch.ones(7, dtype=dtype),
            gy,
            None,
            x_dtype=dtype,
            residual_dtype=torch.float32,
        )
        assert_bitwise(result.grad_weight, torch.ones(7, dtype=dtype))
        assert_bitwise(result.grad_weight_per_row, gy)
        unused = ordered_backward(
            u,
            torch.ones(33),
            torch.ones(7, dtype=dtype),
            None,
            None,
            x_dtype=dtype,
            residual_dtype=torch.float32,
        )
        assert_bitwise(unused.grad_weight, torch.zeros(7, dtype=dtype))
        assert_bitwise(unused.grad_x, torch.zeros_like(u, dtype=dtype))


def test_ordered_rows_keep_their_bits_under_repacking():
    x = torch.randn((5, 34), generator=torch.Generator().manual_seed(462))[:, ::2]
    assert not x.is_contiguous()
    full = ordered_forward(x, -0.25 * x, torch.ones(17), eps=1e-5)
    for indices in (torch.tensor([4, 2, 0]), torch.tensor([1]), torch.tensor([3])):
        part = ordered_forward(x[indices], -0.25 * x[indices], torch.ones(17), eps=1e-5)
        for name in ("y", "updated_residual", "sum_squares", "inverse_rms"):
            assert_bitwise(getattr(part, name), getattr(full, name)[indices])


def test_native_baseline_does_not_use_fixed_path_helpers(monkeypatch):
    """Keep the benchmark/accuracy oracle independent of the declared tree."""
    import rl_engine.reference.norm.fused_add_rmsnorm as reference

    def forbidden(*args, **kwargs):
        raise AssertionError("Native must not call an ordered helper")

    monkeypatch.setattr(reference, "ordered_row_sum", forbidden)
    monkeypatch.setattr(reference, "sum_tile", forbidden)
    x = torch.ones((2, 7), requires_grad=True)
    r = torch.zeros_like(x, requires_grad=True)
    w = torch.ones(7, requires_grad=True)
    y, u = NativeFusedAddRMSNormOp()(x, r, w)
    assert all(gradient is not None for gradient in torch.autograd.grad((y + u).sum(), (x, r, w)))
