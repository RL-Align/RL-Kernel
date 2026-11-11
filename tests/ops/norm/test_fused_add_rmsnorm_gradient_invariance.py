# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Exact gradient comparisons, distinct from independent-reference accuracy tests."""

import importlib

import pytest
import torch

from rl_engine.ops.autograd.vjp_fp32 import reduce_rows_fp32


def _bits(actual, expected):
    """Check all bytes, including signed zero, without rounding either operand."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    actual = actual.detach().clone(memory_format=torch.contiguous_format)
    expected = expected.detach().clone(memory_format=torch.contiguous_format)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.fixture(scope="module")
def kernels():
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    triton = pytest.importorskip("triton")
    if not hasattr(triton, "jit"):
        pytest.skip("A working Triton runtime is required")
    return importlib.import_module("rl_engine.backends.shared.triton.norm.fused_add_rmsnorm")


def test_summing_reduced_microbatches_is_not_the_logical_row_protocol():
    """A fixed strategy alone cannot make independently rounded partials composable."""
    rows = torch.tensor([[1e20], [1.0], [-1e20], [1.0]], dtype=torch.float32)
    whole = reduce_rows_fp32(rows)
    collapsed = reduce_rows_fp32(rows[:2]) + reduce_rows_fp32(rows[2:])
    assert whole.item() == 1.0 and collapsed.item() == 0.0
    # Retaining the original contributions restores the same arithmetic.
    _bits(reduce_rows_fp32(torch.cat((rows[:2], rows[2:]))), whole)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [33, 65, 257])
def test_all_gradients_match_across_strategies_with_cancellation(
    kernels, dtype, weight_dtype, rows
):
    """The old 32-lane TILED fold gives zero instead of one for this weight VJP."""
    shape = (rows, 129)
    inputs = (
        torch.ones(shape, device="cuda", dtype=dtype, requires_grad=True),
        torch.zeros(shape, device="cuda", dtype=dtype, requires_grad=True),
        torch.ones(shape[-1], device="cuda", dtype=weight_dtype, requires_grad=True),
    )
    grad_y = torch.zeros(shape, device="cuda", dtype=torch.float32)
    grad_y[0], grad_y[1], grad_y[32] = 2**24, -(2**24), 1.0
    grad_residual = torch.full(shape, 0.25, device="cuda")
    reference = None
    for strategy in (None, *kernels.RMSNormWeightGradStrategy):
        op = kernels.TritonFusedAddRMSNormOp(weight_grad_strategy=strategy)
        # 1 + eps rounds to exactly 1 in FP32: normalized == 1, so dw is
        # exactly the ordered sum above, with no approximation in the fixture.
        outputs = op(*inputs, eps=1e-9)
        gradients = torch.autograd.grad(outputs, inputs, (grad_y, grad_residual))
        assert torch.equal(gradients[2], torch.ones_like(inputs[2]))
        values = outputs + gradients
        if reference is not None:
            for actual, expected in zip(values, reference, strict=True):
                _bits(actual, expected)
        reference = values


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strategy", [None, "sequential", "tiled"])
@pytest.mark.parametrize("branch", ["both", "residual_only"])
def test_logical_row_contributions_preserve_all_gradients_across_microbatches(
    kernels, dtype, strategy, branch
):
    """Exercise actual backward plus its exported contributions under changed layouts.

    Physical chunks arrive out of order and have different sizes. Keep each
    token's upstream unchanged, restore the complete logical row set, then use
    the project's FP32 row-fold accumulator and one final weight-dtype cast.
    This deliberately does not claim that summing microbatch weight.grad works.
    """
    generator = torch.Generator().manual_seed(461)
    m, d = 65, 129
    tensors = [torch.randn(shape, generator=generator) for shape in ((m, d), (m, d), (d,))]
    inputs = tuple(t.to(device="cuda", dtype=dtype).requires_grad_(True) for t in tensors)
    upstream = tuple(torch.randn((m, d), generator=generator).cuda() for _ in range(2))
    if branch == "residual_only":
        upstream = (None, upstream[1])
    choice = None if strategy is None else kernels.RMSNormWeightGradStrategy(strategy)
    op = kernels.TritonFusedAddRMSNormOp(weight_grad_strategy=choice)

    def evaluate(values, grads):
        outputs = op(*values)
        if grads[0] is None:
            gradients = torch.autograd.grad(outputs[1], values, grads[1], allow_unused=True)
        else:
            gradients = torch.autograd.grad(outputs, values, grads)
        gradients = tuple(
            torch.zeros_like(value) if grad is None else grad
            for value, grad in zip(values, gradients, strict=True)
        )
        contributions = op.parameter_vjp_contributions_fp32(
            x=values[0], residual=values[1], weight=values[2], grad_output=grads[0]
        )["weight"]
        assert contributions.dtype == torch.float32
        _bits(reduce_rows_fp32(contributions).to(values[2].dtype), gradients[2])
        return outputs, gradients, contributions

    whole_outputs, whole_grads, whole_rows = evaluate(inputs, upstream)
    # Unequal chunks, reversed arrival, and row permutation within each chunk.
    chunks = [
        torch.arange(lo, hi, device="cuda").flip(0)
        for lo, hi in ((0, 1), (1, 17), (17, 33), (33, 65))
    ]
    restored_rows = torch.empty_like(whole_rows)
    for ids in reversed(chunks):
        values = tuple(t[ids].detach().requires_grad_(True) for t in inputs[:2]) + (
            inputs[2].detach().requires_grad_(True),
        )
        grads = tuple(None if g is None else g[ids] for g in upstream)
        outputs, gradients, contributions = evaluate(values, grads)
        for actual, expected in zip(
            outputs + gradients[:2], whole_outputs + whole_grads[:2], strict=True
        ):
            _bits(actual, expected[ids])
        _bits(contributions, whole_rows[ids])
        restored_rows[ids] = contributions
    _bits(reduce_rows_fp32(restored_rows).to(inputs[2].dtype), whole_grads[2])


def test_contribution_export_handles_noncurrent_device_and_strides(kernels):
    """The optional integration path has the same device/stream guard as backward."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Two GPUs are required")
    with torch.cuda.device(0):
        x = torch.ones((3, 14), device="cuda:1")[:, ::2]
        residual = torch.zeros_like(x)
        weight = torch.ones(7, device="cuda:1")
        grad_y = torch.ones((3, 14), device="cuda:1")[:, ::2]
        op = kernels.TritonFusedAddRMSNormOp()
        rows = op.parameter_vjp_contributions_fp32(
            x=x, residual=residual, weight=weight, grad_output=grad_y, eps=1e-9
        )["weight"]
        assert torch.cuda.current_device() == 0
        assert rows.device == x.device
        _bits(rows, torch.ones_like(x))
