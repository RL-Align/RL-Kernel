# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Public SwiGLU input forms, packed order and autograd on a CPU host."""

import pytest
import torch

from rl_engine.contracts.operators.activation import SwiGLUPackedOrder
from rl_engine.reference.activation.swiglu import NativeSwiGLUOp


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("method", ["forward", "forward_fp32"])
@pytest.mark.parametrize("order", list(SwiGLUPackedOrder))
def test_packed_order_matches_formula_and_separate_inputs(dtype, method, order):
    """Match the FP32 formula and separate-input path, including gradient cast points."""
    x = torch.linspace(-4, 4, 120, dtype=dtype).reshape(2, 3, 20)[..., ::2]
    x = x.detach().requires_grad_(True)
    first, second = x.chunk(2, dim=-1)
    gate, up = (first, second) if order is SwiGLUPackedOrder.GATE_UP else (second, first)
    # Reuse one FP32 gate node so autograd combines both paths before casting
    # the gate gradient back, as in the existing reference's rounding contract.
    gate_fp32 = gate.float()
    expected = (gate_fp32 * torch.sigmoid(gate_fp32)) * up.float()
    if method == "forward":
        expected = expected.to(dtype)
    fn = getattr(NativeSwiGLUOp(), method)
    actual = fn(input=x, packed_order=order)
    separate = fn(input=gate, up=up)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, separate, rtol=0, atol=0)
    upstream = torch.linspace(-1, 1, actual.numel()).reshape_as(actual).to(actual.dtype)
    actual_grad = torch.autograd.grad(actual, x, upstream)[0]
    expected_grad = torch.autograd.grad(expected, x, upstream)[0]
    assert actual_grad.dtype == dtype
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["forward", "forward_fp32"])
def test_reject_ambiguous_or_inapplicable_packed_order(method):
    """Require an order for packed inputs and reject it for separate inputs."""
    fn = getattr(NativeSwiGLUOp(), method)
    x = torch.ones(2, 8)
    with pytest.raises(ValueError, match="packed_order"):
        fn(x)
    with pytest.raises(ValueError, match="single packed input"):
        fn(x, x, packed_order=SwiGLUPackedOrder.GATE_UP)


@pytest.mark.parametrize("shape", [(), (2, 0), (2, 3)])
def test_reject_invalid_packed_shape(shape):
    """Reject scalar inputs and packed widths that cannot form two nonempty halves."""
    with pytest.raises(ValueError):
        NativeSwiGLUOp()(torch.empty(shape), packed_order=SwiGLUPackedOrder.GATE_UP)


@pytest.mark.parametrize("order", list(SwiGLUPackedOrder))
def test_vector_empty_rows_and_dtype_validation(order):
    """Accept vectors and empty leading dimensions while rejecting unsupported dtypes."""
    op = NativeSwiGLUOp()
    assert op(torch.ones(8), packed_order=order).shape == (4,)
    x = torch.empty(2, 0, 8, requires_grad=True)
    y = op(x, packed_order=order)
    assert y.shape == (2, 0, 4)
    assert torch.autograd.grad(y, x, torch.empty_like(y))[0].shape == x.shape
    with pytest.raises(TypeError, match="torch.int32"):
        op(torch.ones(2, 8, dtype=torch.int32), packed_order=order)
