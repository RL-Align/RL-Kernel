# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Preserve the FP32-to-FP16 rounding boundary across scalar and vector layouts."""

import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from rl_engine.backends.shared.triton.activation.swiglu import (  # noqa: E402
    _cast_preserving_zero_sign,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm required")


@triton.jit
def _cast_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x_ptr + offsets, offsets < N, other=0)
    tl.store(y_ptr + offsets, _cast_preserving_zero_sign(value, tl.float16), offsets < N)


@triton.jit
def _product_kernel(a_ptr, b_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offsets, offsets < N, other=0)
    b = tl.load(b_ptr + offsets, offsets < N, other=0).to(tl.float32)
    value = a * b
    tl.store(y_ptr + offsets, _cast_preserving_zero_sign(value, tl.float16), offsets < N)


@pytest.mark.parametrize("block", [1, 1024])
def test_fp16_cast_at_every_finite_midpoint(block):
    """Check both signs, exact ties, adjacent FP32 values, subnormals and overflow."""
    finite = torch.arange(0x7C00, dtype=torch.int32).to(torch.int16).view(torch.float16).float()
    midpoints = (finite[:-1] + finite[1:]) * 0.5
    positive = torch.cat(
        (
            finite,
            torch.nextafter(midpoints, torch.full_like(midpoints, -torch.inf)),
            midpoints,
            torch.nextafter(midpoints, torch.full_like(midpoints, torch.inf)),
            torch.tensor([65520.0, torch.finfo(torch.float32).max, torch.inf]),
            torch.nextafter(torch.tensor([65520.0]), torch.tensor([-torch.inf])),
            torch.nextafter(torch.tensor([65520.0]), torch.tensor([torch.inf])),
        )
    )
    values = torch.cat((positive, -positive))
    actual = torch.empty(values.numel(), device="cuda", dtype=torch.float16)
    _cast_kernel[(triton.cdiv(values.numel(), block),)](
        values.cuda(), actual, values.numel(), block, enable_fp_fusion=False
    )
    assert torch.equal(actual.cpu().view(torch.int16), values.half().view(torch.int16))


@pytest.mark.parametrize("block", [1, 1024])
def test_fp16_product_rounds_to_fp32_before_cast(block):
    """These products expose the gfx942 mixed-instruction versus FP32 rounding difference."""
    a = torch.tensor(
        [
            -0.41481348872184753,
            -0.2489650547504425,
            -0.3876459002494812,
            0.8100247383117676,
            0.09945843368768692,
            -0.0,
            0.0,
        ]
    )
    b = torch.tensor(
        [0.170166015625, -0.53076171875, 0.188232421875, 0.591796875, 0.62548828125, 1.0, -1.0],
        dtype=torch.float16,
    )
    expected = (a * b.float()).half()
    actual = torch.empty(a.numel(), device="cuda", dtype=torch.float16)
    _product_kernel[(triton.cdiv(a.numel(), block),)](
        a.cuda(), b.cuda(), actual, a.numel(), block, enable_fp_fusion=False
    )
    assert torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))


def test_fp16_cast_preserves_nonfinite_classification():
    """Infinity signs survive and either sign of NaN remains a NaN."""
    values = torch.tensor([torch.inf, -torch.inf, torch.nan, -torch.nan], device="cuda")
    actual = torch.empty_like(values, dtype=torch.float16)
    _cast_kernel[(1,)](values, actual, values.numel(), 1024, enable_fp_fusion=False)
    assert torch.equal(actual[:2].view(torch.int16), values[:2].half().view(torch.int16))
    assert torch.isnan(actual[2:]).all()
