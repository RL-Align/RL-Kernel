# SPDX-License-Identifier: Apache-2.0
"""Execute the actual Triton bodies on CPU; no native GPU evidence is implied."""

import os

import pytest
import torch

from rl_engine.contracts.operators.fixed_output_combine import prepare_fixed_output_combine
from rl_engine.reference.moe.fixed_output_combine import fixed_output_combine_reference
from tests.helpers.dsv4_output_combine import request

pytestmark = pytest.mark.skipif(
    os.getenv("TRITON_INTERPRET") != "1", reason="requires TRITON_INTERPRET=1"
)


def raw_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("h", [1, 255, 257])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_kernel_accumulation_debug_invariance_and_vjp(h, shared, dtype):
    pytest.importorskip("triton")
    from rl_engine.backends.rocm.moe.output_combine_kernels import _combine, _output_vjp

    plan, inputs = request(t=2, h=h, dtype=dtype, shared=shared)
    p = prepare_fixed_output_combine(plan, inputs, expected_mapping_run_id="run-53")
    x, s = inputs.expert_output, inputs.shared_output
    # Include cancellation and signed zeros in the same fixtures.
    rows = p.canonical_rows()[0]
    x[rows, 0] = torch.tensor([2**24, 1, -(2**24), 1, -0.0, 0.0], dtype=dtype)
    reference = fixed_output_combine_reference(p, x, s, mapping_run_id="run-53")
    outputs = []
    for debug in (False, True):
        out, steps, before = (
            torch.empty((2, h), dtype=dtype),
            torch.empty(2, 6, h),
            torch.empty(2, h),
        )
        _combine[(2, (h + 255) // 256)](
            x,
            p._rows,
            out if s is None else s,
            out,
            steps,
            before,
            h,
            *x.stride(),
            *((0, 0) if s is None else s.stride()),
            shared,
            debug,
            256,
        )
        outputs.append(out)
        if dtype == torch.float32:
            raw_equal(out, reference.output)
        # Triton 3.7's InterpreterBuilder.cast_impl uses rounding_mode=None
        # for FP32->BF16, unlike native round-to-nearest-even. Do not treat that
        # truncation as the operator oracle. Check every FP32 stage below and
        # keep the exact BF16 output assertion in the real GPU suite.
        if debug:
            raw_equal(steps, reference.slot_steps)
            raw_equal(before, reference.before_cast)
    raw_equal(outputs[0], outputs[1])
    dy = torch.arange(2 * h * 2).reshape(2, h * 2).to(dtype)[:, ::2]
    dx = torch.empty((12, h), dtype=dtype)
    _output_vjp[(12, (h + 255) // 256)](dy, p._tokens, dx, h, *dy.stride(), 256)
    raw_equal(dx, dy.index_select(0, inputs.token_indices))
