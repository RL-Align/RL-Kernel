# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Check benchmark shapes and graph lifetimes without importing Triton on CPU."""

import pytest
import torch

from benchmarks.models import benchmark_h3_mm_rope as bench


def test_case_plan_uses_real_packed_lengths():
    plan = bench.case_plan(bench.parse_args([]))
    assert len(plan) == 16
    assert {case["shape"][1] for case in plan} == {264, 925, 3160, 73426}
    assert {case["shape"][2] for case in plan} == {14, 56}
    for layout, config in bench.LAYOUTS.items():
        positions = bench.h3_fl2va_layout(**config, device="cpu")["position_ids"]
        assert all(case["shape"][1] == len(positions) for case in plan if case["layout"] == layout)


@pytest.mark.parametrize("mode", bench.MODES)
def test_timed_call_replays_without_accumulating_grad_or_retaining_fresh_graphs(mode):
    x = torch.randn(1, 3, 1, 128)
    cos, sin = torch.ones(3, 96), torch.zeros(3, 96)
    upstream = torch.randn_like(x)
    call = bench.timed_call(bench.NativeH3MMRoPEOp(), mode, x, cos, sin, upstream)
    for _ in range(3):
        actual = call()
        torch.testing.assert_close(actual, x if mode == "forward" else upstream)
        assert not actual.requires_grad
        assert x.grad is None


def test_accuracy_gate_rejects_wrong_candidate():
    x = torch.ones(1, 2, 1, 128)
    tables = torch.ones(2, 96), torch.zeros(2, 96)
    operators = {"native": bench.NativeH3MMRoPEOp(), "triton": lambda x, c, s: x * 2}
    with pytest.raises(AssertionError):
        bench.check_case(operators, x, *tables, torch.ones_like(x))


def test_model_parity_does_not_bypass_the_independent_accuracy_gate():
    """The saved BF16 outlier must prevent timing even when both paths agree."""
    x = torch.zeros(1, 1, 1, 128, dtype=torch.bfloat16)
    x[..., 0], x[..., 48] = 3.296875, 3.609375
    cos = torch.full((1, 96), 0.7241835594177246)
    sin = torch.full((1, 96), 0.6896072626113892)
    op = bench.NativeH3MMRoPEOp()
    assert torch.equal(op(x, cos, sin), bench.provider_apply_rotary(x, cos, sin))
    with pytest.raises(AssertionError, match="forward_accuracy"):
        bench.check_case({"native": op, "triton": op}, x, cos, sin, torch.ones_like(x))
