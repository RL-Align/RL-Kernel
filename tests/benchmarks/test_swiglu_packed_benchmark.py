# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Exercise the public benchmark's planning and fail-closed accuracy gate."""

import json
from functools import partial

import pytest
import torch

from benchmarks.operators.activation import benchmark_swiglu_packed as bench
from rl_engine.contracts.operators.activation import SwiGLUPackedOrder
from rl_engine.reference.activation.swiglu import NativeSwiGLUOp


def test_default_plan_and_cpu_dry_run(capsys):
    """Check planned measurement counts and dry-run output without a GPU."""
    assert bench.case_plan(bench.parse_args([]))["measurement_count"] == 972
    bench.main(["--dry-run", "--rows", "1", "3", "--dtypes", "bf16", "--rounds", "1"])
    assert json.loads(capsys.readouterr().out)["measurement_count"] == 36


def test_separate_plan_and_gradient_gate():
    """Verify two-input planning and reject wrong gradients even with correct outputs."""
    args = bench.parse_args(["--input-layout", "separate"])
    plan = bench.case_plan(args)
    assert plan["measurement_count"] == 648
    assert plan["input_count"] == 2
    assert plan["input_shapes"][0] == [1, 1, args.width]
    assert plan["packed_order"] is None
    inputs = tuple(torch.ones(1, 2, 4, requires_grad=True) for _ in range(2))
    upstream = torch.ones_like(inputs[0])
    reference = NativeSwiGLUOp()
    providers = {"pytorch": reference, "triton": reference}
    errors = bench.check_separate_case(providers, inputs, upstream)
    assert all(check["max_abs_error"] == 0 for check in errors.values())
    for mode in ("backward", "forward_backward"):
        gradients = bench.make_call(reference, inputs, upstream, mode)()
        assert len(gradients) == 2
        expected = reference(gate=inputs[0], up=inputs[1])
        bench.assert_bits(gradients, torch.autograd.grad(expected, inputs, upstream))
    providers["triton"] = lambda gate, up: reference(gate=gate, up=up) + (up - up.detach())
    with pytest.raises(AssertionError):
        bench.check_separate_case(providers, inputs, upstream)


@pytest.mark.parametrize(
    "argv",
    [
        ["--rows", "0"],
        ["--width", "-1"],
        ["--rounds", "0"],
        ["--dtypes", "bf16", "bf16"],
    ],
)
def test_reject_invalid_plan(argv):
    """Reject empty shapes, invalid timing counts and duplicate dtype cases."""
    with pytest.raises(SystemExit):
        bench.parse_args(argv)


@pytest.mark.parametrize("order", list(SwiGLUPackedOrder))
def test_accuracy_gate_and_shared_agreement_are_independent(order):
    """Reject matching provider results when their outputs or gradients are inaccurate."""
    reference = partial(NativeSwiGLUOp(), packed_order=order)
    x = torch.ones(1, 2, 8, requires_grad=True)
    upstream = torch.ones(1, 2, 4)
    providers = {name: reference for name in bench.PROVIDERS}
    errors = bench.check_case(providers, x, upstream, order)
    assert errors["forward_accuracy"]["max_abs_error"] == 0
    for mode in ("forward", "backward", "forward_backward"):
        actual = bench.make_call(reference, (x,), upstream, mode)()
        expected = reference(packed_input=x)
        if mode != "forward":
            expected = torch.autograd.grad(expected, x, upstream)
        bench.assert_bits(actual, expected)
    # Agreement alone must not pass a numerically wrong implementation.
    providers.update(
        packed=lambda packed_input: reference(packed_input=packed_input) + 1,
        adapter=lambda packed_input: reference(packed_input=packed_input) + 1,
    )
    with pytest.raises(AssertionError):
        bench.check_case(providers, x, upstream, order)
    providers.update(
        packed=lambda packed_input: reference(packed_input=packed_input)
        + (packed_input[..., :4] - packed_input[..., :4].detach()),
        adapter=lambda packed_input: reference(packed_input=packed_input)
        + (packed_input[..., :4] - packed_input[..., :4].detach()),
    )
    with pytest.raises(AssertionError):
        bench.check_case(providers, x, upstream, order)


def test_partial_report_is_not_complete(tmp_path):
    """Keep interrupted reports incomplete and report the actual measurement count."""
    report = {
        "complete": False,
        "case_plan": bench.case_plan(bench.parse_args(["--rows", "1", "--rounds", "1"])),
        "settings": {"rounds": 1},
        "results": [],
    }
    bench.write_report(report, tmp_path)
    assert not json.loads((tmp_path / "results.json").read_text())["complete"]
    assert "0/54" in (tmp_path / "report.md").read_text()
