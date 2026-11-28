# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Check benchmark shapes and graph lifetimes without importing Triton on CPU."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from benchmarks.models import benchmark_h3_mm_rope as bench


def test_case_plan_uses_real_packed_lengths():
    plan = bench.case_plan(bench.parse_args([]))
    assert len(plan) == 16
    case_ids = {case["case_id"] for case in plan}
    assert len(case_ids) == len(plan)
    assert "bf16:long:1x73426x56x128" in case_ids
    assert {case["shape"][1] for case in plan} == {264, 925, 3160, 73426}
    assert {case["shape"][2] for case in plan} == {14, 56}
    for layout, config in bench.LAYOUTS.items():
        positions = bench.h3_fl2va_layout(**config, device="cpu")["position_ids"]
        assert all(case["shape"][1] == len(positions) for case in plan if case["layout"] == layout)


def test_repeated_filters_select_each_benchmark_case_once():
    args = bench.parse_args(
        ["--layouts", "tiny", "tiny", "--heads", "14", "14", "--dtypes", "bf16", "bf16"]
    )
    assert [case["case_id"] for case in bench.case_plan(args)] == ["bf16:tiny:1x264x14x128"]


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


@pytest.mark.parametrize("optimize", [0, 1], ids=["normal", "python-O"])
def test_bitwise_gate_rejects_one_ulp_difference(optimize):
    """A one-ULP change at 1.0 passes the FP32 tolerance but must fail bytewise parity."""
    source = Path(bench.__file__)
    namespace = {"__name__": bench.__name__, "__file__": str(source)}
    # Compile the benchmark itself with the same assert-removal flag as python -O.
    exec(compile(source.read_text(), str(source), "exec", optimize=optimize), namespace)

    x = torch.ones(1, 2, 1, 128)
    tables = torch.ones(2, 96), torch.zeros(2, 96)
    operators = {
        "native": bench.NativeH3MMRoPEOp(),
        "triton": lambda x, c, s: x + torch.finfo(x.dtype).eps,
    }
    with pytest.raises(AssertionError, match=r"bitwise mismatch against Native forward.*\[0, 2\)"):
        namespace["check_case"](operators, x, *tables, torch.ones_like(x))


def test_bitwise_gate_rejects_wrong_dtype_even_when_all_bytes_match():
    x = torch.zeros(1, 1, 1, 128, dtype=torch.bfloat16)
    operators = {"native": bench.NativeH3MMRoPEOp(), "triton": lambda x, c, s: x.half()}
    with pytest.raises(AssertionError, match="bitwise mismatch against Native forward"):
        bench.check_case(operators, x, torch.ones(1, 96), torch.zeros(1, 96), torch.zeros_like(x))


def test_check_case_uses_full_candidate_and_chunked_native():
    """The last token exercises a second reference slice in both directions."""
    length = 1025
    x = torch.ones(1, length, 1, 128)
    upstream = torch.ones_like(x)
    cos, sin = torch.ones(length, 96), torch.zeros(length, 96)
    x[:, -1], upstream[:, -1], cos[-1] = 2, 3, -1
    native = Mock(wraps=bench.NativeH3MMRoPEOp())
    candidate = Mock(wraps=bench.NativeH3MMRoPEOp())

    checks = bench.check_case({"native": native, "triton": candidate}, x, cos, sin, upstream)

    assert [call.args[0].shape[1] for call in candidate.call_args_list] == [length]
    assert [call.args[0].shape[1] for call in native.call_args_list] == [1024, 1]
    for direction in ("forward", "backward", "provider_backward"):
        assert checks["unrounded_math_diagnostic"][direction]["elements"] == x.numel()


def test_model_boundary_acceptance_keeps_unrounded_outlier_diagnostic(tmp_path):
    """The saved BF16 outlier can be timed, but its mathematical gap stays visible."""
    x = torch.zeros(1, 1, 1, 128, dtype=torch.bfloat16)
    x[..., 0], x[..., 48] = 3.296875, 3.609375
    cos = torch.full((1, 96), 0.7241835594177246)
    sin = torch.full((1, 96), 0.6896072626113892)
    op = bench.NativeH3MMRoPEOp()
    assert torch.equal(op(x, cos, sin), bench.provider_apply_rotary(x, cos, sin))
    result = bench.check_case({"native": op, "triton": op}, x, cos, sin, torch.ones_like(x))
    assert result["acceptance"] == dict(reference="forward_fp32", passed=True)
    diagnostic = result["unrounded_math_diagnostic"]
    assert diagnostic["reference"] == "forward_unrounded_fp32"
    assert diagnostic["is_acceptance_gate"] is False
    assert diagnostic["forward"]["outside_tolerance"] == 1
    assert diagnostic["forward"]["max_error_over_tolerance"] > 1
    assert diagnostic["forward"]["examples"][0]["index"] == [0, 0, 0, 0]
    bench.write_report(
        tmp_path,
        dict(
            environment={},
            complete=True,
            results=[],
            failures=[],
            validation=[dict(dtype="bf16", shape=list(x.shape), **result)],
        ),
    )
    assert "not an acceptance gate" in (tmp_path / "report.md").read_text()


@pytest.mark.parametrize(
    "bad_value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "posinf", "neginf"]
)
def test_nonfinite_diagnostic_survives_later_finite_chunks(monkeypatch, tmp_path, bad_value):
    """Inject one non-finite reference value in the middle of three token slices."""
    monkeypatch.setattr(bench, "_VALIDATION_CHUNK_TOKENS", 1)
    monkeypatch.setattr(
        bench.NativeH3MMRoPEOp,
        "forward_unrounded_fp32",
        lambda self, x, cos, sin: torch.where(x < 0, bad_value, x),
    )
    x = torch.ones(1, 3, 1, 128)
    x[0, 1, 0, 0] = -1
    op = bench.NativeH3MMRoPEOp()
    checks = bench.check_case(
        {"triton": op, "native": op}, x, torch.ones(3, 96), torch.zeros(3, 96), torch.ones_like(x)
    )
    case = dict(dtype="fp32", shape=list(x.shape), **checks)
    report = dict(environment={}, complete=True, results=[], failures=[], validation=[case])
    bench.write_report(tmp_path, report)

    saved = json.loads((tmp_path / "results.json").read_text())["validation"][0]
    assert saved["acceptance"]["passed"]
    forward = saved["unrounded_math_diagnostic"]["forward"]
    assert forward["elements"] == x.numel() and forward["outside_tolerance"] == 1
    assert forward["max_abs_error"] is None and forward["max_error_over_tolerance"] is None
    assert forward["examples"][0]["index"] == [0, 1, 0, 0]
    assert forward["examples"][0]["reference"] == str(bad_value)
    assert "| 1 / 384 | non-finite | non-finite |" in (tmp_path / "report.md").read_text()


def test_independent_model_boundary_gate_rejects_shared_wrong_backward():
    """Agreeing Native/candidate stubs cannot turn a wrong VJP into acceptance."""

    class WrongBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, dy):
            return dy * 2

    x = torch.ones(1, 2, 1, 128)
    tables = torch.ones(2, 96), torch.zeros(2, 96)
    wrong = lambda x, c, s: WrongBackward.apply(x)  # noqa: E731
    with pytest.raises(AssertionError, match="forward_fp32 gradient_accuracy"):
        bench.check_case({"native": wrong, "triton": wrong}, x, *tables, torch.ones_like(x))
