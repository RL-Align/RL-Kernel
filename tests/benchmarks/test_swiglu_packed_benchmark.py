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
    assert bench.case_plan(bench.parse_args([]))["measurement_count"] == 972
    bench.main(["--dry-run", "--rows", "1", "3", "--dtypes", "bf16", "--rounds", "1"])
    assert json.loads(capsys.readouterr().out)["measurement_count"] == 36


@pytest.mark.parametrize(
    "argv", [["--rows", "0"], ["--width", "-1"], ["--rounds", "0"], ["--dtypes", "bf16", "bf16"]]
)
def test_reject_invalid_plan(argv):
    with pytest.raises(SystemExit):
        bench.parse_args(argv)


@pytest.mark.parametrize("order", list(SwiGLUPackedOrder))
def test_accuracy_gate_and_shared_agreement_are_independent(order):
    reference = partial(NativeSwiGLUOp(), packed_order=order)
    x = torch.ones(1, 2, 8, requires_grad=True)
    upstream = torch.ones(1, 2, 4)
    providers = {name: reference for name in bench.PROVIDERS}
    errors = bench.check_case(providers, x, upstream, order)
    assert errors["forward_accuracy"]["max_abs_error"] == 0
    # Agreement alone must not pass a numerically wrong implementation.
    providers.update(packed=lambda x: reference(x) + 1, adapter=lambda x: reference(x) + 1)
    with pytest.raises(AssertionError):
        bench.check_case(providers, x, upstream, order)
    providers.update(
        packed=lambda x: reference(x) + (x[..., :4] - x[..., :4].detach()),
        adapter=lambda x: reference(x) + (x[..., :4] - x[..., :4].detach()),
    )
    with pytest.raises(AssertionError):
        bench.check_case(providers, x, upstream, order)


def test_partial_report_is_not_complete(tmp_path):
    report = {
        "complete": False,
        "case_plan": bench.case_plan(bench.parse_args(["--rows", "1", "--rounds", "1"])),
        "settings": {"rounds": 1},
        "results": [],
    }
    bench.write_report(report, tmp_path)
    assert not json.loads((tmp_path / "results.json").read_text())["complete"]
    assert "0/54" in (tmp_path / "report.md").read_text()
