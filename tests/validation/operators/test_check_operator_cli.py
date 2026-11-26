# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""User-visible backend observations in the validation CLI."""

import json
import os
import subprocess
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rl_engine.validation.operators import run_operator_suite
from rl_engine.validation.operators.operator_specs import make_candidate, make_operator_case
from tools.validation.operators.check_operator import _summarize

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "op,candidate,actual", [("adaln_modulation", "cuda", "pytorch"), ("logp", "pytorch", "unknown")]
)
def test_cli_text_and_json_report_observed_backend(op, candidate, actual):
    if candidate == "cuda":
        from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp

        try:
            CudaAdaLNModulationOp()
        except RuntimeError:
            if os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1":
                raise
            pytest.skip("compiled CUDA AdaLN extension required")
    command = [
        sys.executable,
        "tools/validation/operators/check_operator.py",
        "--op",
        op,
        "--candidate",
        candidate,
        "--device",
        "cpu",
        "--batch",
        "1",
        "--seq",
        "2",
        "--normalized-dim",
        "2",
        "--vocab",
        "7",
    ]
    results = [
        subprocess.run(
            command + flags,
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=dict(
                os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
            ),
            timeout=30,
        )
        for flags in ([], ["--json"])
    ]
    for result in results:
        assert result.returncode == 0, result.stdout + result.stderr
    output = results[0].stdout
    payload = json.loads(results[1].stdout)
    assert payload["passed"] is True
    assert payload["candidates"][0]["backend"] == candidate
    assert f"requested_backend={candidate}" in output
    assert f"actual_backend={actual}" in output
    assert "passed=True pass_rate=1.0000" in output
    trace = payload["candidates"][0]["cases"][0]["backend_trace"]
    if op == "adaln_modulation":
        assert trace["actual_backend"] == actual and trace["fallback"] is True
        assert "fallback=True" in output
        assert f"fallback_reason={trace['fallback_reason']}" in output
    else:
        assert trace is None
        assert "fallback=unknown" in output


def test_text_reports_each_case_trace_independently(capsys):
    args = Namespace(
        op="adaln_modulation",
        candidate="pytorch",
        arch_key=None,
        batch=1,
        seq=2,
        normalized_dim=2,
        seed=386,
    )
    candidate = make_candidate(args, observe_backend=True)
    case = make_operator_case(args, torch.float32, torch.device("cpu"))
    report = run_operator_suite("adaln_modulation", candidates=[candidate], cases=[case])
    result = report.candidates[0]
    # Formatting consumes report snapshots, including observations from other devices.
    traces = [
        dict(actual_backend="cuda", fallback=False, fallback_reason=None),
        dict(actual_backend="pytorch", fallback=True, fallback_reason="case-domain"),
        None,
        dict(actual_backend=None, fallback=None),
    ]
    cases = [
        replace(result.cases[0], case_name=f"case-{i}", backend_trace=trace)
        for i, trace in enumerate(traces)
    ]
    report = replace(report, candidates=[replace(result, backend="cuda", cases=cases)])
    _summarize(report)
    output = capsys.readouterr().out
    for i, expected in enumerate(
        (
            "actual_backend=cuda fallback=False",
            "actual_backend=pytorch fallback=True",
            "actual_backend=unknown fallback=unknown",
            "actual_backend=unknown fallback=unknown",
        )
    ):
        assert f"case=case-{i} {expected}" in output
    assert "fallback_reason=case-domain" in output


@pytest.mark.parametrize("candidate", ["pytorch", "native"])
@pytest.mark.parametrize("reference_only", [False, True])
def test_timestep_cli_requires_explicit_reference_only_without_qualification(
    candidate, reference_only
):
    command = [
        sys.executable,
        "tools/validation/operators/check_operator.py",
        "--op",
        "timestep_embed_mlp",
        "--candidate",
        candidate,
        "--device",
        "cpu",
        "--batch",
        "1",
        "--seq",
        "1",
        "--json",
        "--check-grad",
    ]
    if reference_only:
        command.append("--reference-only")
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")),
        # The fixed H3072 CPU reference includes forward and five VJPs.
        timeout=90,
    )
    assert result.returncode == (0 if reference_only else 1), result.stdout + result.stderr
    # Some optional native builds print a warning before the structured CLI report.
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    assert payload["passed"] is False
    assert payload["reference_baselines_passed"] is True
    assert payload["validation_scope"] == "reference_baseline"
    assert payload["total_candidates"] == 0 and payload["candidates"] == []
    (baseline,) = payload["reference_baselines"]
    assert baseline["passed"] and baseline["total_outputs"] == 6

    assert baseline["cases"][0]["outputs"][0]["shape"] == [1, 3072]


def test_reference_only_cli_rejects_an_actual_candidate():
    result = subprocess.run(
        [
            sys.executable,
            "tools/validation/operators/check_operator.py",
            "--op",
            "logp",
            "--candidate",
            "pytorch",
            "--device",
            "cpu",
            "--reference-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")),
        timeout=30,
    )
    assert result.returncode != 0
    assert "--reference-only requires a registered reference baseline" in result.stderr
