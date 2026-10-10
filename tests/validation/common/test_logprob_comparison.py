# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import argparse
import io
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rl_engine.reference.logprob.batch_invariant_logp import NativeBatchInvariantLogpOp
from rl_engine.utils.logger import logger
from rl_engine.validation.common.logprob_comparison import (
    LogprobBackendUnavailable,
    LogprobCandidate,
    LogprobComparisonInputs,
    _device,
    _route_rl_kernel_logs_to_stderr,
    compare_single_gpu_logprob,
    make_logprob_candidate,
)
from rl_engine.validation.operators import run_operator_suite
from rl_engine.validation.operators.operator_specs import make_candidate, make_operator_case


def _inputs() -> LogprobComparisonInputs:
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(2, 4, 257, generator=generator, dtype=torch.float32)
    target_ids = torch.tensor([[3, 5, 7, 11], [13, 17, 19, 23]])
    active = torch.tensor([[False, False, True, True], [False, True, True, True]])
    return LogprobComparisonInputs(logits, target_ids, active_token_mask=active)


def test_single_gpu_pytorch_path_is_bitwise_regression_guard():
    report = compare_single_gpu_logprob(_inputs(), candidates=("pytorch",))

    assert report.reference_name == "pytorch-batch-invariant-logp"
    assert len(report.drifts) == 1
    drift = report.drifts[0]
    assert drift.bitwise_logp
    assert drift.lse.max_abs == 0.0
    assert drift.dlogp.max_abs == 0.0
    assert drift.dlogp.active_count == 5
    assert drift.provenance["requested_backend"] == "pytorch"
    assert drift.provenance["actual_backend"] == "pytorch"
    assert drift.provenance["lse_source"] == "direct"
    assert report.input_provenance["tp_world"] == 1
    assert report.input_provenance["communication"] == "none"


def test_report_serializes_lse_and_active_token_percentiles():
    inputs = _inputs()
    reference = make_logprob_candidate("pytorch")

    def shifted(logits, target_ids, ignore_index):
        logp, lse = reference.fn(logits, target_ids, ignore_index)
        logp = logp.clone()
        logp[0, 0] += 100.0  # inactive and therefore excluded from dlogp
        logp[0, 2] += 1.0
        lse = lse + torch.arange(lse.numel(), dtype=lse.dtype).reshape_as(lse) * 0.1
        return logp, lse

    candidate = LogprobCandidate(
        name="shifted",
        requested_backend="shifted",
        actual_backend="shifted",
        fn=shifted,
    )
    report = compare_single_gpu_logprob(inputs, candidates=(candidate,))
    payload = report.to_dict()
    drift = payload["drifts"][0]

    assert drift["dlogp"]["active_count"] == 5
    assert drift["dlogp"]["max_abs"] == pytest.approx(1.0)
    assert drift["dlogp"]["p95_abs"] == pytest.approx(0.8)
    assert drift["dlogp"]["p99_abs"] == pytest.approx(0.96)
    assert drift["lse"]["active_count"] == 8
    assert drift["lse"]["p99_abs"] == pytest.approx(0.693, abs=1e-5)


def test_canonical_provenance_cannot_be_overridden():
    native = make_logprob_candidate("pytorch")
    candidate = LogprobCandidate(
        name="custom",
        requested_backend="pytorch",
        actual_backend="pytorch",
        fn=native.fn,
        provenance={
            "actual_backend": "fallback",
            "tp_world": 8,
            "communication": "all-gather",
            "lse_source": "reconstructed",
            "implementation": "custom",
        },
    )

    provenance = compare_single_gpu_logprob(_inputs(), candidates=(candidate,)).drifts[0].provenance

    assert provenance["actual_backend"] == "pytorch"
    assert provenance["tp_world"] == 1
    assert provenance["communication"] == "none"
    assert provenance["lse_source"] == "direct"
    assert provenance["implementation"] == "custom"


def test_all_inactive_tokens_produce_zero_dlogp_statistics():
    inputs = _inputs()
    inputs = LogprobComparisonInputs(
        inputs.logits,
        inputs.target_ids,
        active_token_mask=torch.zeros_like(inputs.target_ids, dtype=torch.bool),
    )
    drift = compare_single_gpu_logprob(inputs).drifts[0]

    assert drift.dlogp.active_count == 0
    assert drift.dlogp.max_abs == 0.0
    assert drift.dlogp.p95_abs == 0.0
    assert drift.lse.active_count == inputs.target_ids.numel()


def test_explicit_backend_mismatch_fails_closed():
    native = make_logprob_candidate("pytorch")
    disguised = LogprobCandidate(
        name="fallback",
        requested_backend="cuda-sm90",
        actual_backend="pytorch",
        fn=native.fn,
    )

    with pytest.raises(LogprobBackendUnavailable, match="silent fallback is forbidden"):
        compare_single_gpu_logprob(_inputs(), candidates=(disguised,))


def test_active_ignore_index_is_rejected():
    inputs = _inputs()
    targets = inputs.target_ids.clone()
    targets[0, 2] = -100

    with pytest.raises(ValueError, match="active target_ids cannot equal ignore_index"):
        compare_single_gpu_logprob(
            LogprobComparisonInputs(
                inputs.logits,
                targets,
                active_token_mask=inputs.active_token_mask,
            )
        )


def test_native_diagnostic_lse_satisfies_selected_logit_identity():
    inputs = _inputs()
    candidate = make_logprob_candidate("pytorch")
    effective = inputs.target_ids.masked_fill(~inputs.active_token_mask, -100)
    logp, lse = candidate.fn(inputs.logits, effective, -100)
    production_logp = NativeBatchInvariantLogpOp()(
        inputs.logits, effective, ignore_index=-100, validate=True
    )
    safe_targets = effective.masked_fill(~inputs.active_token_mask, 0)
    selected = torch.gather(inputs.logits, -1, safe_targets.unsqueeze(-1)).squeeze(-1)

    assert torch.equal(logp, production_logp)
    assert torch.equal(logp[inputs.active_token_mask], (selected - lse)[inputs.active_token_mask])


def test_unsupported_backend_name_is_rejected():
    with pytest.raises(ValueError, match="unsupported logprob comparison backend"):
        make_logprob_candidate("unknown")


def test_cli_auto_device_resolves_without_constructing_auto(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert _device("auto") == torch.device("cpu")


def test_cli_routes_rl_kernel_logs_to_stderr_for_machine_readable_stdout(monkeypatch):
    stdout = io.StringIO()
    stderr = io.StringIO()
    original_streams = [
        (handler, handler.stream)
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
    ]
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    try:
        _route_rl_kernel_logs_to_stderr()
        logger.info("test backend diagnostic")
        print(json.dumps({"ok": True}))
    finally:
        for handler, stream in original_streams:
            handler.setStream(stream)

    assert json.loads(stdout.getvalue()) == {"ok": True}
    assert "test backend diagnostic" in stderr.getvalue()


def test_cli_runs_directly_from_testing_module():
    script = (
        Path(__file__).resolve().parents[3] / "rl_engine/validation/common/logprob_comparison.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--candidate",
            "pytorch",
            "--device",
            "cpu",
            "--batch",
            "1",
            "--seq",
            "2",
            "--vocab",
            "17",
            "--prompt-tokens",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["drifts"][0]["provenance"]["actual_backend"] == "pytorch"
    assert payload["input_provenance"]["communication"] == "none"


def test_operator_comparison_specs_register_batch_invariant_logp():
    args = argparse.Namespace(
        op="batch_invariant_logp",
        candidate="pytorch",
        arch_key=None,
        batch=2,
        seq=4,
        vocab=17,
        seed=7,
        input_mode="random",
        constant_value=0.5,
        token_value=3,
        normalized_dim=128,
        k_dim=16,
        n_dim=32,
        theta=1.0e6,
        eps=1.0e-6,
    )

    case = make_operator_case(args, torch.float32, torch.device("cpu"))
    candidate = make_candidate(args)
    report = run_operator_suite("batch_invariant_logp", candidates=[candidate], cases=[case])

    assert report.passed
    assert report.candidates[0].cases[0].op_class == "logprob"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_diagnostic_path_reports_direct_lse():
    try:
        candidate = make_logprob_candidate("triton")
    except LogprobBackendUnavailable as exc:
        pytest.skip(str(exc))
    logits = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)
    targets = torch.tensor([0, 17, 511, 1023], device="cuda")
    try:
        report = compare_single_gpu_logprob(
            LogprobComparisonInputs(logits, targets), candidates=(candidate,)
        )
    except LogprobBackendUnavailable as exc:
        if isinstance(exc.__cause__, PermissionError):
            pytest.skip(str(exc))
        raise

    assert report.drifts[0].provenance["actual_backend"] == "triton"
    assert report.drifts[0].provenance["lse_source"] == "direct"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="Hopper CUDA device required",
)
def test_sm90_diagnostic_path_reports_direct_lse_without_fallback():
    try:
        candidate = make_logprob_candidate("cuda-sm90")
    except LogprobBackendUnavailable as exc:
        pytest.skip(str(exc))
    logits = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)
    targets = torch.tensor([0, 17, 511, 1023], device="cuda")
    report = compare_single_gpu_logprob(
        LogprobComparisonInputs(logits, targets), candidates=(candidate,)
    )

    assert report.drifts[0].provenance["actual_backend"] == "cuda-sm90"
    assert report.drifts[0].provenance["lse_source"] == "direct"
