# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp
from rl_engine.kernels.gtest.op_checks import CandidateSpec, OperatorCase, run_operator_suite


def _logp_case(name: str, dtype: torch.dtype, *, seed: int = 0) -> OperatorCase:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 8, 257, dtype=dtype, generator=generator)
    token_ids = torch.randint(0, logits.size(-1), (2, 8), generator=generator)
    return OperatorCase(
        name=name,
        op_class="logprob",
        dtype=dtype,
        inputs={"logits": logits, "token_ids": token_ids},
        gold_fn=NativeLogpOp().forward_fp32,
    )


def test_logp_native_candidate_suite_passes():
    report = run_operator_suite(
        "logp",
        candidates=[CandidateSpec(name="native-logp", backend="pytorch", fn=NativeLogpOp())],
        cases=[
            _logp_case("fp32", torch.float32, seed=1),
            _logp_case("bf16", torch.bfloat16, seed=2),
            _logp_case("fp16", torch.float16, seed=3),
        ],
    )

    assert report.passed
    assert report.pass_rate == 1.0
    assert report.candidates[0].passed_outputs == 3
    assert all(case.passed for case in report.candidates[0].cases)


def test_logp_registry_candidate_suite_passes_on_cpu():
    from rl_engine.kernels.registry import kernel_registry

    report = run_operator_suite(
        "logp",
        candidates=[
            CandidateSpec(
                name="registry-logp",
                backend="registry",
                fn=kernel_registry.get_op("logp"),
            )
        ],
        cases=[_logp_case("fp32", torch.float32, seed=4)],
    )

    assert report.passed
    assert report.candidates[0].candidate_name == "registry-logp"


def test_suite_reports_failure_for_bad_candidate():
    def bad_logp(logits, token_ids):
        del token_ids
        return torch.zeros(logits.shape[:-1], dtype=logits.dtype)

    report = run_operator_suite(
        "logp",
        candidates=[CandidateSpec(name="bad-logp", backend="test", fn=bad_logp)],
        cases=[_logp_case("fp32", torch.float32, seed=5)],
    )

    output = report.candidates[0].cases[0].outputs[0]
    assert not report.passed
    assert report.pass_rate == 0.0
    assert output.max_abs_error > 0.0


def test_suite_report_to_dict_contains_error_metrics():
    report = run_operator_suite(
        "logp",
        candidates=[CandidateSpec(name="native-logp", backend="pytorch", fn=NativeLogpOp())],
        cases=[_logp_case("fp32", torch.float32, seed=6)],
    )

    data = report.to_dict()
    output = data["candidates"][0]["cases"][0]["outputs"][0]
    assert data["suite_name"] == "logp"
    assert "max_abs_error" in output
    assert "atol" in output
    assert "passed" in output


def test_candidate_arch_key_uses_tolerance_override():
    def slightly_shifted_logp(logits, token_ids):
        return NativeLogpOp().forward_fp32(logits, token_ids) + 0.02

    contract = {
        "accuracy": {
            "default": {
                "logprob": {
                    "float32": {"atol": 1.0e-5, "rtol": 0.0},
                }
            },
            "arch_overrides": {
                "testarch": {
                    "logprob": {
                        "float32": {"atol": 5.0e-2, "rtol": 0.0},
                    }
                }
            },
        }
    }
    report = run_operator_suite(
        "logp",
        candidates=[
            CandidateSpec(
                name="shifted-logp",
                backend="test",
                fn=slightly_shifted_logp,
                arch_key="testarch",
            )
        ],
        cases=[_logp_case("fp32", torch.float32, seed=7)],
        contract=contract,
    )

    output = report.candidates[0].cases[0].outputs[0]
    assert report.passed
    assert output.atol == 5.0e-2
