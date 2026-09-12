# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from examples.vime_rocm_attention_ablation.validate_module_artifacts import (
    RL_KERNEL_MISMATCH_SIDECAR_MARKER,
    validate_module_readbacks,
)


def _production_record(
    framework: str,
    module: str,
    *,
    call_count: int = 1,
) -> dict:
    return {
        "case_id": "P/P",
        "implementation": "production",
        "backend_id": f"{framework}.production.{module}",
        "call_count": call_count,
        "provenance": {
            "runtime_platform": "rocm",
            "fallback": False,
        },
    }


def _production_readbacks(*, rollout_ffn_calls: int = 1) -> list[dict]:
    cases = {module: {"case_id": "P/P"} for module in ("attention", "ffn", "logp")}
    return [
        {
            "framework": "megatron",
            "target": "training",
            "plan": {"cases": cases},
            "installed_hooks": {
                "attention": "test.attention",
                "ffn": "test.ffn",
            },
            "operators": {
                module: _production_record("megatron", module) for module in ("attention", "ffn")
            },
            "fallbacks": [],
        },
        {
            "framework": "vllm",
            "target": "rollout",
            "plan": {"cases": cases},
            "installed_hooks": {
                module: f"test.{module}" for module in ("attention", "ffn", "logp")
            },
            "operators": {
                "attention": _production_record("vllm", "attention"),
                "ffn": _production_record(
                    "vllm",
                    "ffn",
                    call_count=rollout_ffn_calls,
                ),
                "logp": _production_record("vllm", "logp"),
            },
            "fallbacks": [],
        },
    ]


def test_rocm_module_validator_accepts_native_production_routes():
    report = validate_module_readbacks(
        _production_readbacks(),
        {
            "attention": "P/P",
            "ffn": "P/P",
            "logp": "P/P",
        },
        log_text=f"{RL_KERNEL_MISMATCH_SIDECAR_MARKER}P/P",
    )

    assert report["passed"]


def test_rocm_module_validator_reports_graph_capture_observability_gap():
    report = validate_module_readbacks(
        _production_readbacks(rollout_ffn_calls=0),
        {
            "attention": "P/P",
            "ffn": "P/P",
            "logp": "P/P",
        },
        log_text=f"{RL_KERNEL_MISMATCH_SIDECAR_MARKER}P/P",
    )

    assert not report["passed"]
    assert report["errors"] == ["vllm/rollout ffn had zero calls"]
