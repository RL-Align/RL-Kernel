# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import pytest

from rl_engine.integrations import IntegrationPlan, MegatronIntegration, VllmIntegration


class FakeOperator:
    def __init__(self, value: str, backend_id: str):
        self.value = value
        self.backend_id = backend_id

    def __call__(self, payload: str) -> str:
        return f"{self.value}:{payload}"


def _operators():
    return {
        "attention": FakeOperator("rlk-attn", "rlkernel.attention.deterministic.v1"),
        "ffn": FakeOperator("rlk-ffn", "rlkernel.ffn.qwen3.deterministic.v1"),
        "logp": FakeOperator("rlk-logp", "pytorch-vocab-parallel-logp-ws2"),
    }


def test_plan_uses_module_matrix_cases_without_cartesian_expansion():
    plan = IntegrationPlan.from_case_ids(attention="P/R", ffn="R/P", logp="R/R")

    assert plan.implementation_for("attention", "training").value == "production"
    assert plan.implementation_for("attention", "rollout").value == "rl_kernel"
    assert plan.implementation_for("ffn", "training").value == "rl_kernel"
    assert plan.implementation_for("ffn", "rollout").value == "production"
    assert plan.to_dict()["schema_version"] == "rlkernel.debug_matrix.v1"


def test_megatron_and_vllm_route_the_same_plan_on_opposite_sides():
    plan = IntegrationPlan.from_case_ids(attention="P/R", ffn="R/P", logp="R/R")
    megatron = MegatronIntegration(plan, rl_kernel_operators=_operators())
    vllm = VllmIntegration(plan, rl_kernel_operators=_operators())
    native = FakeOperator("native", "production.backend")

    assert megatron.attention(native, "x") == "native:x"
    assert vllm.attention(native, "x") == "rlk-attn:x"
    assert megatron.ffn(native, "x") == "rlk-ffn:x"
    assert vllm.ffn(native, "x") == "native:x"
    assert megatron.logp(native, "x") == "rlk-logp:x"
    assert vllm.logp(native, "x") == "rlk-logp:x"

    assert megatron.readback()["operators"]["attention"]["backend_id"] == "production.backend"
    assert vllm.readback()["operators"]["attention"]["backend_id"] == (
        "rlkernel.attention.deterministic.v1"
    )


def test_rl_kernel_selection_fails_closed_when_operator_is_missing():
    plan = IntegrationPlan.from_case_ids(attention="R/R")
    integration = MegatronIntegration(plan, rl_kernel_operators={})

    with pytest.raises(RuntimeError, match="no operator was installed"):
        integration.attention(lambda value: value, "x")


def test_integration_modules_do_not_import_optional_frameworks():
    import sys

    assert "megatron" not in sys.modules
    assert "vllm" not in sys.modules
