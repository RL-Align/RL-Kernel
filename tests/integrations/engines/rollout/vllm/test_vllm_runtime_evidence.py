# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch

from rl_engine.integrations.engines.rollout.vllm.adapter import VllmIntegration
from rl_engine.integrations.engines.rollout.vllm.runtime import _record_native_ffn_graph_completion
from rl_engine.runtime.plan import IntegrationPlan


def test_native_ffn_graph_completion_records_production_route(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    integration = VllmIntegration(
        IntegrationPlan.from_case_ids(ffn="P/P"),
        rl_kernel_operators={},
    )

    assert _record_native_ffn_graph_completion(integration)

    record = integration.readback()["operators"]["ffn"]
    assert record["implementation"] == "production"
    assert record["backend_id"] == "vllm.production.ffn"
    assert record["execution_mode"] == "compiled_hip_graph"
    assert record["provenance"] == {
        "runtime_platform": "rocm",
        "execution_boundary": "vllm.sampler_after_model_forward",
        "compiled_model_forward_completed": True,
    }


def test_native_ffn_graph_completion_does_not_mask_strict_route():
    integration = VllmIntegration(
        IntegrationPlan.from_case_ids(ffn="R/R"),
        rl_kernel_operators={},
    )

    assert not _record_native_ffn_graph_completion(integration)
    assert "ffn" not in integration.readback()["operators"]
