# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for colocated dual-engine setup.

These tests validate the sleep/wake lifecycle and weight bridge integration
WITHOUT requiring a full multi-GPU run. GPU tests are marked and can be
skipped on CPU-only CI.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Unit tests (no GPU required)
# ---------------------------------------------------------------------------


class TestStepResult:
    def test_step_result_fields(self):
        from examples.colocated_dual_engine import StepResult

        r = StepResult(step=0, phase="rollout", duration_ms=123.4, metrics={"n": 10})
        assert r.step == 0
        assert r.phase == "rollout"
        assert r.duration_ms == 123.4
        assert r.metrics["n"] == 10


class TestColocatedOrchestratorInit:
    def _make_args(self, **overrides):
        defaults = dict(
            model="test-model", num_gpus=8, steps=5,
            prompts_per_step=4, samples_per_prompt=4,
            max_prompt_len=256, max_completion_len=64,
            lr=5e-6, lora_rank=16,
            vllm_sleep_level=2, vllm_gpu_memory_utilization=0.40,
            output_log=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_init_creates_orchestrator(self):
        from examples.colocated_dual_engine import ColocatedOrchestrator

        args = self._make_args(steps=3, num_gpus=4)
        orch = ColocatedOrchestrator(args)
        assert orch.args.steps == 3
        assert orch.args.num_gpus == 4
        assert orch.llm is None
        assert orch.worker_proc is None
        assert orch.results == []


class TestInferenceAdapter:
    def test_create_vllm_adapter(self):
        from rl_engine.executors.inference_adapter import create_inference_adapter
        adapter = create_inference_adapter("vllm", model="test", num_gpus=1)
        assert adapter.supports_sleep() is True

    def test_create_sglang_adapter(self):
        from rl_engine.executors.inference_adapter import create_inference_adapter
        adapter = create_inference_adapter("sglang", model="test", num_gpus=1)
        assert adapter.supports_sleep() is False

    def test_invalid_backend_raises(self):
        from rl_engine.executors.inference_adapter import create_inference_adapter
        with pytest.raises(ValueError, match="Unknown inference backend"):
            create_inference_adapter("invalid")

    def test_protocol_conformance(self):
        from rl_engine.executors.inference_adapter import (
            InferenceEngineAdapter, create_inference_adapter,
        )
        for backend in ("vllm", "sglang"):
            adapter = create_inference_adapter(backend, model="test", num_gpus=1)
            assert isinstance(adapter, InferenceEngineAdapter)


class TestWeightBridgeIntegration:
    def test_local_clone_bridge_publish_import(self):
        from rl_engine.executors.bridge import make_weight_bridge
        bridge = make_weight_bridge("local-clone")
        model = torch.nn.Linear(8, 4)
        manifest = bridge.publish(model, weight_version=1)
        assert manifest.weight_version == 1
        assert len(manifest.tensors) > 0
        imported = dict(bridge.import_update(manifest))
        assert len(imported) > 0
        bridge.acknowledge(manifest.update_id)
        bridge.release(manifest.update_id)

    def test_shared_memory_bridge_publish_import(self):
        from rl_engine.executors.bridge import make_weight_bridge
        bridge = make_weight_bridge("shared-memory")
        model = torch.nn.Linear(8, 4)
        manifest = bridge.publish(model, weight_version=1)
        imported = dict(bridge.import_update(manifest))
        assert len(imported) > 0
        bridge.acknowledge(manifest.update_id)
        bridge.release(manifest.update_id)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_vmm_bridge_publish_import(self):
        from rl_engine.executors.bridge import make_weight_bridge
        bridge = make_weight_bridge("cuda-vmm")
        model = torch.nn.Linear(8, 4).cuda()
        manifest = bridge.publish(model, weight_version=1)
        imported = dict(bridge.import_update(manifest))
        assert len(imported) > 0
        bridge.acknowledge(manifest.update_id)
        bridge.release(manifest.update_id)


# ---------------------------------------------------------------------------
# Integration tests (GPU required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestVLLMSleepWake:
    @pytest.fixture(autouse=True)
    def _check_vllm(self):
        try:
            import vllm
        except ImportError:
            pytest.skip("vLLM not installed")
        if not hasattr(vllm.LLM, "sleep"):
            pytest.skip("vLLM version does not support sleep mode")

    def test_sleep_wake_cycle(self):
        from vllm import LLM
        initial_mem = torch.cuda.memory_allocated(0)
        llm = LLM(
            model="facebook/opt-125m",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.3,
            enforce_eager=True,
            enable_sleep_mode=True,
            max_model_len=128,
        )
        loaded_mem = torch.cuda.memory_allocated(0)
        assert loaded_mem > initial_mem
        llm.sleep(level=2)
        sleep_mem = torch.cuda.memory_allocated(0)
        assert sleep_mem < loaded_mem
        llm.wake_up()
        wake_mem = torch.cuda.memory_allocated(0)
        assert wake_mem >= sleep_mem
        del llm


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
