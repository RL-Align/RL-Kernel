# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import sys
from types import ModuleType

import pytest
import torch

import rl_engine.platforms.device as device_module
from rl_engine.executors.rollout import RolloutExecutor
from rl_engine.kernels.registry import KernelRegistry, OpBackend, kernel_registry
from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger


def test_logger_enhancements():
    logger.info("Testing standard info log.")

    print("Next message should only appear ONCE even with 3 calls:")
    for _ in range(3):
        logger.info_once("This is a unique message that should appear only once.")


def test_device_and_registry():
    logger.info(f"Detected Device: {device_ctx.device_type} (ROCm: {device_ctx.is_rocm})")
    logp_op = kernel_registry.get_op("logp")
    attn_op = kernel_registry.get_op("attn")
    logger.info(f"Retrieved Logp Operator: {logp_op}")
    logger.info(f"Retrieved Attention Operator: {attn_op}")


def test_rocm_attention_uses_flash_attention_by_default(monkeypatch):
    monkeypatch.delenv("RL_KERNEL_ROCM_ATTN_BACKEND", raising=False)

    registry = KernelRegistry()

    assert registry._priority_map["rocm"]["attn"][0] == OpBackend.ROCM_FLASH_ATTN


def test_rocm_attention_native_sdpa_opt_out(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_ROCM_ATTN_BACKEND", " sdpa ")

    registry = KernelRegistry()

    assert registry._priority_map["rocm"]["attn"][0] == OpBackend.PYTORCH_ATTN
    assert registry._priority_map["rocm"]["attn"][1] == OpBackend.ROCM_FLASH_ATTN


class TestMusaPlatform:
    @staticmethod
    def _mock_torch_musa_import(monkeypatch):
        monkeypatch.setattr(
            device_module.importlib,
            "import_module",
            lambda name: object() if name == "torch_musa" else None,
        )

    @staticmethod
    def _mock_musa_device(monkeypatch):
        real_device = device_module.torch.device

        class FakeMusaDevice:
            type = "musa"

        def fake_device(value):
            if value == "musa":
                return FakeMusaDevice()
            return real_device(value)

        monkeypatch.setattr(device_module.torch, "device", fake_device)

    def test_musa_import_failure_falls_back_to_unavailable(self, monkeypatch):
        def failing_import(name):
            if name == "torch_musa":
                raise RuntimeError("incompatible torch_musa installation")

        monkeypatch.setattr(device_module.importlib, "import_module", failing_import)

        assert device_module.is_musa_available() is False

    def test_musa_runtime_failure_falls_back_to_unavailable(self, monkeypatch):
        class FailingMusaRuntime:
            @staticmethod
            def is_available():
                raise RuntimeError("driver failure")

        self._mock_torch_musa_import(monkeypatch)
        monkeypatch.setattr(device_module.torch, "musa", FailingMusaRuntime(), raising=False)

        assert device_module.is_musa_available() is False

    def test_musa_device_context_selects_musa(self, monkeypatch):
        class AvailableMusaRuntime:
            @staticmethod
            def is_available():
                return True

        self._mock_torch_musa_import(monkeypatch)
        self._mock_musa_device(monkeypatch)
        monkeypatch.setattr(device_module.torch, "musa", AvailableMusaRuntime(), raising=False)

        context = device_module.DeviceContext()

        assert context.device_type == "musa"
        assert context.is_musa is True
        assert context.device.type == "musa"

    def test_musa_dispatch_uses_only_pytorch_fallbacks(self, monkeypatch):
        self._mock_musa_device(monkeypatch)
        registry = KernelRegistry()

        assert registry._platform_for_device("musa") == "musa"
        for candidates in registry._priority_map["musa"].values():
            assert all(candidate.name.startswith("PYTORCH_") for candidate in candidates)

    def test_musa_det_gemm_fails_closed(self, monkeypatch):
        self._mock_musa_device(monkeypatch)
        registry = KernelRegistry()

        with pytest.raises(RuntimeError, match="No functional backend"):
            registry.get_op("det_gemm", device="musa")


def test_registry_explicit_device_selects_device_platform(monkeypatch):
    registry = KernelRegistry()
    loaded = []

    class DummyOp:
        pass

    def fake_load_backend(backend):
        loaded.append(backend)
        return DummyOp

    monkeypatch.setattr(registry, "_load_backend", fake_load_backend)

    op = registry.get_op("logp", device="cuda:0")

    assert isinstance(op, DummyOp)
    expected = (
        OpBackend.ROCM_AITER if torch.version.hip is not None else OpBackend.CUDA_FUSED_LOGP_GENERIC
    )
    assert loaded[0] == expected


def test_npu_registry_preserves_per_operator_cpu_fallbacks(monkeypatch):
    registry = KernelRegistry()
    loaded = []

    class DummyOp:
        pass

    def fake_load_backend(backend):
        loaded.append(backend)
        return DummyOp

    monkeypatch.setattr(registry, "_load_backend", fake_load_backend)
    monkeypatch.setattr(device_ctx, "device_type", "npu")

    registry.get_op("rms_norm")

    assert registry._priority_map["npu"].keys() == registry._priority_map["cpu"].keys()
    assert loaded[0] == OpBackend.PYTORCH_NATIVE_RMS_NORM
    assert registry._priority_map["npu"]["batch_invariant_logp"] == [
        OpBackend.ASCEND_BATCH_INVARIANT_LOGP,
        OpBackend.PYTORCH_BATCH_INVARIANT_LOGP,
    ]
    assert registry._priority_map["npu"]["rope"] == [
        OpBackend.ASCEND_ROPE,
        OpBackend.PYTORCH_NATIVE_ROPE,
    ]


def test_npu_available_handles_runtime_failure(monkeypatch):
    class BrokenNPU:
        @staticmethod
        def is_available():
            raise RuntimeError("driver unavailable")

    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(device_module.torch, "npu", BrokenNPU(), raising=False)

    assert device_module._npu_available() is False


def test_executor_flow():
    executor = RolloutExecutor()
    mock_input_ids = torch.ones((1, 16), dtype=torch.long)
    result = executor.execute_rollout(mock_input_ids)
    logger.info(f"Executor result: {result}")


if __name__ == "__main__":
    try:
        test_logger_enhancements()
        test_device_and_registry()
        test_executor_flow()
        print("\n All infrastructure tests passed!")
    except Exception as e:
        print(f"\n Test failed with error: {e}")
