# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest

from rl_engine.runtime import registry as registry_module
from rl_engine.runtime.registry import KernelRegistry, OpBackend


def test_rocm_attention_defaults_to_flash_attention(monkeypatch):
    monkeypatch.delenv("RL_KERNEL_ROCM_ATTN_BACKEND", raising=False)

    registry = KernelRegistry()

    assert registry._priority_map["rocm"]["attn"] == [
        OpBackend.ROCM_FLASH_ATTN,
        OpBackend.PYTORCH_ATTN,
        OpBackend.TRITON_GENERIC,
    ]


@pytest.mark.parametrize("value", ["FLASH_ATTN", "flash-attn", "Flash_Attention", " flash_attn "])
def test_rocm_attention_flash_opt_in_aliases(monkeypatch, value):
    monkeypatch.setenv("RL_KERNEL_ROCM_ATTN_BACKEND", value)

    registry = KernelRegistry()

    assert registry._priority_map["rocm"]["attn"] == [
        OpBackend.ROCM_FLASH_ATTN,
        OpBackend.PYTORCH_ATTN,
        OpBackend.TRITON_GENERIC,
    ]


@pytest.mark.parametrize("value", ["native", "PYTORCH", " sdpa "])
def test_rocm_attention_can_opt_out_to_sdpa(monkeypatch, value):
    monkeypatch.setenv("RL_KERNEL_ROCM_ATTN_BACKEND", value)

    registry = KernelRegistry()

    assert registry._priority_map["rocm"]["attn"] == [
        OpBackend.PYTORCH_ATTN,
        OpBackend.ROCM_FLASH_ATTN,
        OpBackend.TRITON_GENERIC,
    ]


def test_rocm_attention_env_override_wins_after_hardware_adjustment(monkeypatch):
    def fake_hardware_adjustment(registry):
        registry._priority_map["rocm"]["attn"] = [
            OpBackend.PYTORCH_ATTN,
            OpBackend.ROCM_FLASH_ATTN,
            OpBackend.TRITON_GENERIC,
        ]

    monkeypatch.setenv("RL_KERNEL_ROCM_ATTN_BACKEND", "flash_attn")
    monkeypatch.setattr(KernelRegistry, "_adjust_priority_for_hardware", fake_hardware_adjustment)

    registry = KernelRegistry()

    assert registry._priority_map["rocm"]["attn"] == [
        OpBackend.ROCM_FLASH_ATTN,
        OpBackend.PYTORCH_ATTN,
        OpBackend.TRITON_GENERIC,
    ]


def test_rocm_attention_unknown_env_value_uses_default_and_warns(monkeypatch):
    warnings = []

    def fake_warning(message, *args):
        warnings.append(message % args)

    monkeypatch.setenv("RL_KERNEL_ROCM_ATTN_BACKEND", "unknown")
    monkeypatch.setattr(registry_module.logger, "warning", fake_warning)

    registry = KernelRegistry()

    assert registry._priority_map["rocm"]["attn"] == [
        OpBackend.ROCM_FLASH_ATTN,
        OpBackend.PYTORCH_ATTN,
        OpBackend.TRITON_GENERIC,
    ]
    assert any("Unknown RL_KERNEL_ROCM_ATTN_BACKEND=unknown" in warning for warning in warnings)


def test_sm90_linear_ops_prioritize_cuda_when_extension_symbols_exist(monkeypatch):
    from rl_engine.backends import extension as base_module

    class FakeExtension:
        fused_linear_logp_sm90 = object()
        embedding_sm90_forward = object()
        lm_head_sm90_forward = object()

    monkeypatch.setattr(registry_module.device_ctx, "device_type", "cuda")
    monkeypatch.setattr(registry_module.torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(base_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(base_module, "_C", FakeExtension())

    registry = KernelRegistry()

    assert registry._priority_map["cuda"]["embedding"][0] is OpBackend.CUDA_SM90_EMBEDDING
    assert registry._priority_map["cuda"]["lm_head"][0] is OpBackend.CUDA_SM90_LM_HEAD


def test_sm90_linear_ops_do_not_prioritize_cuda_on_non_hopper(monkeypatch):
    from rl_engine.backends import extension as base_module

    class FakeExtension:
        embedding_sm90_forward = object()
        lm_head_sm90_forward = object()

    monkeypatch.setattr(registry_module.device_ctx, "device_type", "cuda")
    monkeypatch.setattr(registry_module.torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(base_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(base_module, "_C", FakeExtension())

    registry = KernelRegistry()

    assert OpBackend.CUDA_SM90_EMBEDDING not in registry._priority_map["cuda"]["embedding"]
    assert OpBackend.CUDA_SM90_LM_HEAD not in registry._priority_map["cuda"]["lm_head"]
