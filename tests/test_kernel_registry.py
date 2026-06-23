# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest

from rl_engine.kernels import registry as registry_module
from rl_engine.kernels.registry import KernelRegistry, OpBackend


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
