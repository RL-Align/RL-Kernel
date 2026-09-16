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


@pytest.mark.parametrize(
    "value", ["FLASH_ATTN", "flash-attn", "Flash_Attention", " flash_attn "]
)
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
    monkeypatch.setattr(
        KernelRegistry, "_adjust_priority_for_hardware", fake_hardware_adjustment
    )

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
    assert any(
        "Unknown RL_KERNEL_ROCM_ATTN_BACKEND=unknown" in warning for warning in warnings
    )


def test_sm90_linear_ops_prioritize_cuda_when_extension_symbols_exist(monkeypatch):
    from rl_engine.kernels.ops import base as base_module

    class FakeExtension:
        fused_linear_logp_sm90 = object()
        embedding_sm90_forward = object()
        lm_head_sm90_forward = object()

    monkeypatch.setattr(registry_module.device_ctx, "device_type", "cuda")
    monkeypatch.setattr(
        registry_module.torch.cuda, "get_device_capability", lambda: (9, 0)
    )
    monkeypatch.setattr(registry_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(base_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(base_module, "_C", FakeExtension())

    registry = KernelRegistry()

    assert (
        registry._priority_map["cuda"]["embedding"][0] is OpBackend.CUDA_SM90_EMBEDDING
    )
    assert registry._priority_map["cuda"]["lm_head"][0] is OpBackend.CUDA_SM90_LM_HEAD


def test_sm90_linear_ops_do_not_prioritize_cuda_on_non_hopper(monkeypatch):
    from rl_engine.kernels.ops import base as base_module

    class FakeExtension:
        embedding_sm90_forward = object()
        lm_head_sm90_forward = object()

    monkeypatch.setattr(registry_module.device_ctx, "device_type", "cuda")
    monkeypatch.setattr(
        registry_module.torch.cuda, "get_device_capability", lambda: (8, 0)
    )
    monkeypatch.setattr(registry_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(base_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(base_module, "_C", FakeExtension())

    registry = KernelRegistry()

    assert (
        OpBackend.CUDA_SM90_EMBEDDING not in registry._priority_map["cuda"]["embedding"]
    )
    assert OpBackend.CUDA_SM90_LM_HEAD not in registry._priority_map["cuda"]["lm_head"]


def test_rmsnorm_residual_cuda_priority():
    registry = KernelRegistry()

    assert registry._priority_map["cuda"]["rmsnorm_residual"] == [
        OpBackend.CUDA_RMSNORM_RESIDUAL,
        OpBackend.TRITON_RMSNORM_RESIDUAL,
        OpBackend.PYTORCH_RMSNORM_RESIDUAL,
    ]
    assert registry._priority_map["cpu"]["rmsnorm_residual"] == [
        OpBackend.PYTORCH_RMSNORM_RESIDUAL
    ]


def test_hardware_priority_skips_unavailable_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(registry_module.device_ctx, "device_type", "cuda")
    monkeypatch.setattr(registry_module.torch.cuda, "is_available", lambda: False)

    def probe():
        calls.append("cuda")
        return (9, 0)

    monkeypatch.setattr(registry_module.torch.cuda, "get_device_capability", probe)
    KernelRegistry()
    assert calls == []


@pytest.mark.parametrize(
    ("cuda_ok", "triton_ok", "expected", "probes"),
    [
        (True, True, "RMSNormResidualCudaOp", ["cuda"]),
        (False, True, "RMSNormResidualTritonOp", ["cuda", "triton"]),
        (False, False, "NativeRMSNormResidualOp", ["cuda", "triton"]),
    ],
)
def test_rmsnorm_residual_registry_falls_back_in_order(
    monkeypatch, cuda_ok, triton_ok, expected, probes
):
    import importlib

    cuda_api = importlib.import_module(
        "rl_engine.kernels.ops.cuda.norm.rmsnorm_residual"
    )
    triton_api = importlib.import_module(
        "rl_engine.kernels.ops.triton.rmsnorm_residual_triton"
    )
    calls = []

    def cuda_probe():
        calls.append("cuda")
        if not cuda_ok:
            raise RuntimeError("mock: extension is unavailable")

    def triton_probe():
        calls.append("triton")
        if not triton_ok:
            raise RuntimeError("mock: Triton is unavailable")

    # Only availability probes are mocked; registry loading is real.
    monkeypatch.setattr(registry_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cuda_api, "_extension", cuda_probe)
    monkeypatch.setattr(triton_api, "_require_nvidia_triton", triton_probe)
    registry = KernelRegistry()
    selected = registry.get_op("rmsnorm_residual", device="cuda")

    assert type(selected).__name__ == expected
    assert calls == probes


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_rmsnorm_residual_explicit_candidate_does_not_fallback(monkeypatch, backend):
    import argparse
    import importlib

    from rl_engine.kernels.gtest.operator_specs import make_candidate

    module, probe = {
        "cuda": ("cuda.norm.rmsnorm_residual", "_extension"),
        "triton": ("triton.rmsnorm_residual_triton", "_require_nvidia_triton"),
    }[backend]
    api = importlib.import_module(f"rl_engine.kernels.ops.{module}")

    def unavailable():
        raise RuntimeError("mock: explicit candidate is unavailable")

    monkeypatch.setattr(api, probe, unavailable)
    args = argparse.Namespace(op="rmsnorm_residual", candidate=backend, arch_key=None)
    with pytest.raises(RuntimeError, match="explicit candidate is unavailable"):
        make_candidate(args)
