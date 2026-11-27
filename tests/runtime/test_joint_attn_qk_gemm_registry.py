# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Dispatch trace and fallback coverage for the Qwen-Image joint QK GEMM."""

import pytest
import torch

from rl_engine.reference.attention.joint_attn_qk_gemm import NativeJointAttnQkGemmOp
from rl_engine.runtime.registry import KernelRegistry, OpBackend


def test_cpu_dispatch_selects_the_reference_without_a_fallback() -> None:
    operation = KernelRegistry().get_op("joint_attn_qk_gemm", device="cpu")

    assert isinstance(operation, NativeJointAttnQkGemmOp)
    assert operation.provenance["actual_backend"] == operation.backend_id
    assert operation.provenance["backend_enum"] == "PYTORCH_JOINT_ATTN_QK_GEMM"
    assert operation.provenance["fallback"] is False
    assert operation.provenance["prior_rejections"] == []


@pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA CUDA dispatch only")
def test_cuda_unavailable_falls_back_to_the_reference_with_a_trace(monkeypatch) -> None:
    registry = KernelRegistry()
    load_backend = registry._load_backend
    attempted: list[OpBackend] = []

    def load_without_cuda(backend: OpBackend):
        attempted.append(backend)
        if backend is OpBackend.CUDA_JOINT_ATTN_QK_GEMM:
            return None
        return load_backend(backend)

    monkeypatch.setattr(registry, "_load_backend", load_without_cuda)

    operation = registry.get_op("joint_attn_qk_gemm", device="cuda")

    assert isinstance(operation, NativeJointAttnQkGemmOp)
    assert attempted == [
        OpBackend.CUDA_JOINT_ATTN_QK_GEMM,
        OpBackend.PYTORCH_JOINT_ATTN_QK_GEMM,
    ]
    assert operation.provenance["platform"] == "cuda"
    assert operation.provenance["fallback"] is True
    assert operation.provenance["prior_rejections"] == [
        "CUDA_JOINT_ATTN_QK_GEMM: backend could not be loaded or instantiated",
    ]
    assert NativeJointAttnQkGemmOp.provenance["fallback"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_reference_fallback_on_a_gpu_fails_closed(monkeypatch) -> None:
    """The reference is bit-defined only on CPU, so a GPU fallback must raise."""
    registry = KernelRegistry()
    load_backend = registry._load_backend
    monkeypatch.setattr(
        registry,
        "_load_backend",
        lambda backend: (
            None if backend is OpBackend.CUDA_JOINT_ATTN_QK_GEMM else load_backend(backend)
        ),
    )
    operation = registry.get_op("joint_attn_qk_gemm", device="cuda")
    q = torch.randn(1, 2, 8, 128, device="cuda")
    k = torch.randn(1, 2, 9, 128, device="cuda")
    with pytest.raises(RuntimeError, match="correctly-rounded"):
        operation(q, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_dispatch_prefers_the_kernel_when_built() -> None:
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not (_EXT_AVAILABLE and hasattr(_C, "joint_attn_qk_gemm_forward")):
        pytest.skip("native extension not built with joint_attn_qk_gemm")
    operation = KernelRegistry().get_op("joint_attn_qk_gemm", device="cuda")
    assert type(operation).__name__ == "JointAttnQkGemmCudaOp"
    assert operation.provenance["actual_backend"] == operation.backend_id
    assert operation.provenance["fallback"] is False
