# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Dispatch trace and fallback coverage for joint-attention softmax."""

import pytest
import torch

from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
from rl_engine.kernels.registry import KernelRegistry, OpBackend
from rl_engine.testing.bitwise import tensor_bytes_equal


@pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA CUDA dispatch only")
def test_cuda_and_triton_unavailable_falls_back_to_pytorch(monkeypatch) -> None:
    registry = KernelRegistry()
    load_backend = registry._load_backend
    attempted: list[OpBackend] = []

    def load_without_optimized_backends(backend: OpBackend):
        attempted.append(backend)
        if backend in {
            OpBackend.CUDA_JOINT_ATTN_SOFTMAX,
            OpBackend.TRITON_JOINT_ATTN_SOFTMAX,
        }:
            return None
        return load_backend(backend)

    monkeypatch.setattr(registry, "_load_backend", load_without_optimized_backends)

    operation = registry.get_op("joint_attn_softmax", device="cuda")

    assert isinstance(operation, NativeJointAttnSoftmaxOp)
    assert attempted == [
        OpBackend.CUDA_JOINT_ATTN_SOFTMAX,
        OpBackend.TRITON_JOINT_ATTN_SOFTMAX,
        OpBackend.PYTORCH_JOINT_ATTN_SOFTMAX,
    ]

    assert operation.provenance["actual_backend"] == operation.backend_id
    assert operation.provenance["backend_enum"] == "PYTORCH_JOINT_ATTN_SOFTMAX"
    assert operation.provenance["platform"] == "cuda"
    assert operation.provenance["fallback"] is True
    assert operation.provenance["prior_rejections"] == [
        "CUDA_JOINT_ATTN_SOFTMAX: backend could not be loaded or instantiated",
        "TRITON_JOINT_ATTN_SOFTMAX: backend could not be loaded or instantiated",
    ]

    # The same cached backend selected directly on CPU must not inherit the
    # earlier CUDA selection's fallback trace.
    cpu_operation = registry.get_op("joint_attn_softmax", device="cpu")
    assert cpu_operation is not operation
    assert cpu_operation.provenance["fallback"] is False
    assert cpu_operation.provenance["prior_rejections"] == []
    assert operation.provenance["fallback"] is True
    assert NativeJointAttnSoftmaxOp.provenance["fallback"] is False


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA is required for the Triton fallback",
)
def test_cuda_unavailable_selects_triton_with_trace(monkeypatch) -> None:
    pytest.importorskip("triton")
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    registry = KernelRegistry()
    load_backend = registry._load_backend

    def load_without_cuda(backend: OpBackend):
        if backend == OpBackend.CUDA_JOINT_ATTN_SOFTMAX:
            return None
        return load_backend(backend)

    monkeypatch.setattr(registry, "_load_backend", load_without_cuda)

    operation = registry.get_op("joint_attn_softmax", device="cuda")

    assert isinstance(operation, TritonJointAttnSoftmaxOp)
    assert operation.provenance["selected_backend"] == "triton_cuda"
    assert operation.provenance["actual_backend"] == operation.backend_id
    assert operation.provenance["backend_enum"] == "TRITON_JOINT_ATTN_SOFTMAX"
    assert operation.provenance["fallback"] is True
    assert operation.provenance["prior_rejections"] == [
        "CUDA_JOINT_ATTN_SOFTMAX: backend could not be loaded or instantiated"
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA is required for GPU fallback execution",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("output_fp32", [False, True])
def test_selected_pytorch_fallback_on_cuda_matches_reference_bytes(
    monkeypatch, dtype: torch.dtype, output_fp32: bool
) -> None:
    registry = KernelRegistry()
    load_backend = registry._load_backend

    def load_without_optimized_backends(backend: OpBackend):
        if backend in {
            OpBackend.CUDA_JOINT_ATTN_SOFTMAX,
            OpBackend.TRITON_JOINT_ATTN_SOFTMAX,
        }:
            return None
        return load_backend(backend)

    monkeypatch.setattr(registry, "_load_backend", load_without_optimized_backends)
    operation = registry.get_op("joint_attn_softmax", device="cuda")

    assert isinstance(operation, NativeJointAttnSoftmaxOp)
    assert operation.provenance["actual_backend"] == operation.backend_id
    assert operation.provenance["fallback"] is True
    assert operation.provenance["platform"] == "cuda"

    scores = torch.linspace(-4.0, 4.0, 513, dtype=torch.float32).reshape(1, 513).to(dtype)
    scores[:, :256] = float("-inf")
    scores[:, -1] = float("-inf")
    output_dtype = torch.float32 if output_fp32 else dtype
    upstream = torch.linspace(-1.0, 1.0, 513, dtype=output_dtype).reshape(1, 513)

    def run_forward_backward(op, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        input_scores = scores.to(device).requires_grad_(True)
        probabilities = op.forward_fp32(input_scores) if output_fp32 else op.forward(input_scores)
        (grad_scores,) = torch.autograd.grad(probabilities, input_scores, upstream.to(device))
        return probabilities, grad_scores

    expected_probabilities, expected_grad_scores = run_forward_backward(
        NativeJointAttnSoftmaxOp(), "cpu"
    )
    probabilities, grad_scores = run_forward_backward(operation, "cuda")

    assert probabilities.dtype == output_dtype
    assert grad_scores.dtype == dtype
    assert tensor_bytes_equal(probabilities, expected_probabilities)
    assert tensor_bytes_equal(grad_scores, expected_grad_scores)

    # A source checkout can run this fallback test without the compiled extension.
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp

    try:
        native_cuda = JointAttnSoftmaxCudaOp()
    except RuntimeError as exc:
        if "CUDA symbols are unavailable" not in str(exc):
            raise
    else:
        cuda_probabilities, cuda_grad_scores = run_forward_backward(native_cuda, "cuda")
        assert tensor_bytes_equal(probabilities, cuda_probabilities)
        assert tensor_bytes_equal(grad_scores, cuda_grad_scores)
