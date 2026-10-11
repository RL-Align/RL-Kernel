# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

import rl_engine.backends.rocm.ffn.ffn as ffn_module
from rl_engine.backends.rocm.ffn import qwen3_ffn_training
from rl_engine.integrations.engines.rollout.vllm.operators import VllmFFNOperator
from rl_engine.ops.gemm.det_gemm import DetGemmOp, det_gemm_linear_weight_gradient
from rl_engine.runtime.registry import _default_semantic_descriptors

pytestmark = pytest.mark.skipif(
    getattr(torch.version, "hip", None) is None,
    reason="ROCm packed FFN tests require a ROCm PyTorch build",
)


class _FakeDist:
    @staticmethod
    def get_world_size(*, group):
        del group
        return 4


class _FakeCollective:
    _handle = 17
    backend_id = "rocm_ipc_fixed_tree"

    def __init__(self) -> None:
        self.reduced = None

    def all_reduce(self, value, *, out):
        assert out is value
        self.reduced = value
        return out


def test_ffn_descriptor_advertises_rocm_support():
    descriptor = next(
        item
        for item in _default_semantic_descriptors()
        if item.backend_id == "rlkernel.ffn.qwen3.deterministic.v1"
    )
    assert descriptor.supported_devices == frozenset({"cuda", "rocm"})


def test_rocm_prepare_binds_eager_fixed_tree_without_cuda_graph_staging(monkeypatch):
    collective = _FakeCollective()
    monkeypatch.setattr(ffn_module, "_require_parallel_group", lambda *_args: _FakeDist())
    monkeypatch.setattr(ffn_module, "_collective_for_group", lambda *_args, **_kwargs: collective)
    monkeypatch.delenv("RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE", raising=False)

    operator = ffn_module.Qwen3FFNOp()
    handle, world_size = operator.prepare_packed_inference(
        torch.empty((16, 8), dtype=torch.bfloat16),
        torch.empty((8, 8), dtype=torch.bfloat16),
        tp_group=object(),
    )

    assert (handle, world_size) == (17, 4)
    assert operator.packed_inference_backend_id(handle) == "rocm_ipc_fixed_tree"


def test_rocm_packed_forward_reduces_in_place_with_bound_collective(monkeypatch):
    collective = _FakeCollective()
    expected = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    monkeypatch.setattr(
        ffn_module,
        "_qwen3_ffn_packed_inference",
        lambda *_args: expected,
    )

    actual = ffn_module.qwen3_ffn_packed_inference(
        torch.empty((2, 8), dtype=torch.bfloat16),
        torch.empty((16, 8), dtype=torch.bfloat16),
        torch.empty((8, 8), dtype=torch.bfloat16),
        collective_handle=17,
        tp_world_size=4,
        collective=collective,
    )

    assert actual is expected
    assert collective.reduced is expected


def test_rocm_weight_gradient_uses_parameter_layout_and_is_bitwise_stable():
    inputs = torch.randn((16, 8), device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn((16, 12), device="cuda", dtype=torch.bfloat16)

    first = det_gemm_linear_weight_gradient(inputs, grad_output)
    second = det_gemm_linear_weight_gradient(inputs, grad_output)

    assert first.shape == (12, 8)
    assert torch.equal(first, second)
    assert torch.equal(first, torch.mm(grad_output.t(), inputs))


def test_rocm_det_linear_preserves_autograd_and_bitwise_gradients():
    inputs = torch.randn((16, 8), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn((12, 8), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_output = torch.randn((16, 12), device="cuda", dtype=torch.bfloat16)
    op = DetGemmOp()

    first_output = op.linear(inputs, weight)
    assert first_output.requires_grad
    first_output.backward(grad_output)
    first_input_grad = inputs.grad.detach().clone()
    first_weight_grad = weight.grad.detach().clone()

    inputs.grad = None
    weight.grad = None
    second_output = op.linear(inputs, weight)
    second_output.backward(grad_output)

    assert torch.equal(first_output, second_output)
    assert inputs.grad is not None and inputs.grad.shape == inputs.shape
    assert weight.grad is not None and weight.grad.shape == weight.shape
    assert torch.equal(first_input_grad, inputs.grad)
    assert torch.equal(first_weight_grad, weight.grad)


def test_rocm_training_packed_gate_up_matches_separate_projection_bitwise():
    hidden = torch.randn((16, 64), device="cuda", dtype=torch.bfloat16)
    fused_gate_up = torch.randn((256, 64), device="cuda", dtype=torch.bfloat16)
    down = torch.randn((64, 128), device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn_like(hidden)

    def run(*, packed: bool):
        inputs = hidden.detach().clone().requires_grad_(True)
        packed_weight = fused_gate_up.detach().clone().requires_grad_(True)
        gate, up = packed_weight.chunk(2, dim=0)
        down_weight = down.detach().clone().requires_grad_(True)
        output = qwen3_ffn_training(
            inputs,
            gate,
            up,
            down_weight,
            fused_gate_up_weight=packed_weight if packed else None,
            deterministic=True,
        )
        output.backward(grad_output)
        return (
            output.detach(),
            inputs.grad.detach(),
            packed_weight.grad.detach(),
            down_weight.grad.detach(),
        )

    separate = run(packed=False)
    packed = run(packed=True)

    assert all(
        torch.equal(separate_value, packed_value)
        for separate_value, packed_value in zip(separate, packed, strict=True)
    )


def test_rocm_training_backward_matches_reference_bitwise():
    hidden = torch.randn((16, 64), device="cuda", dtype=torch.bfloat16)
    fused_gate_up = torch.randn((256, 64), device="cuda", dtype=torch.bfloat16)
    down = torch.randn((64, 128), device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn_like(hidden)

    def run(ffn):
        inputs = hidden.detach().clone().requires_grad_(True)
        packed_weight = fused_gate_up.detach().clone().requires_grad_(True)
        gate, up = packed_weight.chunk(2, dim=0)
        down_weight = down.detach().clone().requires_grad_(True)
        output = ffn(
            inputs,
            gate,
            up,
            down_weight,
            fused_gate_up_weight=packed_weight,
            deterministic=True,
        )
        output.backward(grad_output)
        return (
            output.detach(),
            inputs.grad.detach(),
            packed_weight.grad.detach(),
            down_weight.grad.detach(),
        )

    reference = run(ffn_module.qwen3_ffn)
    specialized = run(qwen3_ffn_training)

    assert all(
        torch.equal(reference_value, specialized_value)
        for reference_value, specialized_value in zip(reference, specialized, strict=True)
    )


def test_vllm_ffn_provenance_reports_rocm_triton_and_fixed_tree():
    operator = VllmFFNOperator()
    operator._set_runtime_provenance(4, "rocm_ipc_fixed_tree")

    execution = operator.provenance["execution"]
    assert execution["runtime_platform"] == "rocm"
    assert execution["actual_backend"] == "rlkernel.rocm.det_gemm_swiglu"
    assert execution["deterministic_all_reduce_backend"] == "rocm_ipc_fixed_tree"
    assert execution["triton_used"] is True
