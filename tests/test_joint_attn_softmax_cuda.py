# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA acceptance tests for joint-attention softmax (issue #386)."""

import pytest
import torch

from rl_engine.testing.bitwise import tensor_bytes_equal

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA is required",
)


def test_forward_fp32_uniform_row_returns_uniform_probabilities():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp

    scores = torch.tensor([[0.0, 0.0]], device="cuda", dtype=torch.float32)

    probabilities = JointAttnSoftmaxCudaOp().forward_fp32(scores)

    expected = torch.tensor([[0.5, 0.5]], device="cuda", dtype=torch.float32)
    assert probabilities.dtype == torch.float32
    assert probabilities.shape == scores.shape
    assert tensor_bytes_equal(probabilities, expected)


def test_forward_fp32_matches_reference_across_first_tile_boundary():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp

    scores_cpu = torch.linspace(-10.0, 10.0, 257, dtype=torch.float32).unsqueeze(0)
    expected = NativeJointAttnSoftmaxOp().forward_fp32(scores_cpu)

    probabilities = JointAttnSoftmaxCudaOp().forward_fp32(scores_cpu.cuda()).cpu()

    assert tensor_bytes_equal(probabilities, expected)


@pytest.mark.parametrize("key_length", [1, 255, 256, 257, 513, 1024])
def test_forward_fp32_matches_cpu_reference_byte_for_byte(key_length):
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp

    generator = torch.Generator().manual_seed(386 + key_length)
    scores_cpu = torch.randn(2, key_length, dtype=torch.float32, generator=generator) * 30.0
    expected = NativeJointAttnSoftmaxOp().forward_fp32(scores_cpu)

    probabilities = JointAttnSoftmaxCudaOp().forward_fp32(scores_cpu.cuda()).cpu()

    assert tensor_bytes_equal(probabilities, expected)


def test_forward_fp32_accepts_bf16_input_and_matches_cpu_reference_byte_for_byte():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp

    scores_cpu = torch.linspace(-10.0, 10.0, 513, dtype=torch.bfloat16).reshape(1, 513)
    expected = NativeJointAttnSoftmaxOp().forward_fp32(scores_cpu)

    probabilities = JointAttnSoftmaxCudaOp().forward_fp32(scores_cpu.cuda()).cpu()

    assert probabilities.dtype == torch.float32
    assert tensor_bytes_equal(probabilities, expected)


def test_forward_bf16_casts_once_at_final_cuda_write():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp

    scores_cpu = torch.linspace(-10.0, 10.0, 513, dtype=torch.bfloat16).reshape(1, 513)
    expected = NativeJointAttnSoftmaxOp().forward(scores_cpu)

    probabilities = JointAttnSoftmaxCudaOp().forward(scores_cpu.cuda()).cpu()

    assert probabilities.dtype == torch.bfloat16
    assert tensor_bytes_equal(probabilities, expected)


@pytest.mark.parametrize(
    "dtype,output_fp32",
    [(torch.float32, True), (torch.bfloat16, False), (torch.bfloat16, True)],
)
def test_fully_masked_row_returns_zero_probabilities_and_gradients(dtype, output_fp32):
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp

    valid_row = torch.linspace(-4.0, 4.0, 513, dtype=dtype)
    masked_row = torch.full((513,), float("-inf"), dtype=dtype)
    scores = torch.stack([valid_row, masked_row])
    upstream = torch.linspace(-1.0, 1.0, 513, dtype=torch.float32).repeat(2, 1)
    if not output_fp32:
        upstream = upstream.to(dtype)

    expected_scores = scores.clone().requires_grad_(True)
    reference = NativeJointAttnSoftmaxOp()
    expected_probabilities = (
        reference.forward_fp32(expected_scores) if output_fp32 else reference(expected_scores)
    )
    expected_probabilities.backward(upstream)

    actual_scores = scores.cuda().requires_grad_(True)
    operation = JointAttnSoftmaxCudaOp()
    actual_probabilities = (
        operation.forward_fp32(actual_scores) if output_fp32 else operation(actual_scores)
    )
    actual_probabilities.backward(upstream.cuda())

    assert tensor_bytes_equal(actual_probabilities.cpu(), expected_probabilities)
    assert tensor_bytes_equal(actual_scores.grad.cpu(), expected_scores.grad)
    assert tensor_bytes_equal(
        actual_probabilities[1].cpu(), torch.zeros_like(expected_probabilities[1])
    )
    assert tensor_bytes_equal(
        actual_scores.grad[1].cpu(), torch.zeros_like(expected_scores.grad[1])
    )


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf")], ids=["nan", "posinf"])
def test_unsupported_nonfinite_scores_propagate_nan(invalid_score):
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp

    scores = torch.tensor(
        [[invalid_score, float("-inf")]],
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    probabilities = JointAttnSoftmaxCudaOp().forward_fp32(scores)
    probabilities.backward(torch.ones_like(probabilities))

    assert torch.isnan(probabilities).any()
    assert torch.isnan(scores.grad).any()


@pytest.mark.parametrize("leading_masked_tiles", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("output_fp32", [False, True])
def test_leading_fully_masked_tiles_keep_later_key_valid(leading_masked_tiles, dtype, output_fp32):
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp

    key_length = leading_masked_tiles * 256 + 1
    scores = torch.full((1, key_length), float("-inf"), device="cuda", dtype=dtype)
    scores[0, -1] = 0.0
    scores.requires_grad_(True)

    operation = JointAttnSoftmaxCudaOp()
    probabilities = operation.forward_fp32(scores) if output_fp32 else operation.forward(scores)
    assert probabilities.dtype == (torch.float32 if output_fp32 else dtype)
    assert torch.isfinite(probabilities).all()
    assert probabilities[0, -1].item() == 1.0
    assert torch.count_nonzero(probabilities[0, :-1]) == 0

    upstream = torch.linspace(-1.0, 1.0, key_length, device="cuda", dtype=probabilities.dtype)
    probabilities.backward(upstream.unsqueeze(0))
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad) == 0


def test_forward_fp32_matches_reference_for_multiple_rows_and_tiles():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp

    generator = torch.Generator().manual_seed(386)
    scores_cpu = torch.randn(3, 777, dtype=torch.float32, generator=generator) * 8.0
    expected = NativeJointAttnSoftmaxOp().forward_fp32(scores_cpu)

    probabilities = JointAttnSoftmaxCudaOp().forward_fp32(scores_cpu.cuda()).cpu()

    assert tensor_bytes_equal(probabilities, expected)


def test_forward_fp32_is_byte_invariant_to_batch_companions():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp

    generator = torch.Generator(device="cuda").manual_seed(386)
    rows = torch.randn(3, 513, device="cuda", dtype=torch.float32, generator=generator)
    operation = JointAttnSoftmaxCudaOp()

    alone = operation.forward_fp32(rows[1:2])
    first_in_batch = operation.forward_fp32(rows[[1, 0, 2]])[0:1]
    last_in_batch = operation.forward_fp32(rows[[2, 0, 1]])[2:3]

    assert tensor_bytes_equal(alone, first_in_batch)
    assert tensor_bytes_equal(alone, last_in_batch)


def test_masked_prompt_padding_is_batch_invariant():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp

    target = torch.linspace(-5.0, 5.0, 769, device="cuda", dtype=torch.float32)
    # The first 512 keys are text positions: 73 valid tokens, then padding.
    target[73:512] = float("-inf")
    upstream = torch.linspace(1.0, -1.0, 769, device="cuda", dtype=torch.float32)
    operation = JointAttnSoftmaxCudaOp()

    alone_scores = target.clone().requires_grad_(True)
    alone_probabilities = operation.forward_fp32(alone_scores)
    alone_probabilities.backward(upstream)

    batched_scores = torch.stack([target.flip(0), target, torch.zeros_like(target)])
    batched_scores.requires_grad_(True)
    batched_upstream = torch.stack([torch.zeros_like(upstream), upstream, upstream.flip(0)])
    batched_probabilities = operation.forward_fp32(batched_scores)
    batched_probabilities.backward(batched_upstream)

    assert tensor_bytes_equal(batched_probabilities[1], alone_probabilities)
    assert tensor_bytes_equal(batched_scores.grad[1], alone_scores.grad)


@pytest.mark.parametrize(
    "dtype,output_fp32",
    [(torch.float32, True), (torch.bfloat16, False), (torch.bfloat16, True)],
)
def test_fully_masked_tail_tile_preserves_valid_key_bytes(dtype, output_fp32):
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp

    scores = torch.linspace(-4.0, 4.0, 256, device="cuda", dtype=torch.float32).to(dtype)
    upstream = torch.linspace(-1.0, 1.0, 256, device="cuda", dtype=torch.float32)
    if not output_fp32:
        upstream = upstream.to(dtype)

    short_scores = scores.clone().requires_grad_(True)
    long_scores = torch.cat([scores, scores.new_full((1,), float("-inf"))]).requires_grad_(True)
    op = JointAttnSoftmaxCudaOp()

    short_probabilities = op.forward_fp32(short_scores) if output_fp32 else op(short_scores)
    long_probabilities = op.forward_fp32(long_scores) if output_fp32 else op(long_scores)
    short_probabilities.backward(upstream)
    long_probabilities.backward(torch.cat([upstream, upstream.new_zeros(1)]))

    assert tensor_bytes_equal(long_probabilities[:256], short_probabilities)
    assert tensor_bytes_equal(long_scores.grad[:256], short_scores.grad)
    assert long_probabilities[-1] == 0
    assert long_scores.grad[-1] == 0


def test_backward_fp32_matches_cpu_reference_byte_for_byte():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp

    generator = torch.Generator().manual_seed(386)
    scores = torch.randn(2, 513, dtype=torch.float32, generator=generator)
    upstream = torch.randn(2, 513, dtype=torch.float32, generator=generator)

    expected_scores = scores.clone().requires_grad_(True)
    NativeJointAttnSoftmaxOp().forward_fp32(expected_scores).backward(upstream)

    actual_scores = scores.cuda().requires_grad_(True)
    JointAttnSoftmaxCudaOp().forward_fp32(actual_scores).backward(upstream.cuda())

    assert tensor_bytes_equal(actual_scores.grad.cpu(), expected_scores.grad)


def test_backward_bf16_matches_cpu_reference_byte_for_byte():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp

    scores = torch.linspace(-10.0, 10.0, 513, dtype=torch.bfloat16).reshape(1, 513)
    upstream = torch.linspace(1.0, -1.0, 513, dtype=torch.bfloat16).reshape(1, 513)

    expected_scores = scores.clone().requires_grad_(True)
    expected_probabilities = NativeJointAttnSoftmaxOp().forward(expected_scores)
    expected_probabilities.backward(upstream)

    actual_scores = scores.cuda().requires_grad_(True)
    actual_probabilities = JointAttnSoftmaxCudaOp().forward(actual_scores)
    actual_probabilities.backward(upstream.cuda())

    assert actual_probabilities.dtype == torch.bfloat16
    assert tensor_bytes_equal(actual_probabilities.cpu(), expected_probabilities)
    assert actual_scores.grad.dtype == torch.bfloat16
    assert tensor_bytes_equal(actual_scores.grad.cpu(), expected_scores.grad)


def test_forward_fp32_with_bf16_input_returns_bf16_gradient():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp

    scores = torch.linspace(-8.0, 8.0, 257, dtype=torch.bfloat16).reshape(1, 257)
    upstream = torch.linspace(1.0, -1.0, 257, dtype=torch.float32).reshape(1, 257)

    expected_scores = scores.clone().requires_grad_(True)
    NativeJointAttnSoftmaxOp().forward_fp32(expected_scores).backward(upstream)

    actual_scores = scores.cuda().requires_grad_(True)
    JointAttnSoftmaxCudaOp().forward_fp32(actual_scores).backward(upstream.cuda())

    assert actual_scores.grad.dtype == torch.bfloat16
    assert tensor_bytes_equal(actual_scores.grad.cpu(), expected_scores.grad)


def test_backward_fp32_is_byte_invariant_to_batch_companions():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp

    target = torch.linspace(-5.0, 5.0, 513, device="cuda", dtype=torch.float32)
    upstream = torch.linspace(1.0, -1.0, 513, device="cuda", dtype=torch.float32)
    operation = JointAttnSoftmaxCudaOp()

    alone_scores = target.clone().requires_grad_(True)
    operation.forward_fp32(alone_scores).backward(upstream)

    batched_scores = torch.stack([target.flip(0), target, torch.zeros_like(target)])
    batched_scores.requires_grad_(True)
    batched_upstream = torch.stack([torch.zeros_like(upstream), upstream, upstream.flip(0)])
    operation.forward_fp32(batched_scores).backward(batched_upstream)

    assert tensor_bytes_equal(batched_scores.grad[1], alone_scores.grad)


def test_registry_selects_cuda_backend_on_cuda():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.registry import kernel_registry

    operation = kernel_registry.get_op("joint_attn_softmax", device="cuda")

    assert isinstance(operation, JointAttnSoftmaxCudaOp)


def test_cuda_trace_records_the_frozen_arithmetic_contract():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp

    operation = JointAttnSoftmaxCudaOp()

    assert operation.provenance == {
        "selected_backend": "cuda",
        "reduction_order": "tile256_tree_128_to_1_then_left_to_right",
        "accumulator_precision": "fp32",
        "split_k": False,
        "stream_k": False,
        "tf32": False,
        "kernel_fingerprint": "joint-attn-softmax-v2-logical-mask-tile256-exp7",
        "fallback": False,
    }


def test_forward_without_autograd_uses_the_state_free_cuda_kernel(monkeypatch):
    from rl_engine.kernels.ops.base import _C
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp

    original_forward = _C.joint_attn_softmax_forward
    original_forward_with_state = _C.joint_attn_softmax_forward_with_state
    calls = {"forward": 0, "forward_with_state": 0}

    def record_forward(scores):
        calls["forward"] += 1
        return original_forward(scores)

    def record_forward_with_state(scores):
        calls["forward_with_state"] += 1
        return original_forward_with_state(scores)

    monkeypatch.setattr(_C, "joint_attn_softmax_forward", record_forward)
    monkeypatch.setattr(
        _C,
        "joint_attn_softmax_forward_with_state",
        record_forward_with_state,
    )
    scores = torch.linspace(-5.0, 5.0, 513, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        probabilities = JointAttnSoftmaxCudaOp().forward(scores)

    assert probabilities.dtype == torch.bfloat16
    assert calls == {"forward": 1, "forward_with_state": 0}
