# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the joint-attention softmax operator (issue #386)."""

import pytest
import torch

from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
from rl_engine.testing.bitwise import tensor_bytes_equal


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_byte_comparator_distinguishes_signed_zero(dtype):
    positive_zero = torch.tensor([0.0], dtype=dtype)
    negative_zero = torch.tensor([-0.0], dtype=dtype)

    assert torch.equal(positive_zero, negative_zero)
    assert not tensor_bytes_equal(positive_zero, negative_zero)
    assert tensor_bytes_equal(positive_zero, positive_zero.clone())
    assert not tensor_bytes_equal(positive_zero, positive_zero.reshape(1, 1))
    assert not tensor_bytes_equal(positive_zero, positive_zero.to(torch.float64))


def test_forward_fp32_uniform_row_returns_uniform_probabilities():
    scores = torch.tensor([[0.0, 0.0]])

    probabilities = NativeJointAttnSoftmaxOp().forward_fp32(scores)

    expected = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    assert probabilities.dtype == torch.float32
    assert probabilities.shape == scores.shape
    assert tensor_bytes_equal(probabilities, expected)


def test_forward_fp32_is_stable_for_large_scores():
    scores = torch.tensor([[1000.0, 1001.0, 1002.0]], dtype=torch.float32)

    probabilities = NativeJointAttnSoftmaxOp().forward_fp32(scores)

    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities, torch.softmax(scores, dim=-1), rtol=1e-6, atol=1e-7)


def test_forward_fp32_matches_fixed_tile_order_across_boundary():
    """Pin the frozen 256-key online/tree order shared by every backend."""
    scores = torch.linspace(-10.0, 10.0, 257, dtype=torch.float32).unsqueeze(0)

    probabilities = NativeJointAttnSoftmaxOp().forward_fp32(scores)

    sample_columns = torch.tensor([0, 64, 128, 192, 255, 256])
    expected_fp32_bits = torch.tensor(
        [791302132, 851802413, 912586570, 973389200, 1032738776, 1033496798],
        dtype=torch.int32,
    )
    actual_fp32_bits = probabilities[0, sample_columns].contiguous().view(torch.int32)
    assert tensor_bytes_equal(actual_fp32_bits, expected_fp32_bits)


def test_bf16_forward_casts_once_after_fp32_computation():
    scores = torch.linspace(-4.0, 4.0, 257, dtype=torch.bfloat16).reshape(1, 1, 257)
    op = NativeJointAttnSoftmaxOp()

    probabilities = op(scores)
    expected = op.forward_fp32(scores).to(torch.bfloat16)

    assert probabilities.dtype == torch.bfloat16
    assert tensor_bytes_equal(probabilities, expected)


def test_backward_matches_fixed_tree_order():
    scores = torch.linspace(-10.0, 10.0, 257, dtype=torch.float32, requires_grad=True)
    upstream = torch.linspace(1.0, -1.0, 257, dtype=torch.float32)

    NativeJointAttnSoftmaxOp().forward_fp32(scores).backward(upstream)

    sample_columns = torch.tensor([0, 64, 128, 192, 255, 256])
    expected_fp32_bits = torch.tensor(
        [799154177, 856333485, 911143872, 961965710, -1144443673, -1142111523],
        dtype=torch.int32,
    )
    actual_fp32_bits = scores.grad[sample_columns].contiguous().view(torch.int32)
    assert tensor_bytes_equal(actual_fp32_bits, expected_fp32_bits)


def test_rejects_empty_key_sequence():
    scores = torch.empty(2, 0, dtype=torch.float32)

    with pytest.raises(ValueError, match="key dimension must be non-empty"):
        NativeJointAttnSoftmaxOp().forward_fp32(scores)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_rejects_dtype_outside_bf16_fp32(dtype):
    scores = torch.zeros(2, 4, dtype=dtype)

    with pytest.raises(TypeError, match="BF16 or FP32"):
        NativeJointAttnSoftmaxOp().forward_fp32(scores)


@pytest.mark.parametrize("key_length", [1, 255, 256, 257, 513])
def test_forward_fp32_matches_softmax_definition(key_length):
    generator = torch.Generator().manual_seed(386 + key_length)
    scores = torch.randn(2, 3, key_length, generator=generator, dtype=torch.float32)

    probabilities = NativeJointAttnSoftmaxOp().forward_fp32(scores)
    expected = torch.softmax(scores, dim=-1)

    assert probabilities.shape == scores.shape
    assert probabilities.dtype == torch.float32
    torch.testing.assert_close(probabilities, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities[..., 0]),
        rtol=1e-6,
        atol=1e-7,
    )


def test_forward_is_batch_invariant():
    target = torch.linspace(-5.0, 5.0, 257, dtype=torch.float32)
    companions = torch.stack([target.flip(0), torch.zeros_like(target)])
    op = NativeJointAttnSoftmaxOp()

    alone = op.forward_fp32(target.unsqueeze(0))[0]
    in_batch = op.forward_fp32(torch.cat([companions[:1], target[None], companions[1:]]))[1]

    assert tensor_bytes_equal(in_batch, alone)


@pytest.mark.parametrize(
    "dtype,output_fp32",
    [(torch.float32, True), (torch.bfloat16, False), (torch.bfloat16, True)],
)
def test_fully_masked_tail_tile_preserves_valid_key_bytes(dtype, output_fp32):
    scores = torch.linspace(-4.0, 4.0, 256, dtype=torch.float32).to(dtype)
    upstream = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)
    if not output_fp32:
        upstream = upstream.to(dtype)

    short_scores = scores.clone().requires_grad_(True)
    long_scores = torch.cat([scores, scores.new_full((1,), float("-inf"))]).requires_grad_(True)
    op = NativeJointAttnSoftmaxOp()

    short_probabilities = op.forward_fp32(short_scores) if output_fp32 else op(short_scores)
    long_probabilities = op.forward_fp32(long_scores) if output_fp32 else op(long_scores)
    short_probabilities.backward(upstream)
    long_probabilities.backward(torch.cat([upstream, upstream.new_zeros(1)]))

    assert tensor_bytes_equal(long_probabilities[:256], short_probabilities)
    assert tensor_bytes_equal(long_scores.grad[:256], short_scores.grad)
    assert long_probabilities[-1] == 0
    assert long_scores.grad[-1] == 0


@pytest.mark.parametrize("key_length,masked_tile_count", [(513, 1), (513, 2), (769, 2)])
@pytest.mark.parametrize(
    "dtype,output_fp32",
    [(torch.float32, True), (torch.bfloat16, False), (torch.bfloat16, True)],
)
def test_negative_infinity_mask_can_cover_leading_tiles(
    key_length, masked_tile_count, dtype, output_fp32
):
    scores = torch.linspace(-4.0, 4.0, key_length, dtype=torch.float32).to(dtype)
    masked_key_count = masked_tile_count * 256
    scores[:masked_key_count] = float("-inf")
    if key_length > masked_key_count + 1:
        scores[-1] = float("-inf")
    scores.requires_grad_(True)
    upstream = torch.linspace(-1.0, 1.0, key_length, dtype=torch.float32)
    if not output_fp32:
        upstream = upstream.to(dtype)

    op = NativeJointAttnSoftmaxOp()
    probabilities = op.forward_fp32(scores) if output_fp32 else op(scores)
    probabilities.backward(upstream)

    expected_scores = scores.detach().float().requires_grad_(True)
    expected_probabilities = torch.softmax(expected_scores, dim=-1)
    expected_probabilities.backward(upstream.float())
    expected_probabilities = expected_probabilities.to(probabilities.dtype)
    expected_grad_scores = expected_scores.grad.to(dtype)

    assert probabilities.dtype == (torch.float32 if output_fp32 else dtype)
    assert scores.grad.dtype == dtype
    assert torch.isfinite(probabilities).all()
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(probabilities[:masked_key_count]) == 0
    assert torch.count_nonzero(scores.grad[:masked_key_count]) == 0
    if key_length > masked_key_count + 1:
        assert probabilities[-1] == 0
        assert scores.grad[-1] == 0
    rtol, atol = (5e-3, 5e-4) if dtype == torch.bfloat16 else (1e-6, 1e-7)
    torch.testing.assert_close(
        probabilities.float(), expected_probabilities.float(), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        scores.grad.float(), expected_grad_scores.float(), rtol=rtol, atol=atol
    )


def test_backward_matches_softmax_derivative():
    generator = torch.Generator().manual_seed(386)
    scores = torch.randn(2, 257, generator=generator, dtype=torch.float32)
    upstream = torch.randn(2, 257, generator=generator, dtype=torch.float32)

    actual_scores = scores.clone().requires_grad_(True)
    NativeJointAttnSoftmaxOp().forward_fp32(actual_scores).backward(upstream)

    expected_scores = scores.clone().requires_grad_(True)
    torch.softmax(expected_scores, dim=-1).backward(upstream)

    torch.testing.assert_close(actual_scores.grad, expected_scores.grad, rtol=2e-6, atol=1e-7)


def test_backward_is_batch_invariant():
    target = torch.linspace(-5.0, 5.0, 257, dtype=torch.float32)
    upstream = torch.linspace(1.0, -1.0, 257, dtype=torch.float32)
    op = NativeJointAttnSoftmaxOp()

    alone_scores = target.clone().requires_grad_(True)
    op.forward_fp32(alone_scores).backward(upstream)

    batched_scores = torch.stack([target.flip(0), target, torch.zeros_like(target)])
    batched_scores.requires_grad_(True)
    batched_upstream = torch.stack([torch.zeros_like(upstream), upstream, upstream.flip(0)])
    op.forward_fp32(batched_scores).backward(batched_upstream)

    assert tensor_bytes_equal(batched_scores.grad[1], alone_scores.grad)


def test_bf16_backward_returns_bf16_gradient_from_fp32_reference():
    scores = torch.linspace(-4.0, 4.0, 257, dtype=torch.bfloat16).requires_grad_(True)
    upstream = torch.linspace(1.0, -1.0, 257, dtype=torch.bfloat16)

    NativeJointAttnSoftmaxOp()(scores).backward(upstream)

    fp32_scores = scores.detach().float().requires_grad_(True)
    NativeJointAttnSoftmaxOp().forward_fp32(fp32_scores).backward(upstream.float())

    assert scores.grad.dtype == torch.bfloat16
    assert tensor_bytes_equal(scores.grad, fp32_scores.grad.to(torch.bfloat16))


def test_empty_batch_preserves_shape_in_forward_and_backward():
    scores = torch.empty(0, 257, dtype=torch.float32, requires_grad=True)

    probabilities = NativeJointAttnSoftmaxOp().forward_fp32(scores)
    probabilities.sum().backward()

    assert probabilities.shape == scores.shape
    assert scores.grad is not None
    assert scores.grad.shape == scores.shape


def test_registry_selects_native_reference_on_cpu():
    from rl_engine.kernels.registry import kernel_registry

    operation = kernel_registry.get_op("joint_attn_softmax", device="cpu")

    assert isinstance(operation, NativeJointAttnSoftmaxOp)


def test_native_trace_records_the_frozen_arithmetic_contract():
    operation = NativeJointAttnSoftmaxOp()

    assert operation.provenance == {
        "selected_backend": "pytorch_reference",
        "reduction_order": "tile256_tree_128_to_1_then_left_to_right",
        "accumulator_precision": "fp32",
        "split_k": False,
        "stream_k": False,
        "tf32": False,
        "kernel_fingerprint": "joint-attn-softmax-v1-tile256-exp7",
        "fallback": False,
    }
