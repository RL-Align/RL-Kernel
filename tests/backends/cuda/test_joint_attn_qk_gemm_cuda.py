# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA backend for the Qwen-Image joint attention QK GEMM (issue #386).

Every device result is compared with the fp32 CPU gold byte for byte, forward
and backward, with no tolerance path.
"""

from __future__ import annotations

import math

import pytest
import torch

from rl_engine.backends.cuda.attention.joint_attn_qk_gemm import cuda_geometry_support
from rl_engine.reference.attention.joint_attn_qk_gemm import (
    JOINT_ATTN_QK_GEMM_CONTRACT_VERSION,
    NativeJointAttnQkGemmOp,
    joint_attn_qk_gemm_backward_with_leaves,
    joint_attn_qk_gemm_reference_backward,
)
from rl_engine.validation.common.bitwise import tensor_bytes_equal as bitwise_equal

HEAD_DIM = 128
QWENIMAGE_HEADS = 24
SCALE = 1.0 / math.sqrt(HEAD_DIM)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _cuda_op():
    """The CUDA backend, or a skip when the native extension is not built."""
    from rl_engine.backends.cuda.attention.joint_attn_qk_gemm import JointAttnQkGemmCudaOp

    try:
        return JointAttnQkGemmCudaOp()
    except RuntimeError as exc:  # pragma: no cover - depends on the build
        pytest.skip(f"joint_attn_qk_gemm CUDA kernel unavailable: {exc}")


@requires_cuda
class TestCudaForward:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "shape",
        [
            (1, QWENIMAGE_HEADS, 1, 1),
            (1, QWENIMAGE_HEADS, 5, 7),
            (2, QWENIMAGE_HEADS, 64, 64),
            (3, 8, 33, 129),
            (2, 4, 7, 300),
        ],
        ids=["singleton", "tiny", "tile-aligned", "ragged", "wide"],
    )
    def test_matches_the_fp32_cpu_gold_byte_for_byte(self, dtype, shape):
        cuda_op = _cuda_op()
        gold = NativeJointAttnQkGemmOp()
        torch.manual_seed(13)
        batch, heads, queries, keys = shape
        q = torch.randn(batch, heads, queries, HEAD_DIM, dtype=dtype)
        k = torch.randn(batch, heads, keys, HEAD_DIM, dtype=dtype)

        assert bitwise_equal(
            cuda_op.forward_fp32(q.cuda(), k.cuda(), scale=SCALE).cpu(),
            gold.forward_fp32(q, k, scale=SCALE),
        )
        assert bitwise_equal(
            cuda_op.forward(q.cuda(), k.cuda(), scale=SCALE).cpu(),
            gold.forward(q, k, scale=SCALE),
        )

    def test_subnormal_scores_match_the_gold(self):
        """The kernel keeps subnormals; the gold refuses to run when it would flush them."""
        cuda_op = _cuda_op()
        q = torch.full((1, 1, 1, 32), 2.0**-70)
        assert bitwise_equal(
            cuda_op.forward_fp32(q.cuda(), q.cuda(), scale=1.0).cpu(),
            NativeJointAttnQkGemmOp().forward_fp32(q, q, scale=1.0),
        )

    @pytest.mark.parametrize("batch", [2, 4, 16, 300])
    def test_batch_and_position_invariance(self, batch):
        cuda_op = _cuda_op()
        torch.manual_seed(14)
        base_q = torch.randn(1, 4, 9, HEAD_DIM, device="cuda")
        base_k = torch.randn(1, 4, 11, HEAD_DIM, device="cuda")
        alone = cuda_op.forward_fp32(base_q, base_k, scale=SCALE)

        many_q = torch.randn(batch, 4, 9, HEAD_DIM, device="cuda")
        many_k = torch.randn(batch, 4, 11, HEAD_DIM, device="cuda")
        position = batch - 1
        many_q[position] = base_q[0]
        many_k[position] = base_k[0]
        batched = cuda_op.forward_fp32(many_q, many_k, scale=SCALE)
        assert bitwise_equal(batched[position : position + 1], alone)

    def test_appended_key_padding_does_not_move_the_valid_block(self):
        cuda_op = _cuda_op()
        torch.manual_seed(15)
        q = torch.randn(1, 4, 9, HEAD_DIM, device="cuda")
        k = torch.randn(1, 4, 11, HEAD_DIM, device="cuda")
        clean = cuda_op.forward_fp32(q, k, scale=SCALE)
        padded_k = torch.cat([k, torch.randn(1, 4, 37, HEAD_DIM, device="cuda")], dim=-2)
        padded = cuda_op.forward_fp32(q, padded_k, scale=SCALE)
        assert bitwise_equal(padded[..., :11].contiguous(), clean)

    def test_launch_geometry_does_not_change_bits(self):
        """Growing the key extent re-tiles the grid; the shared block must not move."""
        cuda_op = _cuda_op()
        torch.manual_seed(16)
        q = torch.randn(1, 2, 8, HEAD_DIM, device="cuda")
        k = torch.randn(1, 2, 8, HEAD_DIM, device="cuda")
        reference = cuda_op.forward_fp32(q, k, scale=SCALE)
        for extra in (1, 24, 57, 96):
            wider_k = torch.cat([k, torch.randn(1, 2, extra, HEAD_DIM, device="cuda")], dim=-2)
            wider = cuda_op.forward_fp32(q, wider_k, scale=SCALE)
            assert bitwise_equal(wider[..., :8].contiguous(), reference), extra

    @pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
    def test_supported_head_dims_match_the_gold(self, head_dim):
        cuda_op = _cuda_op()
        gold = NativeJointAttnQkGemmOp()
        torch.manual_seed(17)
        q = torch.randn(1, 2, 5, head_dim)
        k = torch.randn(1, 2, 7, head_dim)
        scale = 1.0 / math.sqrt(head_dim)
        assert bitwise_equal(
            cuda_op.forward_fp32(q.cuda(), k.cuda(), scale=scale).cpu(),
            gold.forward_fp32(q, k, scale=scale),
        )

    @pytest.mark.parametrize("head_dim", [96, 100])
    def test_unsupported_head_dim_is_a_declared_fallback(self, head_dim):
        cuda_op = _cuda_op()
        torch.manual_seed(18)
        q = torch.randn(1, 2, 5, head_dim, device="cuda")
        k = torch.randn(1, 2, 7, head_dim, device="cuda")
        out = cuda_op.forward(q, k, scale=1.0 / math.sqrt(head_dim))
        assert out.shape == (1, 2, 5, 7)
        plan = cuda_op.last_plan
        assert plan is not None and plan.fallback is True
        assert plan.fallback_reason and str(head_dim) in plan.fallback_reason
        assert cuda_op.trace()["fallback"] is True
        assert cuda_op.trace()["actual_backend"] == NativeJointAttnQkGemmOp.backend_id

    @pytest.mark.parametrize("head_dim", [96, 100])
    def test_fallback_bytes_still_equal_the_gold(self, head_dim):
        """The declared fallback runs the reference on CPU, so bits stay gold-equal."""
        cuda_op = _cuda_op()
        gold = NativeJointAttnQkGemmOp()
        torch.manual_seed(25)
        q = torch.randn(1, 2, 5, head_dim)
        k = torch.randn(1, 2, 7, head_dim)
        scale = 1.0 / math.sqrt(head_dim)
        assert bitwise_equal(
            cuda_op.forward(q.cuda(), k.cuda(), scale=scale).cpu(),
            gold.forward(q, k, scale=scale),
        )
        assert bitwise_equal(
            cuda_op.forward_fp32(q.cuda(), k.cuda(), scale=scale).cpu(),
            gold.forward_fp32(q, k, scale=scale),
        )

    def test_fallback_gradients_still_flow(self):
        """Autograd must survive the CPU round-trip inside the fallback."""
        cuda_op = _cuda_op()
        torch.manual_seed(26)
        q = torch.randn(1, 2, 5, 96, device="cuda", requires_grad=True)
        k = torch.randn(1, 2, 7, 96, device="cuda", requires_grad=True)
        scores = cuda_op.forward(q, k, scale=1.0 / math.sqrt(96))
        scores.backward(torch.randn_like(scores))
        assert q.grad is not None and k.grad is not None
        assert q.grad.device.type == "cuda" and k.grad.device.type == "cuda"
        assert bool(torch.isfinite(q.grad).all()) and bool(torch.isfinite(k.grad).all())

    def test_supported_call_records_no_fallback(self):
        cuda_op = _cuda_op()
        q = torch.randn(1, 2, 4, HEAD_DIM, device="cuda")
        k = torch.randn(1, 2, 4, HEAD_DIM, device="cuda")
        cuda_op.forward(q, k, scale=SCALE)
        plan = cuda_op.last_plan
        assert plan is not None
        assert plan.fallback is False
        assert plan.backend_id == cuda_op.backend_id
        assert plan.contract_version == JOINT_ATTN_QK_GEMM_CONTRACT_VERSION
        assert cuda_op.trace()["actual_backend"] == cuda_op.backend_id
        assert cuda_op.trace()["fallback"] is False


@requires_cuda
class TestCudaBackward:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "shape",
        [(1, QWENIMAGE_HEADS, 9, 13), (2, 8, 70, 70), (1, 4, 33, 217)],
        ids=["one-leaf", "three-leaves", "seven-leaves"],
    )
    def test_matches_the_reference_byte_for_byte(self, dtype, shape):
        cuda_op = _cuda_op()
        torch.manual_seed(19)
        batch, heads, queries, keys = shape
        q = torch.randn(batch, heads, queries, HEAD_DIM, dtype=dtype)
        k = torch.randn(batch, heads, keys, HEAD_DIM, dtype=dtype)
        grad = torch.randn(batch, heads, queries, keys, dtype=dtype)

        q_cuda = q.cuda().requires_grad_()
        k_cuda = k.cuda().requires_grad_()
        cuda_op.forward(q_cuda, k_cuda, scale=SCALE).backward(grad.cuda())

        expected_q, expected_k = joint_attn_qk_gemm_reference_backward(q, k, grad, scale=SCALE)
        assert bitwise_equal(q_cuda.grad, expected_q)
        assert bitwise_equal(k_cuda.grad, expected_k)

    def test_masked_backward_matches_the_reference(self):
        cuda_op = _cuda_op()
        torch.manual_seed(20)
        batch, heads, queries, keys = 2, 4, 9, 13
        q = torch.randn(batch, heads, queries, HEAD_DIM)
        k = torch.randn(batch, heads, keys, HEAD_DIM)
        grad = torch.randn(batch, heads, queries, keys)
        key_mask = torch.ones(batch, keys, dtype=torch.bool)
        key_mask[:, 9:] = False
        query_mask = torch.ones(batch, queries, dtype=torch.bool)
        query_mask[:, 7:] = False

        q_cuda = q.cuda().requires_grad_()
        k_cuda = k.cuda().requires_grad_()
        cuda_op.forward(
            q_cuda,
            k_cuda,
            scale=SCALE,
            query_padding_mask=query_mask.cuda(),
            key_padding_mask=key_mask.cuda(),
        ).backward(grad.cuda())

        expected_q, expected_k = joint_attn_qk_gemm_reference_backward(
            q, k, grad, scale=SCALE, query_padding_mask=query_mask, key_padding_mask=key_mask
        )
        assert bitwise_equal(q_cuda.grad, expected_q)
        assert bitwise_equal(k_cuda.grad, expected_k)

    def test_device_leaves_merge_through_the_shared_host_tree(self):
        """The CUDA backward must reuse the reference's merge code, not clone it."""
        _cuda_op()  # skips cleanly when the extension is missing
        from rl_engine.backends.extension import _C

        torch.manual_seed(21)
        q = torch.randn(1, 3, 40, HEAD_DIM)
        k = torch.randn(1, 3, 70, HEAD_DIM)
        grad = torch.randn(1, 3, 40, 70)

        device_q, device_k = joint_attn_qk_gemm_backward_with_leaves(
            q.cuda(),
            k.cuda(),
            grad.cuda(),
            scale=SCALE,
            dq_leaf_fn=_C.joint_attn_qk_gemm_dq_leaf,
            dk_leaf_fn=_C.joint_attn_qk_gemm_dk_leaf,
        )
        host_q, host_k = joint_attn_qk_gemm_reference_backward(q, k, grad, scale=SCALE)
        assert bitwise_equal(device_q, host_q)
        assert bitwise_equal(device_k, host_k)

    @pytest.mark.parametrize("batch", [2, 4, 16])
    def test_gradient_batch_and_position_invariance(self, batch):
        cuda_op = _cuda_op()
        torch.manual_seed(22)
        q = torch.randn(1, 3, 6, HEAD_DIM, device="cuda")
        k = torch.randn(1, 3, 40, HEAD_DIM, device="cuda")
        grad = torch.randn(1, 3, 6, 40, device="cuda")

        alone_q = q.clone().requires_grad_()
        alone_k = k.clone().requires_grad_()
        cuda_op.forward(alone_q, alone_k, scale=SCALE).backward(grad)

        many_q = torch.randn(batch, 3, 6, HEAD_DIM, device="cuda")
        many_k = torch.randn(batch, 3, 40, HEAD_DIM, device="cuda")
        many_g = torch.randn(batch, 3, 6, 40, device="cuda")
        position = batch - 1
        many_q[position], many_k[position], many_g[position] = q[0], k[0], grad[0]
        many_q = many_q.requires_grad_()
        many_k = many_k.requires_grad_()
        cuda_op.forward(many_q, many_k, scale=SCALE).backward(many_g)

        assert bitwise_equal(many_q.grad[position : position + 1], alone_q.grad)
        assert bitwise_equal(many_k.grad[position : position + 1], alone_k.grad)


@requires_cuda
class TestCudaInputs:
    def test_strided_inputs_match_contiguous_inputs(self):
        """The [B, S, H, D].transpose(1, 2) layout from RoPE must not change bits."""
        cuda_op = _cuda_op()
        torch.manual_seed(27)
        x = torch.randn(1, 70, 4, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        y = torch.randn(1, 70, 4, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        grad = torch.randn(1, 4, 70, 70, dtype=torch.bfloat16, device="cuda")

        results = []
        for contiguous in (False, True):
            q = x.transpose(1, 2)
            k = y.transpose(1, 2)
            if contiguous:
                q, k = q.contiguous(), k.contiguous()
            q = q.detach().requires_grad_()
            k = k.detach().requires_grad_()
            scores = cuda_op.forward(q, k, scale=SCALE)
            scores.backward(grad)
            results.append((scores, q.grad, k.grad))
        for strided, dense in zip(*results):
            assert bitwise_equal(strided, dense)

    def test_backward_skips_an_unneeded_gradient(self):
        cuda_op = _cuda_op()
        torch.manual_seed(28)
        q = torch.randn(1, 2, 9, HEAD_DIM, device="cuda")
        k = torch.randn(1, 2, 40, HEAD_DIM, device="cuda")
        grad = torch.randn(1, 2, 9, 40, device="cuda")

        only_q = q.clone().requires_grad_()
        cuda_op.forward(only_q, k, scale=SCALE).backward(grad)
        both_q = q.clone().requires_grad_()
        both_k = k.clone().requires_grad_()
        cuda_op.forward(both_q, both_k, scale=SCALE).backward(grad)
        assert bitwise_equal(only_q.grad, both_q.grad)

    def test_mask_on_another_device_is_rejected(self):
        cuda_op = _cuda_op()
        q = torch.randn(1, 2, 8, HEAD_DIM, device="cuda")
        k = torch.randn(1, 2, 9, HEAD_DIM, device="cuda")
        with pytest.raises(ValueError, match="key_padding_mask must be on"):
            cuda_op.forward(q, k, key_padding_mask=torch.ones(1, 9, dtype=torch.bool))


def test_geometry_rejects_an_empty_head_dim():
    supported, reason = cuda_geometry_support(0)
    assert not supported and reason


@requires_cuda
class TestLaunchLimits:
    """Grid extents past the CUDA gridDim.y/z limit raise instead of launching."""

    def test_forward_rejects_batch_head_product_past_grid_limit(self):
        _cuda_op()
        from rl_engine.backends.extension import _C

        q = torch.zeros(1, 65536, 1, 32, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(RuntimeError, match="exceeds the CUDA limit"):
            _C.joint_attn_qk_gemm_forward(q, q, 1.0, False)

    def test_dq_leaf_rejects_query_extent_past_grid_limit(self):
        _cuda_op()
        from rl_engine.backends.extension import _C

        grad = torch.zeros(1, 1, 65536, 1, dtype=torch.float32, device="cuda")
        k = torch.zeros(1, 1, 1, 32, dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError, match="exceeds the CUDA limit"):
            _C.joint_attn_qk_gemm_dq_leaf(grad, k, 0, 1)

    def test_dk_leaf_rejects_key_extent_past_grid_limit(self):
        _cuda_op()
        from rl_engine.backends.extension import _C

        grad = torch.zeros(1, 1, 1, 65536, dtype=torch.float32, device="cuda")
        q = torch.zeros(1, 1, 1, 32, dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError, match="exceeds the CUDA limit"):
            _C.joint_attn_qk_gemm_dk_leaf(grad, q, 0, 1)

    def test_leaf_rejects_an_operand_with_fewer_batches_or_heads(self):
        _cuda_op()
        from rl_engine.backends.extension import _C

        grad = torch.zeros(2, 1, 1, 32, dtype=torch.float32, device="cuda")
        k = torch.zeros(1, 1, 32, HEAD_DIM, dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError, match="batch and head extents"):
            _C.joint_attn_qk_gemm_dq_leaf(grad, k, 0, 1)
        q = torch.zeros(2, 2, 1, HEAD_DIM, dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError, match="batch and head extents"):
            _C.joint_attn_qk_gemm_dk_leaf(grad, q, 0, 1)

    def test_leaf_rejects_a_non_fp32_operand(self):
        _cuda_op()
        from rl_engine.backends.extension import _C

        grad = torch.zeros(1, 1, 1, 32, dtype=torch.float32, device="cuda")
        k = torch.zeros(1, 1, 32, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(RuntimeError, match="must be fp32"):
            _C.joint_attn_qk_gemm_dq_leaf(grad, k, 0, 1)

    def test_empty_batch_returns_an_empty_result(self):
        cuda_op = _cuda_op()
        q = torch.randn(0, 2, 4, HEAD_DIM, device="cuda")
        out = cuda_op.forward(q, q, scale=SCALE)
        assert out.shape == (0, 2, 4, 4)
