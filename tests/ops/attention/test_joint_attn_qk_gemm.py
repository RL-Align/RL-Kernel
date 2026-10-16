# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Qwen-Image WS1 joint attention QK GEMM (issue #386).

Two orthogonal properties are asserted throughout, both bitwise:

* **Accuracy.** Every backend reproduces the fp32 CPU gold byte for byte, in
  fp32 and bf16, forward and backward. This operator is a reduction, but the
  reduction tree is *frozen and shared* rather than left to each backend, so
  byte equality is achievable and there are no tolerance paths anywhere in this
  file: no ``allclose``, no ``atol``/``rtol``, no ULP thresholds.
* **Invariance.** A logical row's bytes do not change with batch size, batch
  position, prompt padding, or launch geometry.

The file also carries an **independent FMA oracle**: an exact-rational
implementation of a correctly-rounded FP32 fused multiply-add, used to verify
the reference's ``torch.addcmul`` primitive from first principles rather than
by assumption, and to pin the measured CPU/CUDA divergence of that primitive.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
import torch

from rl_engine.reference.attention.joint_attn_qk_gemm import (
    JOINT_ATTN_QK_GEMM_CONTRACT_VERSION,
    LEAF_WIDTH,
    JointAttnQkGemmPlan,
    NativeJointAttnQkGemmOp,
    fma_is_single_rounding,
    joint_attn_qk_gemm_backward_with_leaves,
    joint_attn_qk_gemm_reference_backward,
    num_leaves,
    qk_scores_tree_fp32,
    tree_from_leaf_evaluator,
)

HEAD_DIM = 128
QWENIMAGE_HEADS = 24
SCALE = 1.0 / math.sqrt(HEAD_DIM)
# Image-token counts at the three reference resolutions of issue #386
# (1024², 1328², 1664x928) after 2x2 packing of the 16-channel latents.
REFERENCE_IMAGE_TOKENS = (4096, 6889, 6032)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _cuda_op():
    """The CUDA backend, or a skip when the native extension is not built."""
    from rl_engine.backends.cuda.attention.joint_attn_qk_gemm import JointAttnQkGemmCudaOp

    try:
        return JointAttnQkGemmCudaOp()
    except RuntimeError as exc:  # pragma: no cover - depends on the build
        pytest.skip(f"joint_attn_qk_gemm CUDA kernel unavailable: {exc}")


def bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """True when two tensors agree in shape, dtype and every payload bit."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    view = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[left.dtype]
    return torch.equal(
        left.detach().cpu().contiguous().view(view),
        right.detach().cpu().contiguous().view(view),
    )


# --------------------------------------------------------------------------- #
# Independent correctly-rounded FMA oracle (exact rational arithmetic)
# --------------------------------------------------------------------------- #
def _round_fp32_nearest_even(exact: Fraction) -> torch.Tensor:
    """Correctly round an exact rational to fp32, ties-to-even.

    Independent of any FMA primitive: the candidate set is the fp32 neighbourhood
    of the double approximation, and the winner is chosen by exact rational
    distance with an even-mantissa tie-break.
    """
    approx = torch.tensor([float(exact)], dtype=torch.float32)
    candidates = {float(approx.item())}
    for _ in range(2):
        for value in list(candidates):
            base = torch.tensor([value], dtype=torch.float32)
            for direction in (math.inf, -math.inf):
                step = torch.nextafter(base, torch.tensor([direction], dtype=torch.float32))
                candidates.add(float(step.item()))
    best_key = None
    best = None
    for value in candidates:
        if not math.isfinite(value):
            continue
        tensor = torch.tensor([value], dtype=torch.float32)
        distance = abs(Fraction(value) - exact)
        mantissa_odd = int(tensor.view(torch.int32).item()) & 1
        key = (distance, mantissa_odd)
        if best_key is None or key < best_key:
            best_key, best = key, tensor
    assert best is not None
    return best


def fma_oracle(a: float, b: float, c: float) -> torch.Tensor:
    """Correctly-rounded fp32 ``a * b + c`` with a single rounding."""
    return _round_fp32_nearest_even(Fraction(a) * Fraction(b) + Fraction(c))


def _fp32(value: float) -> float:
    return float(torch.tensor([value], dtype=torch.float32).item())


# The classic double-rounding counterexample: a true FMA returns 0x3F801001 and
# a multiply-then-add with two roundings returns 0x3F801000.
_COUNTEREXAMPLE_A = _fp32(1.0 + 2.0**-12)
_COUNTEREXAMPLE_C = _fp32(2.0**-60)


class TestFmaOracle:
    """The reference's multiply-into-add primitive, verified from first principles."""

    def test_oracle_resolves_the_double_rounding_counterexample(self):
        oracle = fma_oracle(_COUNTEREXAMPLE_A, _COUNTEREXAMPLE_A, _COUNTEREXAMPLE_C)
        assert int(oracle.view(torch.int32).item()) & 0xFFFFFFFF == 0x3F801001

    def test_addcmul_is_a_correctly_rounded_fma_on_cpu(self):
        a = torch.tensor([_COUNTEREXAMPLE_A], dtype=torch.float32)
        c = torch.tensor([_COUNTEREXAMPLE_C], dtype=torch.float32)
        assert bitwise_equal(
            torch.addcmul(c, a, a),
            fma_oracle(_COUNTEREXAMPLE_A, _COUNTEREXAMPLE_A, _COUNTEREXAMPLE_C),
        )
        assert fma_is_single_rounding("cpu")

    def test_separate_multiply_then_add_double_rounds_on_cpu(self):
        a = torch.tensor([_COUNTEREXAMPLE_A], dtype=torch.float32)
        c = torch.tensor([_COUNTEREXAMPLE_C], dtype=torch.float32)
        assert not bitwise_equal(
            c + a * a, fma_oracle(_COUNTEREXAMPLE_A, _COUNTEREXAMPLE_A, _COUNTEREXAMPLE_C)
        )

    def test_addcmul_matches_the_oracle_on_adversarial_samples(self):
        torch.manual_seed(11)
        count = 4096
        a = torch.rand(count, dtype=torch.float32) * 4 - 2
        b = torch.rand(count, dtype=torch.float32) * 4 - 2
        scales = torch.tensor([1.0, 2.0**-30, 2.0**-55, 2.0**-12]).repeat(count // 4)
        c = (torch.rand(count, dtype=torch.float32) * 4 - 2) * scales
        observed = torch.addcmul(c, a, b)
        for index in range(0, count, 17):  # strided subsample keeps the exact oracle cheap
            expected = fma_oracle(
                float(a[index].item()), float(b[index].item()), float(c[index].item())
            )
            assert bitwise_equal(observed[index : index + 1], expected), index

    @requires_cuda
    def test_addcmul_is_not_a_single_rounding_fma_on_cuda(self):
        """Measured portability hazard; it is why the gold is pinned to CPU.

        ``torch.addcmul`` lowers to a separate multiply and add on CUDA, so the
        same inputs yield different bits than on CPU. The reference fails closed
        there rather than silently returning different bytes.
        """
        assert not fma_is_single_rounding("cuda")
        a = torch.tensor([_COUNTEREXAMPLE_A], dtype=torch.float32, device="cuda")
        c = torch.tensor([_COUNTEREXAMPLE_C], dtype=torch.float32, device="cuda")
        assert bitwise_equal(torch.addcmul(c, a, a), c + a * a)

    @requires_cuda
    def test_fp32_gold_fails_closed_on_cuda(self):
        op = NativeJointAttnQkGemmOp()
        q = torch.randn(1, 4, 3, HEAD_DIM, device="cuda")
        k = torch.randn(1, 4, 5, HEAD_DIM, device="cuda")
        with pytest.raises(RuntimeError, match="correctly-rounded"):
            op.forward_fp32(q, k)


# --------------------------------------------------------------------------- #
# Reference structure and self-consistency
# --------------------------------------------------------------------------- #
class TestReferenceTree:
    def test_production_head_dim_is_four_leaves(self):
        assert LEAF_WIDTH == 32
        assert num_leaves(HEAD_DIM) == 4

    @pytest.mark.parametrize("length,expected", [(1, 1), (32, 1), (33, 2), (128, 4), (6889, 216)])
    def test_leaf_counts(self, length, expected):
        assert num_leaves(length) == expected

    def test_midsplit_tree_is_a_halving_tree_only_for_power_of_two_leaf_counts(self):
        """Why the device forward may close the tree in registers but the backward may not."""

        def midsplit(lo, hi):
            if hi - lo == 1:
                return (lo,)
            mid = lo + (hi - lo) // 2
            return ("+", midsplit(lo, mid), midsplit(mid, hi))

        def halving(count):
            level = [(index,) for index in range(count)]
            while len(level) > 1:
                nxt = [("+", level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
                if len(level) % 2:
                    nxt.append(level[-1])
                level = nxt
            return level[0]

        for count in (1, 2, 4, 8, 16, 32):
            assert midsplit(0, count) == halving(count), count
        for count in (3, 5, 6, 7, 216):
            assert midsplit(0, count) != halving(count), count

    def test_tree_requires_at_least_one_leaf(self):
        with pytest.raises(ValueError, match="at least one leaf"):
            tree_from_leaf_evaluator(lambda _: torch.zeros(1), 0)

    def test_scale_is_not_folded_into_the_unscaled_tree(self):
        torch.manual_seed(3)
        q = torch.randn(1, 2, 4, HEAD_DIM)
        k = torch.randn(1, 2, 6, HEAD_DIM)
        unscaled = qk_scores_tree_fp32(q, k)
        op = NativeJointAttnQkGemmOp()
        assert bitwise_equal(op.forward_fp32(q, k, scale=1.0), unscaled)


class TestReferenceForwardInvariance:
    """The forward reduces over the head dim alone, so it is invariant by construction."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("batch", [2, 4, 16, 300])
    def test_batch_size_and_position(self, dtype, batch):
        torch.manual_seed(5)
        op = NativeJointAttnQkGemmOp()
        base_q = torch.randn(1, 4, 6, HEAD_DIM, dtype=dtype)
        base_k = torch.randn(1, 4, 7, HEAD_DIM, dtype=dtype)
        alone = op.forward(base_q, base_k, scale=SCALE)

        companions_q = torch.randn(batch, 4, 6, HEAD_DIM, dtype=dtype)
        companions_k = torch.randn(batch, 4, 7, HEAD_DIM, dtype=dtype)
        position = batch - 1
        companions_q[position] = base_q[0]
        companions_k[position] = base_k[0]
        batched = op.forward(companions_q, companions_k, scale=SCALE)
        assert bitwise_equal(batched[position : position + 1], alone)

    def test_appended_key_padding_does_not_move_the_valid_block(self):
        torch.manual_seed(6)
        op = NativeJointAttnQkGemmOp()
        q = torch.randn(1, 3, 5, HEAD_DIM)
        k = torch.randn(1, 3, 9, HEAD_DIM)
        clean = op.forward_fp32(q, k, scale=SCALE)
        padded_k = torch.cat([k, torch.randn(1, 3, 13, HEAD_DIM)], dim=-2)
        padded = op.forward_fp32(q, padded_k, scale=SCALE)
        assert bitwise_equal(padded[..., :9].contiguous(), clean)

    def test_repeated_runs_are_bit_identical(self):
        torch.manual_seed(7)
        op = NativeJointAttnQkGemmOp()
        q = torch.randn(2, 3, 5, HEAD_DIM)
        k = torch.randn(2, 3, 5, HEAD_DIM)
        first = op.forward_fp32(q, k, scale=SCALE)
        for _ in range(3):
            assert bitwise_equal(op.forward_fp32(q, k, scale=SCALE), first)


class TestReferenceBackward:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_autograd_matches_the_explicit_vjp(self, dtype):
        torch.manual_seed(8)
        op = NativeJointAttnQkGemmOp()
        q = torch.randn(2, 4, 7, HEAD_DIM, dtype=dtype, requires_grad=True)
        k = torch.randn(2, 4, 11, HEAD_DIM, dtype=dtype, requires_grad=True)
        scores = op(q, k, scale=SCALE)
        grad = torch.randn_like(scores)
        scores.backward(grad)
        expected_q, expected_k = joint_attn_qk_gemm_reference_backward(
            q.detach(), k.detach(), grad, scale=SCALE
        )
        assert bitwise_equal(q.grad, expected_q)
        assert bitwise_equal(k.grad, expected_k)

    @pytest.mark.parametrize("where", ["front", "back"])
    def test_backward_is_padding_invariant_with_explicit_masks(self, where):
        torch.manual_seed(9)
        batch, heads, queries, keys, pad = 2, 3, 7, 11, 5
        q = torch.randn(batch, heads, queries, HEAD_DIM)
        k = torch.randn(batch, heads, keys, HEAD_DIM)
        grad = torch.randn(batch, heads, queries, keys)
        all_valid_q = torch.ones(batch, queries, dtype=torch.bool)

        clean_q, clean_k = joint_attn_qk_gemm_reference_backward(
            q,
            k,
            grad,
            scale=SCALE,
            query_padding_mask=all_valid_q,
            key_padding_mask=torch.ones(batch, keys, dtype=torch.bool),
        )

        junk_k = torch.randn(batch, heads, pad, HEAD_DIM)
        junk_g = torch.randn(batch, heads, queries, pad)
        invalid = torch.zeros(batch, pad, dtype=torch.bool)
        valid = torch.ones(batch, keys, dtype=torch.bool)
        if where == "back":
            padded_k = torch.cat([k, junk_k], dim=-2)
            padded_g = torch.cat([grad, junk_g], dim=-1)
            mask = torch.cat([valid, invalid], dim=1)
            keep = slice(0, keys)
        else:
            padded_k = torch.cat([junk_k, k], dim=-2)
            padded_g = torch.cat([junk_g, grad], dim=-1)
            mask = torch.cat([invalid, valid], dim=1)
            keep = slice(pad, pad + keys)

        padded_dq, padded_dk = joint_attn_qk_gemm_reference_backward(
            q,
            padded_k,
            padded_g,
            scale=SCALE,
            query_padding_mask=all_valid_q,
            key_padding_mask=mask,
        )
        assert bitwise_equal(padded_dq, clean_q)
        assert bitwise_equal(padded_dk[..., keep, :].contiguous(), clean_k)

    def test_padded_key_rows_have_exactly_zero_gradient(self):
        torch.manual_seed(10)
        batch, heads, queries, keys = 1, 2, 4, 9
        q = torch.randn(batch, heads, queries, HEAD_DIM)
        k = torch.randn(batch, heads, keys, HEAD_DIM)
        grad = torch.randn(batch, heads, queries, keys)
        mask = torch.ones(batch, keys, dtype=torch.bool)
        mask[:, 5:] = False
        _, grad_k = joint_attn_qk_gemm_reference_backward(
            q, k, grad, scale=SCALE, key_padding_mask=mask
        )
        assert bool((grad_k[:, :, 5:, :] == 0).all())

    @pytest.mark.parametrize("batch", [2, 4, 16])
    def test_gradient_batch_and_position_invariance(self, batch):
        torch.manual_seed(12)
        q = torch.randn(1, 3, 6, HEAD_DIM)
        k = torch.randn(1, 3, 8, HEAD_DIM)
        grad = torch.randn(1, 3, 6, 8)
        alone_q, alone_k = joint_attn_qk_gemm_reference_backward(q, k, grad, scale=SCALE)

        many_q = torch.randn(batch, 3, 6, HEAD_DIM)
        many_k = torch.randn(batch, 3, 8, HEAD_DIM)
        many_g = torch.randn(batch, 3, 6, 8)
        position = batch - 1
        many_q[position], many_k[position], many_g[position] = q[0], k[0], grad[0]
        batched_q, batched_k = joint_attn_qk_gemm_reference_backward(
            many_q, many_k, many_g, scale=SCALE
        )
        assert bitwise_equal(batched_q[position : position + 1], alone_q)
        assert bitwise_equal(batched_k[position : position + 1], alone_k)


class TestValidation:
    """Unsupported configurations fail closed rather than degrading silently."""

    def test_fp16_is_rejected(self):
        op = NativeJointAttnQkGemmOp()
        q = torch.randn(1, 2, 3, HEAD_DIM, dtype=torch.float16)
        with pytest.raises(ValueError, match="outside the WS1 contract"):
            op(q, q)

    def test_rank_is_checked(self):
        op = NativeJointAttnQkGemmOp()
        with pytest.raises(ValueError, match=r"\[B, H, S, D\]"):
            op(torch.randn(3, HEAD_DIM), torch.randn(5, HEAD_DIM))

    def test_gqa_head_counts_are_rejected(self):
        op = NativeJointAttnQkGemmOp()
        with pytest.raises(ValueError, match="head count mismatch"):
            op(torch.randn(1, 24, 4, HEAD_DIM), torch.randn(1, 8, 4, HEAD_DIM))

    def test_head_dim_mismatch_is_rejected(self):
        op = NativeJointAttnQkGemmOp()
        with pytest.raises(ValueError, match="head dim mismatch"):
            op(torch.randn(1, 2, 4, HEAD_DIM), torch.randn(1, 2, 4, 64))

    def test_mask_dtype_and_shape_are_checked(self):
        op = NativeJointAttnQkGemmOp()
        q = torch.randn(1, 2, 4, HEAD_DIM)
        k = torch.randn(1, 2, 6, HEAD_DIM)
        with pytest.raises(ValueError, match="must be bool"):
            op(q, k, key_padding_mask=torch.ones(1, 6))
        with pytest.raises(ValueError, match=r"must be \[B, S\]"):
            op(q, k, key_padding_mask=torch.ones(1, 5, dtype=torch.bool))

    def test_mixed_dtypes_are_rejected(self):
        op = NativeJointAttnQkGemmOp()
        with pytest.raises(ValueError, match="dtypes must match"):
            op(torch.randn(1, 2, 4, HEAD_DIM), torch.randn(1, 2, 4, HEAD_DIM, dtype=torch.bfloat16))


class TestPlanReadback:
    def test_declares_the_contract(self):
        plan = JointAttnQkGemmPlan(backend_id="test")
        data = plan.to_dict()
        assert data["contract_version"] == JOINT_ATTN_QK_GEMM_CONTRACT_VERSION
        assert data["accumulation_dtype"] == "torch.float32"
        assert data["forward_reduction_dim"] == "head_dim"
        assert data["scale_applied"] == "once_fp32_after_tree"
        assert data["causal"] is False
        assert data["fallback"] is False
        assert not any(data[flag] for flag in ("split_k", "atomics", "tf32", "fast_math"))

    @pytest.mark.parametrize(
        "override",
        [
            {"split_k": True},
            {"atomics": True},
            {"tf32": True},
            {"fast_math": True},
            {"accumulation_dtype": "torch.bfloat16"},
            {"scale_applied": "folded_into_q"},
        ],
    )
    def test_rejects_policies_outside_the_contract(self, override):
        with pytest.raises(ValueError):
            JointAttnQkGemmPlan(backend_id="test", **override)


# --------------------------------------------------------------------------- #
# CUDA backend
# --------------------------------------------------------------------------- #
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
        from rl_engine.backends.cuda.attention.joint_attn_qk_gemm import (
            _dk_leaf_cuda,
            _dq_leaf_cuda,
        )

        torch.manual_seed(21)
        q = torch.randn(1, 3, 40, HEAD_DIM)
        k = torch.randn(1, 3, 70, HEAD_DIM)
        grad = torch.randn(1, 3, 40, 70)
        _cuda_op()  # skips cleanly when the extension is missing

        device_q, device_k = joint_attn_qk_gemm_backward_with_leaves(
            q.cuda(),
            k.cuda(),
            grad.cuda(),
            scale=SCALE,
            dq_leaf_fn=_dq_leaf_cuda,
            dk_leaf_fn=_dk_leaf_cuda,
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
class TestReferenceShapes:
    """The three reference resolutions of issue #386, plus prompt text tokens.

    The full fp32 gold is not computed at these sizes; each score element reduces
    over the head dim independently, so comparing a query slice against the gold
    is a complete check of the arithmetic for those rows.
    """

    @pytest.mark.parametrize("image_tokens", REFERENCE_IMAGE_TOKENS)
    def test_query_slice_matches_the_gold(self, image_tokens):
        cuda_op = _cuda_op()
        gold = NativeJointAttnQkGemmOp()
        torch.manual_seed(23)
        text_tokens = 64
        joint = text_tokens + image_tokens
        q = torch.randn(1, 1, joint, HEAD_DIM, dtype=torch.bfloat16)
        k = torch.randn(1, 1, joint, HEAD_DIM, dtype=torch.bfloat16)

        device_rows = cuda_op.forward(q.cuda(), k.cuda(), scale=SCALE)[:, :, :8, :].cpu()
        gold_rows = gold.forward(q[:, :, :8, :], k, scale=SCALE)
        assert bitwise_equal(device_rows.contiguous(), gold_rows)

    @pytest.mark.parametrize("image_tokens", REFERENCE_IMAGE_TOKENS)
    def test_text_prefix_rows_are_padding_invariant(self, image_tokens):
        cuda_op = _cuda_op()
        torch.manual_seed(24)
        text_tokens = 64
        q = torch.randn(1, 1, 8, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 1, text_tokens, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        clean = cuda_op.forward(q, k, scale=SCALE)
        image_k = torch.randn(1, 1, image_tokens, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        joined = cuda_op.forward(q, torch.cat([k, image_k], dim=-2), scale=SCALE)
        assert bitwise_equal(joined[..., :text_tokens].contiguous(), clean)


class TestRegistryDispatch:
    def test_cpu_dispatch_selects_the_reference(self):
        from rl_engine.runtime.registry import KernelRegistry

        op = KernelRegistry().get_op("joint_attn_qk_gemm", device="cpu")
        assert type(op).__name__ == "NativeJointAttnQkGemmOp"

    @requires_cuda
    def test_cuda_dispatch_prefers_the_kernel_when_built(self):
        from rl_engine.backends.extension import _C, _EXT_AVAILABLE
        from rl_engine.runtime.registry import KernelRegistry

        if not (_EXT_AVAILABLE and hasattr(_C, "joint_attn_qk_gemm_forward")):
            pytest.skip("native extension not built with joint_attn_qk_gemm")
        op = KernelRegistry().get_op("joint_attn_qk_gemm", device="cuda")
        assert type(op).__name__ == "JointAttnQkGemmCudaOp"

    def test_gtest_spec_is_registered(self):
        from rl_engine.validation.operators.operator_specs import OP_SPECS

        spec = OP_SPECS["joint_attn_qk_gemm"]
        assert spec.grad_input_names == ("q", "k")
        assert set(spec.candidate_paths) == {"pytorch", "cuda"}
        assert spec.gold_method == "forward_fp32"
