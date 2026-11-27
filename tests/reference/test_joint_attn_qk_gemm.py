# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Qwen-Image WS1 joint attention QK GEMM reference (issue #386).

The reference is the fp32 CPU gold for contract ``joint-attn-qk-gemm-tree-v1``.
Every assertion is bitwise except the fp64 oracle, which bounds the tree's own
rounding error. The file also carries an **independent FMA oracle**: an
exact-rational correctly-rounded FP32 fused multiply-add, used to verify the
reference's ``torch.addcmul`` primitive from first principles and to pin the
measured CPU/CUDA divergence of that primitive.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
import torch

import rl_engine.reference.attention.joint_attn_qk_gemm as reference_module
from rl_engine.reference.attention.joint_attn_qk_gemm import (
    JOINT_ATTN_QK_GEMM_CONTRACT_VERSION,
    LEAF_WIDTH,
    JointAttnQkGemmPlan,
    NativeJointAttnQkGemmOp,
    fma_is_single_rounding,
    joint_attn_qk_gemm_reference_backward,
    num_leaves,
    qk_scores_tree_fp32,
    tree_from_leaf_evaluator,
)
from rl_engine.validation.common.bitwise import tensor_bytes_equal as bitwise_equal

HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


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
    def test_every_entry_fails_closed_on_cuda(self):
        op = NativeJointAttnQkGemmOp()
        q = torch.randn(1, 4, 3, HEAD_DIM, device="cuda")
        k = torch.randn(1, 4, 5, HEAD_DIM, device="cuda")
        with pytest.raises(RuntimeError, match="correctly-rounded"):
            op.forward_fp32(q, k)
        with pytest.raises(RuntimeError, match="correctly-rounded"):
            op.forward(q, k)
        with pytest.raises(RuntimeError, match="correctly-rounded"):
            joint_attn_qk_gemm_reference_backward(q, k, torch.randn(1, 4, 3, 5, device="cuda"))

    def test_every_entry_fails_closed_when_the_cpu_probe_fails(self, monkeypatch):
        """A CPU without FMA kernels (ATen's ``default`` capability) must not produce gold."""
        monkeypatch.setitem(reference_module._fma_regime_cache, "cpu", False)
        op = NativeJointAttnQkGemmOp()
        q = torch.randn(1, 2, 3, HEAD_DIM)
        k = torch.randn(1, 2, 5, HEAD_DIM)
        for entry in (op.forward, op.forward_fp32, op.__call__):
            with pytest.raises(RuntimeError, match="correctly-rounded"):
                entry(q, k)
        with pytest.raises(RuntimeError, match="correctly-rounded"):
            joint_attn_qk_gemm_reference_backward(q, k, torch.randn(1, 2, 3, 5))

    def test_every_entry_fails_closed_under_flush_to_zero(self):
        """The device kernels keep subnormals, so a flushed CPU gold must not run."""
        minimum_normal = torch.tensor([torch.finfo(torch.float32).tiny])
        was_flushing = (minimum_normal * 0.5).item() == 0.0
        op = NativeJointAttnQkGemmOp()
        q = torch.full((1, 1, 1, 32), 2.0**-70)
        try:
            if not torch.set_flush_denormal(True):
                pytest.skip("torch.set_flush_denormal is unsupported on this CPU")
            for entry in (op.forward, op.forward_fp32):
                with pytest.raises(RuntimeError, match=r"set_flush_denormal\(False\)"):
                    entry(q, q)
            with pytest.raises(RuntimeError, match=r"set_flush_denormal\(False\)"):
                joint_attn_qk_gemm_reference_backward(q, q, torch.ones(1, 1, 1, 1))
        finally:
            torch.set_flush_denormal(was_flushing)

    def test_subnormal_scores_survive_gradual_underflow(self):
        q = torch.full((1, 1, 1, 32), 2.0**-70)
        scores = NativeJointAttnQkGemmOp().forward_fp32(q, q, scale=1.0)
        assert float(scores.item()) == 32 * 2.0**-140

    def test_probe_covers_the_vector_and_outer_product_layouts(self, monkeypatch):
        """The leaves run vectorized and broadcast addcmul, not only a scalar lane."""
        monkeypatch.delitem(reference_module._fma_regime_cache, "cpu", raising=False)
        lanes = 67
        a = torch.full((lanes,), _COUNTEREXAMPLE_A, dtype=torch.float32)
        c = torch.full((lanes,), _COUNTEREXAMPLE_C, dtype=torch.float32)
        expected = fma_oracle(_COUNTEREXAMPLE_A, _COUNTEREXAMPLE_A, _COUNTEREXAMPLE_C)
        outer = torch.addcmul(c.expand(lanes, lanes), a.unsqueeze(-1), a.unsqueeze(-2))
        assert bitwise_equal(torch.addcmul(c, a, a), expected.expand(lanes))
        assert bitwise_equal(outer, expected.expand(lanes, lanes))
        assert fma_is_single_rounding("cpu")


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

    @pytest.mark.parametrize("masked", [False, True])
    def test_backward_is_within_the_tree_error_bound_of_an_fp64_oracle(self, masked):
        """An oracle independent of the shared host tree, leaves and compaction."""
        torch.manual_seed(29)
        batch, heads, queries, keys = 2, 3, 40, 70
        q = torch.randn(batch, heads, queries, HEAD_DIM)
        k = torch.randn(batch, heads, keys, HEAD_DIM)
        grad = torch.randn(batch, heads, queries, keys)
        query_mask = key_mask = None
        valid_q = torch.ones(batch, queries, dtype=torch.bool)
        valid_k = torch.ones(batch, keys, dtype=torch.bool)
        if masked:
            valid_q[0, 33:] = False
            valid_k[1, :9] = False
            query_mask, key_mask = valid_q, valid_k
        grad_q, grad_k = joint_attn_qk_gemm_reference_backward(
            q, k, grad, scale=SCALE, query_padding_mask=query_mask, key_padding_mask=key_mask
        )

        # The masked backward reduces over valid positions only (contract item 4).
        g64 = grad.double() * (valid_q[:, None, :, None] & valid_k[:, None, None, :])
        exact_q = torch.einsum("bhij,bhjd->bhid", g64, k.double()) * SCALE
        exact_k = torch.einsum("bhij,bhid->bhjd", g64, q.double()) * SCALE
        exact_q = exact_q * valid_q[:, None, :, None]
        exact_k = exact_k * valid_k[:, None, :, None]
        magnitude_q = torch.einsum("bhij,bhjd->bhid", g64.abs(), k.double().abs()) * SCALE
        magnitude_k = torch.einsum("bhij,bhid->bhjd", g64.abs(), q.double().abs()) * SCALE
        # A 32-term FMA leaf, a mid-split tree of depth ceil(log2(leaves)), the
        # scale multiply and the output cast: each adds at most one fp32 rounding.
        for observed, exact, magnitude, length in (
            (grad_q, exact_q, magnitude_q, keys),
            (grad_k, exact_k, magnitude_k, queries),
        ):
            roundings = LEAF_WIDTH + math.ceil(math.log2(num_leaves(length))) + 2
            bound = roundings * 2.0**-24 * magnitude + 1e-30
            assert bool(((observed.double() - exact).abs() <= bound).all())

    def test_masked_backward_equals_the_unmasked_vjp_for_zeroed_padded_gradients(self):
        """Contract item 4: masks drop padded gradient entries, which must be zero."""
        torch.manual_seed(30)
        queries, keys = 40, 64
        q = torch.randn(1, 2, queries, HEAD_DIM)
        k = torch.randn(1, 2, keys, HEAD_DIM)
        query_mask = torch.ones(1, queries, dtype=torch.bool)
        key_mask = torch.ones(1, keys, dtype=torch.bool)
        query_mask[:, 32:] = False
        key_mask[:, 32:] = False
        grad = torch.randn(1, 2, queries, keys)
        grad = grad * (query_mask[:, None, :, None] & key_mask[:, None, None, :])

        masked = joint_attn_qk_gemm_reference_backward(
            q, k, grad, scale=SCALE, query_padding_mask=query_mask, key_padding_mask=key_mask
        )
        unmasked = joint_attn_qk_gemm_reference_backward(q, k, grad, scale=SCALE)
        assert bitwise_equal(masked[0], unmasked[0])
        assert bitwise_equal(masked[1], unmasked[1])

    def test_backward_skips_an_unneeded_gradient(self):
        torch.manual_seed(31)
        op = NativeJointAttnQkGemmOp()
        q = torch.randn(1, 2, 5, HEAD_DIM)
        k = torch.randn(1, 2, 7, HEAD_DIM)
        grad = torch.randn(1, 2, 5, 7)
        only_q = q.clone().requires_grad_()
        op(only_q, k, scale=SCALE).backward(grad)
        expected_q, _ = joint_attn_qk_gemm_reference_backward(q, k, grad, scale=SCALE)
        assert bitwise_equal(only_q.grad, expected_q)
        _, missing_k = joint_attn_qk_gemm_reference_backward(q, k, grad, scale=SCALE, need_dk=False)
        assert missing_k is None

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

    def test_mask_device_is_checked(self):
        op = NativeJointAttnQkGemmOp()
        q = torch.randn(1, 2, 4, HEAD_DIM)
        k = torch.randn(1, 2, 6, HEAD_DIM)
        mask = torch.ones(1, 6, dtype=torch.bool, device="meta")
        with pytest.raises(ValueError, match="key_padding_mask must be on cpu"):
            op(q, k, key_padding_mask=mask)

    def test_mixed_dtypes_are_rejected(self):
        op = NativeJointAttnQkGemmOp()
        with pytest.raises(ValueError, match="dtypes must match"):
            op(torch.randn(1, 2, 4, HEAD_DIM), torch.randn(1, 2, 4, HEAD_DIM, dtype=torch.bfloat16))


class TestPlanReadback:
    def test_reference_reads_back_its_plan_and_provenance(self):
        op = NativeJointAttnQkGemmOp()
        assert op.last_plan.backend_id == op.backend_id
        assert op.last_plan.fallback is False
        assert op.provenance["fallback"] is False
        assert op.provenance["contract_version"] == JOINT_ATTN_QK_GEMM_CONTRACT_VERSION
        assert op.trace()["actual_backend"] == op.backend_id
        assert op.trace()["fallback"] is False

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
