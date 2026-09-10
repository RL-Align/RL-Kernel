# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the Ascend NPU deterministic standard-softmax attention.

Validates the same two orthogonal properties as the CUDA deterministic op:
1. **Correctness** - output matches the ``NativeAttentionOp.forward_fp32``
   ground truth within the reduction tolerances.
2. **Batch-invariance** - a query row's output is bitwise identical regardless
   of batch size, batch position, or how many AI-core blocks were launched
   (each row is reduced end-to-end by one block; no split-K merge exists).
"""

import math

import pytest
import torch

from rl_engine.kernels.ops.pytorch.attention.standard_attn import NativeAttentionOp

_D = 128

# Accuracy tolerance from the gtest contract, "attention" op class.
_ATOL = {torch.bfloat16: 5.0e-2, torch.float16: 1.0e-3}
_RTOL = {torch.bfloat16: 2.0e-2, torch.float16: 1.0e-3}


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


def _ascend_kernel_available() -> bool:
    if not _npu_available():
        return False
    try:
        from rl_engine.kernels.ops.ascend.attention.deterministic_attn import (
            _NPU_EXT_AVAILABLE,
            _C_npu,
        )
    except Exception:
        return False
    return _NPU_EXT_AVAILABLE and hasattr(_C_npu, "deterministic_attention_ascend")


requires_ascend = pytest.mark.skipif(
    not _ascend_kernel_available(),
    reason="deterministic_attention_ascend kernel not compiled "
    "(needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).",
)


def _get_op():
    from rl_engine.kernels.ops.ascend.attention.deterministic_attn import (
        DeterministicAttentionAscendOp,
    )

    return DeterministicAttentionAscendOp()


def _gold(q, k, v, causal=True, scale=None, key_padding_mask=None):
    """fp32 ground truth: NativeAttentionOp.forward_fp32."""
    return NativeAttentionOp().forward_fp32(
        q, k, v, causal=causal, scale=scale, key_padding_mask=key_padding_mask
    )


def _make_qkv(batch, hq, hkv, sq, skv, dtype, seed=0):
    # Independent generator per tensor: batch size must not shift the k/v
    # content (a shared generator would make k[0] differ between batch sizes,
    # breaking the batch-invariance comparisons below).
    gq = torch.Generator(device="cpu").manual_seed(seed)
    gk = torch.Generator(device="cpu").manual_seed(seed + 1)
    gv = torch.Generator(device="cpu").manual_seed(seed + 2)
    q = torch.randn(batch, hq, sq, _D, dtype=dtype, generator=gq).to("npu")
    k = torch.randn(batch, hkv, skv, _D, dtype=dtype, generator=gk).to("npu")
    v = torch.randn(batch, hkv, skv, _D, dtype=dtype, generator=gv).to("npu")
    return q, k, v


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@requires_ascend
class TestAscendAttentionCorrectness:
    def test_prefill_causal(self, dtype):
        op = _get_op()
        q, k, v = _make_qkv(2, 8, 2, 128, 128, dtype)
        out = op(q, k, v, causal=True)
        gold = _gold(q, k, v, causal=True)
        assert out.dtype == dtype
        assert torch.allclose(out.float(), gold, atol=_ATOL[dtype], rtol=_RTOL[dtype])

    def test_gqa(self, dtype):
        op = _get_op()
        q, k, v = _make_qkv(2, 8, 2, 64, 64, dtype)  # g = 8/2 = 4
        out = op(q, k, v, causal=True)
        gold = _gold(q, k, v, causal=True)
        assert torch.allclose(out.float(), gold, atol=_ATOL[dtype], rtol=_RTOL[dtype])

    def test_decode_window(self, dtype):
        op = _get_op()
        q, k, v = _make_qkv(2, 8, 2, 4, 96, dtype)  # Sq < Skv
        out = op(q, k, v, causal=True)
        gold = _gold(q, k, v, causal=True)
        assert torch.allclose(out.float(), gold, atol=_ATOL[dtype], rtol=_RTOL[dtype])

    def test_non_causal(self, dtype):
        op = _get_op()
        q, k, v = _make_qkv(2, 8, 2, 64, 64, dtype)
        out = op(q, k, v, causal=False)
        gold = _gold(q, k, v, causal=False)
        assert torch.allclose(out.float(), gold, atol=_ATOL[dtype], rtol=_RTOL[dtype])

    def test_key_padding_mask(self, dtype):
        op = _get_op()
        q, k, v = _make_qkv(2, 8, 2, 64, 100, dtype)  # Skv not a multiple of 64
        mask = torch.ones(2, 100, dtype=torch.bool, device="npu")
        mask[:, 80:] = False
        out = op(q, k, v, causal=True, key_padding_mask=mask)
        gold = _gold(q, k, v, causal=True, key_padding_mask=mask)
        assert torch.allclose(out.float(), gold, atol=_ATOL[dtype], rtol=_RTOL[dtype])

    def test_fully_masked_row_is_zero(self, dtype):
        op = _get_op()
        q, k, v = _make_qkv(2, 4, 4, 2, 32, dtype)
        mask = torch.ones(2, 32, dtype=torch.bool, device="npu")
        mask[1, :] = False  # batch 1 has no valid key at all
        out, lse = op.forward_with_lse(q, k, v, causal=True, key_padding_mask=mask)
        # Batch 1 has zero valid keys -> defined as 0, lse = -inf.
        assert torch.equal(out[1], torch.zeros_like(out[1]))
        assert torch.all(lse[1] == float("-inf"))
        # Batch 0 still finite.
        assert torch.isfinite(out[0]).all()
        assert torch.isfinite(lse[0]).all()

    def test_explicit_scale(self, dtype):
        op = _get_op()
        q, k, v = _make_qkv(1, 4, 4, 32, 32, dtype)
        out = op(q, k, v, causal=False, scale=0.05)
        gold = _gold(q, k, v, causal=False, scale=0.05)
        assert torch.allclose(out.float(), gold, atol=_ATOL[dtype], rtol=_RTOL[dtype])

    def test_forward_with_lse(self, dtype):
        op = _get_op()
        q, k, v = _make_qkv(1, 4, 4, 64, 64, dtype)
        out, lse = op.forward_with_lse(q, k, v, causal=True)
        scale = 1.0 / math.sqrt(_D)
        qf, kf = q.float(), k.float()
        scores = qf @ kf.transpose(-1, -2) * scale
        cm = torch.triu(torch.ones(64, 64, dtype=torch.bool, device="npu"), 1)
        scores = scores.masked_fill(cm, float("-inf"))
        ref_lse = torch.logsumexp(scores, dim=-1)
        assert lse.dtype == torch.float32
        assert lse.shape == (1, 4, 64)
        assert torch.allclose(lse, ref_lse, atol=1e-3, rtol=1e-3)

    def test_backward_grads(self, dtype):
        op = _get_op()
        q, k, v = _make_qkv(1, 4, 4, 32, 48, dtype)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        out = op(q, k, v, causal=True)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        assert all(g is not None for g in (q.grad, k.grad, v.grad))
        assert all(torch.isfinite(g).all() for g in (q.grad, k.grad, v.grad))

        # The backward is the VJP of the fp32 reference forward; compare.
        with torch.enable_grad():
            q_ref = q.detach().requires_grad_(True)
            k_ref = k.detach().requires_grad_(True)
            v_ref = v.detach().requires_grad_(True)
            ref_out = NativeAttentionOp().forward_fp32(q_ref, k_ref, v_ref, causal=True)
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(ref_out, (q_ref, k_ref, v_ref), grad_out)
        # The backward recomputes the same reference forward, so the VJPs
        # match to numerical noise.
        assert torch.allclose(q.grad.float(), dq_ref.float(), atol=1e-6, rtol=1e-5)
        assert torch.allclose(k.grad.float(), dk_ref.float(), atol=1e-6, rtol=1e-5)
        assert torch.allclose(v.grad.float(), dv_ref.float(), atol=1e-6, rtol=1e-5)


@requires_ascend
class TestAscendAttentionRejects:
    """Out-of-domain inputs must be rejected up front."""

    def test_rejects_fp32(self):
        op = _get_op()
        q, k, v = _make_qkv(1, 4, 4, 32, 32, torch.float32)
        with pytest.raises(ValueError, match="only FP16/BF16"):
            op(q, k, v)

    def test_rejects_bad_head_dim(self):
        op = _get_op()
        q = torch.randn(1, 4, 32, 64, device="npu", dtype=torch.float16)
        k = torch.randn(1, 4, 32, 64, device="npu", dtype=torch.float16)
        v = torch.randn(1, 4, 32, 64, device="npu", dtype=torch.float16)
        with pytest.raises(ValueError, match="head dim D must be 128"):
            op(q, k, v)


# ---------------------------------------------------------------------------
# Batch invariance
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendAttentionBatchInvariance:
    def _run_row(self, batch, hq, hkv, sq, skv, dtype, pos, seed=7):
        """One fixed query row embedded at position `pos` of a random batch."""
        op = _get_op()
        q, k, v = _make_qkv(batch, hq, hkv, sq, skv, dtype, seed=seed)
        out = op(q, k, v, causal=True)
        return out[0, 0, pos, :].clone()

    def test_batch_size_1_vs_n(self):
        dtype = torch.float16
        alone = self._run_row(1, 8, 2, 64, 64, dtype, pos=0, seed=7)
        for batch in (2, 4, 8):
            in_batch = self._run_row(batch, 8, 2, 64, 64, dtype, pos=0, seed=7)
            assert torch.equal(alone, in_batch), f"drift at batch_size={batch}"

    def test_different_positions_in_batch(self):
        # Non-causal: every position attends to the same full window. Copy the
        # same query row content into every position, then require
        # bitwise-identical output wherever it sits. (Causal windows differ
        # per position, so a causal sweep would compare different reductions.)
        dtype = torch.bfloat16
        op = _get_op()
        q, k, v = _make_qkv(2, 8, 2, 64, 64, dtype, seed=11)
        q[0, :, :, :] = q[0, :, 0:1, :]  # same row content at every position
        out = op(q, k, v, causal=False)
        baseline = out[0, 0, 0, :].clone()
        for pos in range(1, 16):
            assert torch.equal(baseline, out[0, 0, pos, :]), f"drift at position={pos}"

    def test_block_striding(self):
        # 2 * 8 * 128 = 2048 work items > MAX_BLOCKS (512): rows are strided
        # across blocks, so numerics must not depend on block assignment.
        # The small run (1 * 4 * 32 = 128 items, one block per item) and the
        # strided run must give the bitwise-identical row for the same content.
        dtype = torch.float16
        op = _get_op()
        small_q, small_k, small_v = _make_qkv(1, 4, 2, 32, 32, dtype, seed=3)
        small = op(small_q, small_k, small_v, causal=True)
        big_q, big_k, big_v = _make_qkv(2, 8, 2, 128, 128, dtype, seed=3)
        big_q[:, 0, 0, :] = small_q[0, 0, 0, :]
        big_k[:, 0, :32, :] = small_k[0, 0, :, :]
        big_v[:, 0, :32, :] = small_v[0, 0, :, :]
        big = op(big_q, big_k, big_v, causal=True)
        # Row (0, head 0, pos 0): causal window is j <= 0 in both runs, so the
        # other 96 keys cannot influence the result.
        assert torch.equal(big[0, 0, 0, :], small[0, 0, 0, :])

    def test_repeated_runs_deterministic(self):
        dtype = torch.bfloat16
        q, k, v = _make_qkv(2, 8, 2, 128, 128, dtype, seed=5)
        op = _get_op()
        first = op(q, k, v, causal=True)
        for _ in range(3):
            again = op(q, k, v, causal=True)
            assert torch.equal(first, again)


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendRegistryDispatch:
    def test_get_op_attention(self):
        from rl_engine.kernels.registry import kernel_registry

        op = kernel_registry.get_op("attention", device="npu")
        assert type(op).__name__ == "DeterministicAttentionAscendOp"

    def test_get_op_attn_falls_back_to_sdpa(self):
        from rl_engine.kernels.registry import kernel_registry

        op = kernel_registry.get_op("attn", device="npu")
        assert type(op).__name__ == "NativeAttentionOp"
