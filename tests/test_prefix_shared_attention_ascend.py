# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the Ascend NPU prefix-shared fused attention.

Validates the same properties as the CUDA prefix-shared op:
1. **Correctness** - output matches the ``NativeAttentionOp.forward_fp32``
   ground truth (full softmax over the shared K/V, no causal mask) within the
   reduction tolerances.
2. **Batch-invariance** - a query row's output is bitwise identical regardless
   of batch size, batch position, or how many AI-core blocks were launched
   (each (bs, g, 64-row block) item is processed end-to-end by one block).
"""

import math

import pytest
import torch

from rl_engine.kernels.ops.pytorch.attention.standard_attn import NativeAttentionOp

_D = 128

# Accuracy tolerance from the gtest contract, "attention" op class.
_ATOL = {torch.bfloat16: 5.0e-2}
_RTOL = {torch.bfloat16: 2.0e-2}


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
        from rl_engine.kernels.ops.ascend.attention.prefix_shared_attn import (
            _C_npu,
            _NPU_EXT_AVAILABLE,
        )
    except Exception:
        return False
    return _NPU_EXT_AVAILABLE and hasattr(_C_npu, "prefix_shared_attention_ascend")


requires_ascend = pytest.mark.skipif(
    not _ascend_kernel_available(),
    reason="prefix_shared_attention_ascend kernel not compiled "
    "(needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).",
)


def _get_op():
    from rl_engine.kernels.ops.ascend.attention.prefix_shared_attn import (
        PrefixSharedAttentionAscendOp,
    )

    return PrefixSharedAttentionAscendOp()


def _gold(q, k, v):
    """fp32 ground truth: softmax(Q K^T / sqrt(D)) V over the shared K/V."""
    return NativeAttentionOp().forward_fp32(
        q,
        k.unsqueeze(1),  # [bs, 1, Skv, D]: every G group shares the same KV head
        v.unsqueeze(1),
        causal=False,
        scale=1.0 / math.sqrt(_D),
    )


def _make_qkv(batch, groups, sq, skv, dtype, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(batch, groups, sq, _D, dtype=dtype, generator=generator).to("npu")
    k = torch.randn(batch, skv, _D, dtype=dtype, generator=generator).to("npu")
    v = torch.randn(batch, skv, _D, dtype=dtype, generator=generator).to("npu")
    return q, k, v


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendPrefixSharedAttentionCorrectness:
    def test_basic(self):
        op = _get_op()
        q, k, v = _make_qkv(2, 4, 65, 130, torch.bfloat16)  # ragged Sq/Skv
        out = op(q, k, v)
        gold = _gold(q, k, v)
        assert out.dtype == torch.bfloat16
        assert out.shape == (2, 4, 65, _D)
        assert torch.allclose(
            out.float(), gold, atol=_ATOL[torch.bfloat16], rtol=_RTOL[torch.bfloat16]
        )

    def test_exact_tiles(self):
        op = _get_op()
        q, k, v = _make_qkv(2, 8, 64, 64, torch.bfloat16)  # one Q block, one KV tile
        out = op(q, k, v)
        gold = _gold(q, k, v)
        assert torch.allclose(
            out.float(), gold, atol=_ATOL[torch.bfloat16], rtol=_RTOL[torch.bfloat16]
        )

    def test_decode_window(self):
        op = _get_op()
        q, k, v = _make_qkv(1, 16, 1, 512, torch.bfloat16)  # Sq << Skv, 8 KV tiles
        out = op(q, k, v)
        gold = _gold(q, k, v)
        assert torch.allclose(
            out.float(), gold, atol=_ATOL[torch.bfloat16], rtol=_RTOL[torch.bfloat16]
        )

    def test_long_prefix_multi_tile(self):
        op = _get_op()
        q, k, v = _make_qkv(1, 2, 32, 1024, torch.bfloat16)  # 16 KV tiles
        out = op(q, k, v)
        gold = _gold(q, k, v)
        assert torch.allclose(
            out.float(), gold, atol=_ATOL[torch.bfloat16], rtol=_RTOL[torch.bfloat16]
        )

    def test_shared_kv_across_groups(self):
        # Every G group attends over the exact same K/V; a G-sweep must match
        # the per-group reference (and be bitwise equal across G for equal q).
        op = _get_op()
        q, k, v = _make_qkv(1, 4, 32, 96, torch.bfloat16)
        q[0, 1, :, :] = q[0, 0, :, :]  # force identical query rows in 2 groups
        out = op(q, k, v)
        assert torch.equal(out[0, 0], out[0, 1])
        gold = _gold(q, k, v)
        assert torch.allclose(
            out.float(), gold, atol=_ATOL[torch.bfloat16], rtol=_RTOL[torch.bfloat16]
        )


# ---------------------------------------------------------------------------
# Batch invariance
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendPrefixSharedAttentionBatchInvariance:
    def test_batch_size_1_vs_n(self):
        # The same (b=0, g=0, row 0) content computed alone vs embedded in a
        # larger batch must be bitwise identical. (Content is copied in
        # explicitly: CPU bf16 randn consumes two fp32 draws per element, so
        # same-seed tensors of different batch sizes would not line up.)
        dtype = torch.bfloat16
        op = _get_op()
        q1, k1, v1 = _make_qkv(1, 4, 64, 64, dtype, seed=7)
        alone = op(q1, k1, v1)[0, 0, 0, :].clone()
        for batch in (2, 4, 8):
            q, k, v = _make_qkv(batch, 4, 64, 64, dtype, seed=7)
            q[0, 0, 0, :] = q1[0, 0, 0, :]
            k[0, :, :] = k1[0, :, :]
            v[0, :, :] = v1[0, :, :]
            in_batch = op(q, k, v)[0, 0, 0, :].clone()
            assert torch.equal(alone, in_batch), f"drift at batch_size={batch}"

    def test_different_positions_in_batch(self):
        # One fixed query row embedded at several positions (spanning both
        # 64-row query blocks) must give the bitwise-identical output at each.
        dtype = torch.bfloat16
        op = _get_op()
        q1, k1, v1 = _make_qkv(1, 1, 1, 64, dtype, seed=11)
        baseline = op(q1, k1, v1)[0, 0, 0, :].clone()
        q, k, v = _make_qkv(2, 4, 128, 64, dtype, seed=11)
        k[0, :, :] = k1[0, :, :]
        v[0, :, :] = v1[0, :, :]
        for pos in (0, 63, 64, 127):
            q[0, 0, pos, :] = q1[0, 0, 0, :]
        out = op(q, k, v)
        for pos in (0, 63, 64, 127):
            assert torch.equal(baseline, out[0, 0, pos, :]), f"drift at position={pos}"

    def test_block_striding(self):
        # The strided run below has 8 * 16 * (320/64) = 640 work items >
        # MAX_BLOCKS (512), so items are strided across blocks; numerics must
        # not depend on block assignment. The 1-item run and the strided run
        # must give the bitwise-identical rows for the same content.
        dtype = torch.bfloat16
        op = _get_op()
        small_q, small_k, small_v = _make_qkv(1, 1, 64, 64, dtype, seed=3)
        small = op(small_q, small_k, small_v)
        big_q, big_k, big_v = _make_qkv(8, 16, 320, 64, dtype, seed=3)
        big_q[0, 0, :64, :] = small_q[0, 0, :, :]
        big_k[0, :, :] = small_k[0, :, :]
        big_v[0, :, :] = small_v[0, :, :]
        big = op(big_q, big_k, big_v)
        assert torch.equal(big[0, 0, :64, :], small[0, 0, :, :])

    def test_repeated_runs_deterministic(self):
        dtype = torch.bfloat16
        q, k, v = _make_qkv(2, 4, 128, 128, dtype, seed=5)
        op = _get_op()
        first = op(q, k, v)
        for _ in range(3):
            again = op(q, k, v)
            assert torch.equal(first, again)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendPrefixSharedAttentionValidation:
    def test_rejects_fp32(self):
        op = _get_op()
        q, k, v = _make_qkv(1, 4, 32, 32, torch.float32)
        with pytest.raises(ValueError, match="only BF16"):
            op(q, k, v)

    def test_rejects_fp16(self):
        op = _get_op()
        q, k, v = _make_qkv(1, 4, 32, 32, torch.float16)
        with pytest.raises(ValueError, match="only BF16"):
            op(q, k, v)

    def test_rejects_bad_head_dim(self):
        op = _get_op()
        q = torch.randn(1, 4, 32, 64, device="npu", dtype=torch.bfloat16)
        k = torch.randn(1, 32, 64, device="npu", dtype=torch.bfloat16)
        v = torch.randn(1, 32, 64, device="npu", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="head dim D must be 128"):
            op(q, k, v)

    def test_rejects_4d_kv(self):
        op = _get_op()
        q = torch.randn(1, 4, 32, 128, device="npu", dtype=torch.bfloat16)
        k = torch.randn(1, 1, 32, 128, device="npu", dtype=torch.bfloat16)
        v = torch.randn(1, 1, 32, 128, device="npu", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="k/v 3-D"):
            op(q, k, v)

    def test_rejects_kv_length_mismatch(self):
        op = _get_op()
        q = torch.randn(1, 4, 32, 128, device="npu", dtype=torch.bfloat16)
        k = torch.randn(1, 32, 128, device="npu", dtype=torch.bfloat16)
        v = torch.randn(1, 64, 128, device="npu", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="key length mismatch"):
            op(q, k, v)
