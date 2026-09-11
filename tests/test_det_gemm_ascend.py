# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the Ascend NPU batch-invariant deterministic GEMM (WS1 #146).

Validates the same properties as the CUDA deterministic op:
1. **Correctness** - output matches the canonical FP32-leaf / BF16-node
   midpoint tree reference within the reduction tolerances.
2. **Batch-invariance** - a row's output (and gradient) is bitwise identical
   regardless of batch size, batch position, or how many AI-core blocks were
   launched (every output row-tile is reduced end-to-end by one block with a
   fixed leaf order; no split-K merge exists).
3. **TP-shard invariance** - contiguous half-K shards combined with one BF16
   add reproduce the full GEMM bitwise (the tree's "one child" property).
"""

import pytest
import torch

from rl_engine.kernels.ops.ascend.matmul.det_gemm import DetGemmAscendOp

# Accuracy tolerance from the gtest contract, "reduction" op class, bf16.
_ATOL = 5.0e-2
_RTOL = 2.0e-2

_K_TREE_LEAF = 32


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
        from rl_engine.kernels.ops.ascend.matmul.det_gemm import (
            _NPU_EXT_AVAILABLE,
            _C_npu,
        )
    except Exception:
        return False
    return _NPU_EXT_AVAILABLE and hasattr(_C_npu, "det_gemm_ascend_fwd")


requires_ascend = pytest.mark.skipif(
    not _ascend_kernel_available(),
    reason="det_gemm_ascend kernel not compiled "
    "(needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).",
)


def _get_op():
    return DetGemmAscendOp()


def _rand(*shape, seed=0):
    # Independent generator per call: batch size must not shift the operand
    # content (a shared generator would make b[0] differ between batch sizes,
    # breaking the batch-invariance comparisons below).
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=torch.bfloat16).to("npu")


def _k_tree_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Canonical FP32-leaf / BF16-node midpoint tree (the CUDA/Triton reference)."""

    a = a.detach().contiguous()
    b = b.detach().contiguous()

    def reduce_range(lo: int, hi: int) -> torch.Tensor:
        if hi - lo <= _K_TREE_LEAF:
            return (a[:, lo:hi].float() @ b[lo:hi, :].float()).to(torch.bfloat16)
        midpoint = lo + (hi - lo) // 2
        return reduce_range(lo, midpoint) + reduce_range(midpoint, hi)

    return reduce_range(0, a.size(1))


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendDetGemmCorrectness:
    @pytest.mark.parametrize(
        "shape",
        [
            (128, 128, 128),  # aligned, single-tile
            (128, 2048, 2048),
            (31, 70, 65),  # ragged scalar-fallback shape
            (1, 32, 32),  # single leaf, no tree merges
            (4, 12288, 64),  # non-power-of-two midpoint tree (Qwen down-proj K)
        ],
    )
    def test_forward_matches_tree_reference(self, shape):
        m, k, n = shape
        op = _get_op()
        a, b = _rand(m, k, seed=3), _rand(k, n, seed=4)
        out = op(a, b)
        ref = _k_tree_gemm(a, b)
        assert out.dtype == torch.bfloat16
        assert tuple(out.shape) == (m, n)
        torch.testing.assert_close(
            out.float(), ref.float(), atol=_ATOL, rtol=_RTOL
        )

    @pytest.mark.parametrize(
        "shape",
        [
            (128, 128, 128),
            (31, 70, 65),
        ],
    )
    def test_forward_fp32_matches_tree_reference(self, shape):
        m, k, n = shape
        op = _get_op()
        a, b = _rand(m, k, seed=5), _rand(k, n, seed=6)
        out = op.forward_fp32(a, b)
        ref = _k_tree_gemm(a, b)
        assert out.dtype == torch.float32
        # FP32 output carries the exact BF16-rounded root; loose bf16-scale
        # tolerance suffices against the reference.
        torch.testing.assert_close(out, ref.float(), atol=_ATOL, rtol=_RTOL)

    @pytest.mark.parametrize("shape", [(128, 128, 128), (31, 70, 65)])
    def test_rhs_transposed_layout_matches_forward_bitwise(self, shape):
        m, k, n = shape
        op = _get_op()
        a = _rand(m, k, seed=7)
        bt = _rand(n, k, seed=8)
        expected = op(a, bt.t().contiguous())
        actual = _DetGemmAscendFn_rhs_transposed(a, bt)
        assert actual.is_contiguous()
        assert tuple(actual.shape) == (m, n)
        assert torch.equal(actual, expected)


def _DetGemmAscendFn_rhs_transposed(a, bt):
    from rl_engine.kernels.ops.ascend.matmul.det_gemm import _C_npu

    return _C_npu.det_gemm_ascend_fwd_rhs_transposed(a, bt)


# ---------------------------------------------------------------------------
# Batch invariance
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendDetGemmInvariance:
    @pytest.mark.parametrize(
        "shape",
        [
            (4096, 4096, 12288),  # qkv
            (4096, 4096, 4096),  # o_proj
            (4096, 4096, 14336),  # mlp_up
            (4096, 14336, 4096),  # mlp_dn
            (4096, 4096, 32000),  # lm_head
        ],
    )
    def test_forward_batch_invariance(self, shape):
        # A row's output must not change when other rows join the batch.
        _, k, n = shape
        op = _get_op()
        b = _rand(k, n, seed=0)
        row = _rand(1, k, seed=1)
        out1 = op(row, b)
        big = _rand(64, k, seed=2)
        big[0] = row[0]
        outN = op(big, b)
        assert torch.equal(out1[0], outN[0])

    def test_forward_padding_invariance(self):
        # Padding rows must not affect valid rows' output.
        op = _get_op()
        m, k, n = 100, 4096, 4096
        a, b = _rand(m, k, seed=3), _rand(k, n, seed=4)
        base = op(a, b)
        a_pad = torch.cat([a, _rand(28, k, seed=5)], dim=0)
        padded = op(a_pad, b)
        assert torch.equal(base, padded[:m])

    def test_backward_batch_invariance(self):
        # dA for a row must be invariant to the surrounding batch.
        op = _get_op()
        k, n = 2048, 2048
        b = _rand(k, n, seed=6)
        row = _rand(1, k, seed=7).requires_grad_(True)
        op(row, b).sum().backward()
        g1 = row.grad.clone()
        big = _rand(256, k, seed=8)
        big[0] = row.detach()[0]
        big.requires_grad_(True)
        op(big, b).sum().backward()
        assert torch.equal(g1[0], big.grad[0])

    def test_tp2_contiguous_k_shards_match_full_bitwise(self):
        # The GEMM tree and the TP collective rank tree must be the same graph:
        # TP=2 (a+b) over contiguous half-K shards matches TP=1 bitwise.
        op = _get_op()
        m, k, n = 4, 12288, 64
        a, b = _rand(m, k, seed=9), _rand(k, n, seed=10)
        half = k // 2
        full = op(a, b)
        part0 = op(a[:, :half].contiguous(), b[:half].contiguous())
        part1 = op(a[:, half:].contiguous(), b[half:].contiguous())
        sharded = part0 + part1
        assert torch.equal(full, sharded), (
            f"full GEMM differed from TP=2 shards at "
            f"{int((full != sharded).sum().item())} elements"
        )


# ---------------------------------------------------------------------------
# Backward correctness
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendDetGemmBackward:
    def test_backward_matches_tree_reference(self):
        op = _get_op()
        m, k, n = 64, 1024, 1024
        a = _rand(m, k, seed=11).requires_grad_(True)
        b = _rand(k, n, seed=12).requires_grad_(True)
        g = _rand(m, n, seed=13)
        op(a, b).backward(g)
        expected_da = _k_tree_gemm(g, b.detach().t().contiguous())
        expected_db = _k_tree_gemm(a.detach().t().contiguous(), g)
        torch.testing.assert_close(
            a.grad.float(), expected_da.float(), atol=_ATOL, rtol=_RTOL
        )
        torch.testing.assert_close(
            b.grad.float(), expected_db.float(), atol=_ATOL, rtol=_RTOL
        )

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 128, 128),  # short-K weight gradient
            (8, 128, 128),
            (128, 128, 128),
            (128, 96, 64),
            (31, 70, 65),
        ],
    )
    def test_transposed_db_is_canonical_contiguous_and_matches_db(self, shape):
        from rl_engine.kernels.ops.ascend.matmul.det_gemm import _C_npu

        tokens, in_features, out_features = shape
        a = _rand(tokens, in_features, seed=14)
        dc = _rand(tokens, out_features, seed=15)

        expected = _C_npu.det_gemm_ascend_db(a, dc).t().contiguous()
        actual = _C_npu.det_gemm_ascend_db_transposed(a, dc)

        assert tuple(actual.shape) == (out_features, in_features)
        assert tuple(actual.stride()) == (in_features, 1)
        assert actual.is_contiguous()
        assert torch.equal(actual, expected)

    def test_da_physical_transpose_contract_matches_fwd_bitwise(self):
        from rl_engine.kernels.ops.ascend.matmul.det_gemm import _C_npu

        op = _get_op()
        m, k, n = 128, 128, 128
        dc = _rand(m, n, seed=16)
        b = _rand(k, n, seed=17)

        expected = op(dc, b.t().contiguous())
        actual = _C_npu.det_gemm_ascend_da(dc, b)

        assert torch.equal(actual, expected)
