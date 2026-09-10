# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the Ascend NPU deterministic token embedding.

Validates the same two orthogonal properties as the CUDA deterministic op,
but with a stronger correctness claim than the attention op: embedding is a
pure row gather (a bit copy, no arithmetic), so the Ascend output is
**bitwise identical** to the ``NativeEmbeddingOp`` PyTorch reference at every
dtype -- there is no reduction tolerance to calibrate.

1. **Correctness** - ``forward``/``forward_fp32`` match the PyTorch reference
   bitwise (``torch.equal``), and the deterministic sorted-segment backward
   reproduces the fixed-order duplicate-id sum bitwise in the gradient dtype.
2. **Batch-invariance** - a token's gathered row is bitwise identical
   regardless of batch size, batch position, or how many AI-core blocks were
   launched (each row is copied end-to-end by one block).
"""

import pytest
import torch

from rl_engine.kernels.ops.cuda.linear.embedding import _deterministic_embedding_grad_weight
from rl_engine.kernels.ops.pytorch.linear.embedding import NativeEmbeddingOp

_VOCAB = 128
_HIDDEN = 64

# Gradient tolerances from the gtest contract, "elementwise" op class.
_GRAD_ATOL = {
    torch.float32: 1.0e-5,
    torch.bfloat16: 2.0e-2,
    torch.float16: 1.0e-3,
}
_GRAD_RTOL = {
    torch.float32: 1.0e-5,
    torch.bfloat16: 1.6e-2,
    torch.float16: 1.0e-3,
}


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
        from rl_engine.kernels.ops.ascend.linear.embedding import _NPU_EXT_AVAILABLE, _C_npu
    except Exception:
        return False
    return _NPU_EXT_AVAILABLE and hasattr(_C_npu, "embedding_ascend")


requires_ascend = pytest.mark.skipif(
    not _ascend_kernel_available(),
    reason="embedding_ascend kernel not compiled "
    "(needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).",
)


def _get_op():
    from rl_engine.kernels.ops.ascend.linear.embedding import AscendEmbeddingOp

    return AscendEmbeddingOp()


def _make_inputs(shape, vocab=_VOCAB, hidden=_HIDDEN, dtype=torch.float32, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = torch.randn(vocab, hidden, dtype=dtype, generator=generator).to("npu")
    token_ids = torch.randint(0, vocab, shape, generator=generator).long().to("npu")
    return token_ids, weight


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@requires_ascend
class TestAscendEmbeddingCorrectness:
    def test_forward_matches_pytorch_reference_bitwise(self, dtype):
        """Ascend forward == NativeEmbeddingOp.forward, bitwise (pure gather)."""
        op = _get_op()
        token_ids, weight = _make_inputs((3, 5), dtype=dtype)
        out = op(token_ids, weight)
        ref = NativeEmbeddingOp().forward(token_ids, weight)
        assert out.dtype == dtype
        assert torch.equal(out, ref)

    def test_forward_matches_direct_indexing_bitwise(self, dtype):
        op = _get_op()
        token_ids, weight = _make_inputs((3, 5), dtype=dtype)
        out = op(token_ids, weight)
        assert torch.equal(out, weight[token_ids])

    def test_forward_fp32_matches_reference_bitwise(self, dtype):
        """Ascend forward_fp32 == NativeEmbeddingOp.forward_fp32, bitwise."""
        op = _get_op()
        token_ids, weight = _make_inputs((3, 5), dtype=dtype)
        out = op.forward_fp32(token_ids, weight)
        ref = NativeEmbeddingOp().forward_fp32(token_ids, weight)
        assert out.dtype == torch.float32
        assert torch.equal(out, ref)

    def test_output_shape_leading_dims(self, dtype):
        op = _get_op()
        token_ids, weight = _make_inputs((2, 4, 3), dtype=dtype)
        out = op(token_ids, weight)
        assert out.shape == (2, 4, 3, _HIDDEN)

    def test_backward_matches_fixed_order_sum_bitwise(self, dtype):
        """The sorted-segment dweight equals the input-order row sum, bitwise.

        The backward is the same deterministic formula the SM90 CUDA op uses
        (stable-sorted segments, fixed addition order), so this asserts the
        Ascend op reproduces that exact arithmetic on NPU.
        """
        op = _get_op()
        token_ids, weight = _make_inputs((3, 5), dtype=dtype)
        flat = token_ids.reshape(-1)
        flat[1::3] = flat[0]  # force duplicates of the first token id
        grad_out = torch.randn(3, 5, _HIDDEN, device="npu", dtype=dtype)

        weight_g = weight.clone().requires_grad_()
        op(flat.reshape(3, 5), weight_g).backward(grad_out)
        grad_asc = weight_g.grad

        grad_weight = _deterministic_embedding_grad_weight(
            flat,
            grad_out.reshape(flat.numel(), _HIDDEN),
            weight_shape=tuple(weight.shape),
            weight_dtype=dtype,
        )
        assert torch.equal(grad_asc, grad_weight)

    def test_backward_matches_native_reference(self, dtype):
        """vs the native op's backward at the elementwise gradient contract.

        Not bitwise by design: the deterministic formula accumulates
        duplicate-id rows in the grad dtype (one rounding per add) while the
        native backward accumulates in fp32, and the native reduction order
        is unspecified. Two duplicates keep the drift within the contract.
        """
        op = _get_op()
        token_ids, weight = _make_inputs((3, 5), dtype=dtype)
        flat = token_ids.reshape(-1)
        flat[1] = flat[0]  # a single duplicate exercises multi-row accumulation
        grad_out = torch.randn(3, 5, _HIDDEN, device="npu", dtype=dtype)

        weight_a = weight.clone().requires_grad_()
        op(flat.reshape(3, 5), weight_a).backward(grad_out)

        weight_n = weight.clone().requires_grad_()
        NativeEmbeddingOp().forward(flat.reshape(3, 5), weight_n).backward(grad_out)

        assert torch.allclose(
            weight_a.grad.float(),
            weight_n.grad.float(),
            atol=_GRAD_ATOL[dtype],
            rtol=_GRAD_RTOL[dtype],
        )

    def test_unused_rows_stay_zero(self, dtype):
        op = _get_op()
        token_ids, weight = _make_inputs((1, 2), dtype=dtype)
        grad_out = torch.randn(1, 2, _HIDDEN, device="npu", dtype=dtype)
        weight_g = weight.clone().requires_grad_()
        op(token_ids, weight_g).backward(grad_out)
        used = set(token_ids.reshape(-1).cpu().tolist())
        for row in range(_VOCAB):
            if row not in used:
                assert torch.equal(
                    weight_g.grad[row], torch.zeros(_HIDDEN, device="npu", dtype=dtype)
                )


# ---------------------------------------------------------------------------
# Input guards
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendEmbeddingGuards:
    def test_rejects_non_npu(self):
        op = _get_op()
        token_ids, weight = _make_inputs((2, 3))
        with pytest.raises(RuntimeError):
            op(token_ids.cpu(), weight.cpu())


# ---------------------------------------------------------------------------
# Batch invariance
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendEmbeddingBatchInvariance:
    def _run_row(self, batch, seq, dtype, pos, seed=7):
        """One fixed token embedded at position `pos` of a random batch."""
        op = _get_op()
        token_ids, weight = _make_inputs((batch, seq), dtype=dtype, seed=seed)
        out = op(token_ids, weight)
        return out[0, pos, :].clone()

    def test_batch_size_1_vs_n(self):
        dtype = torch.float16
        alone = self._run_row(1, 8, dtype, pos=0, seed=7)
        for batch in (2, 4, 8):
            in_batch = self._run_row(batch, 8, dtype, pos=0, seed=7)
            assert torch.equal(alone, in_batch), f"drift at batch_size={batch}"

    def test_different_positions_in_batch(self):
        # The same weight row gathered at every position of a batch must be
        # bitwise-identical regardless of where the token lands.
        dtype = torch.bfloat16
        op = _get_op()
        token_ids, weight = _make_inputs((2, 16), dtype=dtype, seed=11)
        fixed_id = token_ids[0, 0]
        token_ids[0, :] = fixed_id  # one token id repeated across positions
        out = op(token_ids, weight)
        ref = weight[fixed_id]
        for pos in range(16):
            assert torch.equal(out[0, pos, :], ref), f"drift at position={pos}"

    def test_block_striding(self):
        # 1024 tokens > MAX_BLOCKS (128): rows are strided across blocks, so
        # the copied bytes must not depend on block assignment. The same
        # (weight row, token id) gathered in a small run and in the strided
        # run must be bitwise-identical.
        dtype = torch.bfloat16
        op = _get_op()
        small_ids, small_weight = _make_inputs((1,), dtype=dtype, seed=3)
        small = op(small_ids, small_weight)
        big_ids, big_weight = _make_inputs((1024,), dtype=dtype, seed=4)
        big_ids[511] = small_ids[0]
        big_weight[:] = small_weight  # same table content
        big = op(big_ids, big_weight)
        assert torch.equal(big[511, :], small[0, :])

    def test_multi_tile_rows(self):
        # hidden > TILE_LENGTH would need a 4096+ column table; use a
        # multi-tile-equivalent via a large hidden with the tile loop.
        # (TILE_LENGTH = 4096; hidden = 12288 exercises 3 tiles per row.)
        dtype = torch.float16
        op = _get_op()
        generator = torch.Generator(device="cpu").manual_seed(9)
        weight = torch.randn(256, 12288, dtype=dtype, generator=generator).to("npu")
        token_ids = torch.randint(0, 256, (2, 3), generator=generator).long().to("npu")
        out = op(token_ids, weight)
        assert torch.equal(out, weight[token_ids])

    def test_repeated_runs_deterministic(self):
        dtype = torch.bfloat16
        token_ids, weight = _make_inputs((3, 5), dtype=dtype, seed=5)
        op = _get_op()
        first = op(token_ids, weight)
        for _ in range(3):
            again = op(token_ids, weight)
            assert torch.equal(first, again)


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendRegistryDispatch:
    def test_get_op_embedding(self):
        from rl_engine.kernels.registry import kernel_registry

        op = kernel_registry.get_op("embedding", device="npu")
        assert type(op).__name__ == "AscendEmbeddingOp"
