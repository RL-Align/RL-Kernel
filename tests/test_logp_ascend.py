# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the Ascend NPU batch-invariant fused selected-token logp.

Validates the same two orthogonal properties as the CUDA deterministic op:

1. **Correctness** - output matches the ``NativeLogpOp.forward_fp32`` ground
   truth within the logprob contract tolerance. The Ascend C kernel mirrors
   the CUDA deterministic kernel's two-pass (row max, then sum-exp) fp32
   reduction with a fixed tile order; the hardware reduction trees differ
   from CUDA's, so the comparison is tolerance-based (fp32 drift ~1e-7).
2. **Batch-invariance** - a row's logp is bitwise identical regardless of
   batch size, batch position, or how many AI-core blocks were launched
   (each row is reduced end-to-end by one block; no split-K merge exists).
"""

import pytest
import torch

from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp

# Accuracy tolerances from the gtest contract, "logprob" op class.
_ATOL = {
    torch.float32: 1.0e-5,
    torch.bfloat16: 6.0e-2,
    torch.float16: 5.0e-3,
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
        from rl_engine.kernels.ops.ascend.loss.logp import _NPU_EXT_AVAILABLE, _C_npu
    except Exception:
        return False
    return _NPU_EXT_AVAILABLE and hasattr(_C_npu, "fused_logp_ascend")


requires_ascend = pytest.mark.skipif(
    not _ascend_kernel_available(),
    reason="fused_logp_ascend kernel not compiled "
    "(needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).",
)


def _get_op():
    from rl_engine.kernels.ops.ascend.loss.logp import FusedLogpAscendOp

    return FusedLogpAscendOp()


def _make_inputs(shape, vocab, dtype, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(*shape, vocab, dtype=dtype, generator=generator).to("npu")
    token_ids = torch.randint(0, vocab, shape, generator=generator).long().to("npu")
    return logits, token_ids


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@requires_ascend
class TestAscendFusedLogpCorrectness:
    def test_forward_matches_pytorch_reference(self, dtype):
        op = _get_op()
        logits, token_ids = _make_inputs((3, 5), 257, dtype)
        out = op(logits, token_ids)
        ref = NativeLogpOp().forward_fp32(logits, token_ids)
        assert out.dtype == torch.float32  # matches DeterministicLogpCUDAOp's contract
        assert out.shape == (3, 5)
        assert torch.allclose(out.float(), ref, atol=_ATOL[dtype], rtol=0.0)

    def test_apply_fp32_matches_reference(self, dtype):
        op = _get_op()
        logits, token_ids = _make_inputs((3, 5), 257, dtype)
        out = op.apply_fp32(logits, token_ids)
        ref = NativeLogpOp().forward_fp32(logits, token_ids)
        assert torch.allclose(out.float(), ref, atol=_ATOL[dtype], rtol=0.0)

    def test_out_of_range_target_is_zero(self, dtype):
        op = _get_op()
        logits, token_ids = _make_inputs((2, 4), 32, dtype)
        token_ids = token_ids.reshape(-1)
        token_ids[1] = 32 + 5  # out of [0, V)
        out = op(logits, token_ids.reshape(2, 4))
        assert out.reshape(-1)[1].item() == 0.0

    def test_backward_matches_native_reference(self, dtype):
        op = _get_op()
        logits, token_ids = _make_inputs((3, 5), 257, dtype)

        logits_a = logits.clone().requires_grad_()
        op(logits_a, token_ids).backward(torch.ones(3, 5, device="npu", dtype=dtype))

        logits_n = logits.clone().requires_grad_()
        NativeLogpOp()(logits_n, token_ids).backward(torch.ones(3, 5, device="npu", dtype=dtype))

        assert torch.allclose(
            logits_a.grad.float(), logits_n.grad.float(), atol=1.0e-4, rtol=1.0e-4
        )

    def test_backward_has_no_cross_row_leak(self, dtype):
        """Row-local VJP: the same row's grad is bitwise identical wherever the
        row sits in the batch."""
        op = _get_op()
        logits, token_ids = _make_inputs((4, 8), 257, dtype)
        logits[1].copy_(logits[0])
        token_ids[1] = token_ids[0]

        logits_g = logits.clone().requires_grad_()
        grad_out = torch.randn(4, 8, device="npu", dtype=dtype)
        grad_out[1] = grad_out[0]  # identical (logits, target, dy) triples
        op(logits_g, token_ids).backward(grad_out)
        grad = logits_g.grad
        # Rows 0 and 1 saw identical inputs, so their VJPs must be bitwise
        # identical; rows 2/3 stay independent.
        assert torch.equal(grad[0], grad[1])
        for row in range(2, 4):
            assert not torch.equal(grad[0], grad[row])

# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendFusedLogpFallback:
    def test_rejects_non_npu_falls_back_to_native(self):
        op = _get_op()
        logits, token_ids = _make_inputs((2, 3), 17, torch.float32)
        out = op(logits.cpu(), token_ids.cpu())
        ref = NativeLogpOp()(logits.cpu(), token_ids.cpu())
        assert torch.equal(out, ref)


# ---------------------------------------------------------------------------
# Batch invariance
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendFusedLogpBatchInvariance:
    def _run_row(self, batch, vocab, dtype, pos, seed=7):
        """One fixed row embedded at position `pos` of a random batch."""
        op = _get_op()
        logits, token_ids = _make_inputs((batch,), vocab, dtype, seed=seed)
        out = op(logits, token_ids)
        return out[pos].clone()

    def test_batch_size_1_vs_n(self):
        # One fixed (row, target) pair embedded in batches of growing size:
        # its logp must be bitwise identical regardless of batch size.
        dtype = torch.bfloat16
        op = _get_op()
        alone_logits, alone_ids = _make_inputs((1,), 257, dtype, seed=7)
        alone = op(alone_logits, alone_ids)[0]
        for batch in (2, 4, 16, 300):  # 300 > MAX_BLOCKS -> strided blocks
            logits, token_ids = _make_inputs((batch,), 257, dtype, seed=7)
            logits[0].copy_(alone_logits[0])
            token_ids[0] = alone_ids[0]
            in_batch = op(logits, token_ids)[0]
            assert torch.equal(alone, in_batch), f"drift at batch_size={batch}"

    def test_different_positions_in_batch(self):
        # The same row content copied to every position reduces bitwise
        # identically regardless of where it lands.
        dtype = torch.float16
        op = _get_op()
        logits, token_ids = _make_inputs((8,), 257, dtype, seed=11)
        base, base_id = logits[0].clone(), int(token_ids[0])
        for pos in range(1, 8):
            logits[pos].copy_(base)
            token_ids[pos] = base_id
        out = op(logits, token_ids)
        for pos in range(1, 8):
            assert torch.equal(out[pos], out[0]), f"drift at position={pos}"

    def test_block_striding(self):
        # 300 rows > MAX_BLOCKS (128): rows are strided across blocks, so
        # numerics must not depend on block assignment.
        dtype = torch.bfloat16
        op = _get_op()
        logits, token_ids = _make_inputs((300,), 257, dtype, seed=13)
        out = op(logits, token_ids)
        again = op(logits, token_ids)
        assert torch.equal(out, again)

    def test_multi_tile_rows(self):
        # vocab > TILE_LENGTH (4096): rows span multiple fixed-order tiles.
        dtype = torch.float32
        op = _get_op()
        logits, token_ids = _make_inputs((4,), 10000, dtype, seed=5)
        out = op(logits, token_ids)
        assert torch.equal(out, op(logits, token_ids))

    def test_repeated_runs_deterministic(self):
        dtype = torch.bfloat16
        logits, token_ids = _make_inputs((3, 5), 257, dtype, seed=5)
        op = _get_op()
        first = op(logits, token_ids)
        for _ in range(3):
            again = op(logits, token_ids)
            assert torch.equal(first, again)


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendRegistryDispatch:
    def test_get_op_logp(self):
        from rl_engine.kernels.registry import kernel_registry

        op = kernel_registry.get_op("logp", device="npu")
        assert type(op).__name__ == "FusedLogpAscendOp"
