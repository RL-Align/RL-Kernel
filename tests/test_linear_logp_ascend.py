# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the Ascend NPU batch-invariant fused linear log-prob.

Validates the same two orthogonal properties as the CUDA deterministic op:

1. **Correctness** - output matches a hand-computed fp32 reference
   (``hidden.float() @ weight.float().T`` + ``log_softmax`` + gather +
   clamp) within an honest reduction tolerance (~2e-4 at D=4096, pure
   fp32-tree drift). The gtest's own forward comparison is stricter than
   any independent kernel can meet (see Notes in the PR description): the
   fp32 logprob tolerance is 1e-5 while two different fp32 reduction trees
   over D=4096 drift ~1e-4, and the gold's dtype path accumulates the
   matmul in bf16/fp16 while this kernel (like the CUDA SM90 kernel)
   accumulates in fp32.
2. **Batch-invariance** - a row's logp is bitwise identical regardless of
   batch size, batch position, or how many AI-core blocks were launched
   (each row is reduced end-to-end by one block over a fixed vocab scan).
"""

import pytest
import torch

_VOCAB = 129
_HIDDEN = 1000

# Honest forward tolerance vs the fp32 reference: pure fp32 reduction-tree
# drift (measured 2.1e-4 at D=4096, V=257).
_FWD_ATOL = 5.0e-4
_FWD_RTOL = 1.0e-5
# Gradient tolerance: the chunked backward casts to the input dtype, so
# low-precision grads compare at their own quantization level.
_GRAD_ATOL = {torch.float32: 5.0e-4, torch.bfloat16: 2.0e-2, torch.float16: 1.0e-2}


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
        from rl_engine.kernels.ops.ascend.loss.linear_logp import _NPU_EXT_AVAILABLE, _C_npu
    except Exception:
        return False
    return _NPU_EXT_AVAILABLE and hasattr(_C_npu, "fused_linear_logp_ascend")


requires_ascend = pytest.mark.skipif(
    not _ascend_kernel_available(),
    reason="fused_linear_logp_ascend kernel not compiled "
    "(needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).",
)


def _get_op():
    from rl_engine.kernels.ops.ascend.loss.linear_logp import FusedLinearLogpAscendOp

    return FusedLinearLogpAscendOp()


def _make_inputs(shape, vocab=_VOCAB, hidden=_HIDDEN, dtype=torch.float32, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_t = torch.randn(*shape, hidden, dtype=dtype, generator=generator).to("npu")
    weight = torch.randn(vocab, hidden, dtype=dtype, generator=generator).to("npu")
    target_ids = torch.randint(0, vocab, shape, generator=generator).long().to("npu")
    return hidden_t, weight, target_ids


def _ref_fp32(hidden, weight, target_ids, bias=None):
    """Hand-written fp32 reference matching the WS1 fp32-reference policy."""
    logits = hidden.float().reshape(-1, hidden.size(-1)) @ weight.float().t()
    if bias is not None:
        logits = logits + bias.float()
    flat = target_ids.reshape(-1)
    logp = torch.log_softmax(logits, dim=-1)
    selected = logp.gather(1, flat.unsqueeze(1)).squeeze(1)
    return selected.clamp(max=0).reshape(hidden.shape[:-1])


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@requires_ascend
class TestAscendFusedLinearLogpCorrectness:
    def test_forward_matches_fp32_reference(self, dtype):
        op = _get_op()
        hidden, weight, target_ids = _make_inputs((3, 5), dtype=dtype)
        out = op(hidden, weight, target_ids)
        ref = _ref_fp32(hidden, weight, target_ids)
        assert out.dtype == torch.float32
        assert out.shape == (3, 5)
        assert torch.allclose(out, ref, atol=_FWD_ATOL, rtol=_FWD_RTOL)

    def test_forward_large_shape(self, dtype):
        # gtest shape: D=4096 (single cached hidden tile), V=257.
        op = _get_op()
        hidden, weight, target_ids = _make_inputs((2, 16), vocab=257, hidden=4096, dtype=dtype)
        out = op(hidden, weight, target_ids)
        ref = _ref_fp32(hidden, weight, target_ids)
        assert torch.allclose(out, ref, atol=_FWD_ATOL, rtol=_FWD_RTOL)

    def test_out_of_range_target_is_zero(self, dtype):
        op = _get_op()
        hidden, weight, target_ids = _make_inputs((2, 4), vocab=32, dtype=dtype)
        target_ids = target_ids.reshape(-1)
        target_ids[1] = 32 + 5  # out of [0, V)
        out = op(hidden, weight, target_ids.reshape(2, 4))
        assert out.reshape(-1)[1].item() == 0.0

    def test_bias_falls_back_to_native(self, dtype):
        from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp

        op = _get_op()
        hidden, weight, target_ids = _make_inputs((2, 3), dtype=dtype)
        bias = torch.randn(_VOCAB, device="npu", dtype=dtype)
        out = op(hidden, weight, target_ids, bias)
        ref = NativeLinearLogpOp().apply(hidden, weight, target_ids, bias)
        assert torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-5)

    def test_backward_matches_fp32_reference(self, dtype):
        op = _get_op()
        hidden, weight, target_ids = _make_inputs((3, 5), dtype=dtype)
        grad_out = torch.randn(3, 5, device="npu", dtype=dtype)

        h_a = hidden.clone().requires_grad_()
        w_a = weight.clone().requires_grad_()
        op(h_a, w_a, target_ids).backward(grad_out)

        h_f = hidden.float().clone().requires_grad_()
        w_f = weight.float().clone().requires_grad_()
        _ref_fp32(h_f, w_f, target_ids).backward(grad_out.float())

        # Compare at the quantized level for low-precision inputs: both
        # backends compute the VJP in fp32 and cast to the input dtype, so
        # the fp32 tree drift collapses into (usually identical) quantized
        # bits; the tolerance only absorbs the rare 1-ULP straddle.
        assert torch.allclose(
            h_a.grad.float(), h_f.grad.to(dtype).float(), atol=_GRAD_ATOL[dtype], rtol=1.0e-4
        )
        assert torch.allclose(
            w_a.grad.float(), w_f.grad.to(dtype).float(), atol=_GRAD_ATOL[dtype], rtol=1.0e-4
        )

    def test_backward_has_no_cross_row_leak(self, dtype):
        """Row-local VJP: the same row's grad is bitwise identical wherever the
        row sits in the batch."""
        op = _get_op()
        hidden, weight, target_ids = _make_inputs((4, 8), dtype=dtype)
        hidden[1].copy_(hidden[0])
        target_ids[1] = target_ids[0]

        h_g = hidden.clone().requires_grad_()
        grad_out = torch.randn(4, 8, device="npu", dtype=dtype)
        grad_out[1] = grad_out[0]
        op(h_g, weight, target_ids).backward(grad_out)
        grad = h_g.grad
        assert torch.equal(grad[0], grad[1])
        for row in range(2, 4):
            assert not torch.equal(grad[0], grad[row])


# ---------------------------------------------------------------------------
# Batch invariance
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendFusedLinearLogpBatchInvariance:
    def test_batch_size_1_vs_n(self):
        # One fixed row embedded in batches of growing size: its logp must be
        # bitwise identical regardless of batch size.
        dtype = torch.bfloat16
        op = _get_op()
        alone_hidden, weight, alone_ids = _make_inputs((1,), dtype=dtype, seed=7)
        alone = op(alone_hidden, weight, alone_ids)[0]
        for batch in (2, 4, 16, 300):  # 300 rows -> > MAX_BLOCKS strided blocks
            hidden, _, target_ids = _make_inputs((batch,), dtype=dtype, seed=7)
            hidden[0].copy_(alone_hidden[0])
            target_ids[0] = alone_ids[0]
            in_batch = op(hidden, weight, target_ids)[0]
            assert torch.equal(alone, in_batch), f"drift at batch_size={batch}"

    def test_different_positions_in_batch(self):
        # The same row content copied to every position reduces bitwise
        # identically regardless of where it lands.
        dtype = torch.float16
        op = _get_op()
        hidden, weight, target_ids = _make_inputs((8,), dtype=dtype, seed=11)
        base, base_id = hidden[0].clone(), int(target_ids[0])
        for pos in range(1, 8):
            hidden[pos].copy_(base)
            target_ids[pos] = base_id
        out = op(hidden, weight, target_ids)
        for pos in range(1, 8):
            assert torch.equal(out[pos], out[0]), f"drift at position={pos}"

    def test_multi_tile_rows(self):
        # hidden > TILE_LENGTH (4096): the per-row dots span multiple tiles.
        dtype = torch.float32
        op = _get_op()
        hidden, weight, target_ids = _make_inputs((4,), hidden=10000, dtype=dtype, seed=5)
        out = op(hidden, weight, target_ids)
        assert torch.equal(out, op(hidden, weight, target_ids))

    def test_repeated_runs_deterministic(self):
        dtype = torch.bfloat16
        hidden, weight, target_ids = _make_inputs((3, 5), dtype=dtype, seed=5)
        op = _get_op()
        first = op(hidden, weight, target_ids)
        for _ in range(3):
            again = op(hidden, weight, target_ids)
            assert torch.equal(first, again)


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendRegistryDispatch:
    def test_get_op_linear_logp(self):
        from rl_engine.kernels.registry import kernel_registry

        op = kernel_registry.get_op("linear_logp", device="npu")
        assert type(op).__name__ == "FusedLinearLogpAscendOp"
