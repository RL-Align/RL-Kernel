# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the Ascend NPU batch-invariant LM-head projection.

Validates the same two orthogonal properties as the CUDA deterministic op:

1. **Correctness** - output matches the ``NativeLMHeadOp.forward_fp32`` ground
   truth within the reduction contract tolerance. The Ascend C kernel mirrors
   the SM90 CUDA kernel's structure (one output element per block, full
   hidden-dimension fp32 reduction over a fixed tile order); the hardware
   reduction trees differ from CUDA's (and from torch.mv's), so the
   comparison is tolerance-based.
2. **Batch-invariance** - a row's logits are bitwise identical regardless of
   batch size, batch position, or how many AI-core blocks were launched
   (each element is reduced end-to-end by one block; no Split-K merge).
"""

import pytest
import torch

from rl_engine.kernels.ops.pytorch.linear.lm_head import NativeLMHeadOp

# Accuracy tolerances from the gtest contract, "reduction" op class.
_ATOL = {
    torch.float32: 1.0e-4,
    torch.bfloat16: 5.0e-2,
    torch.float16: 1.0e-3,
}
_RTOL = {
    torch.float32: 1.0e-4,
    torch.bfloat16: 2.0e-2,
    torch.float16: 1.0e-3,
}
# Gradient tolerances from the gtest contract, "gradient_accuracy" reduction row.
_GRAD_ATOL = {
    torch.float32: 1.0e-4,
    torch.bfloat16: 1.0e-1,
    torch.float16: 1.0e-3,
}
_GRAD_RTOL = {
    torch.float32: 1.0e-4,
    torch.bfloat16: 2.0e-2,
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
        from rl_engine.kernels.ops.ascend.linear.lm_head import _NPU_EXT_AVAILABLE, _C_npu
    except Exception:
        return False
    return _NPU_EXT_AVAILABLE and hasattr(_C_npu, "lm_head_ascend")


requires_ascend = pytest.mark.skipif(
    not _ascend_kernel_available(),
    reason="lm_head_ascend kernel not compiled "
    "(needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).",
)


def _get_op():
    from rl_engine.kernels.ops.ascend.linear.lm_head import AscendLMHeadOp

    return AscendLMHeadOp()


def _make_inputs(shape, vocab, hidden, dtype, seed=0, with_bias=False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_t = torch.randn(*shape, hidden, dtype=dtype, generator=generator).to("npu")
    weight = torch.randn(vocab, hidden, dtype=dtype, generator=generator).to("npu")
    bias = torch.randn(vocab, dtype=dtype, generator=generator).to("npu") if with_bias else None
    return hidden_t, weight, bias


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@requires_ascend
class TestAscendLMHeadCorrectness:
    def test_forward_matches_pytorch_reference(self, dtype):
        op = _get_op()
        hidden, weight, _ = _make_inputs((3, 5), 129, 1000, dtype)
        out = op(hidden, weight)
        ref = NativeLMHeadOp().forward_fp32(hidden, weight)
        assert out.dtype == dtype
        assert out.shape == (3, 5, 129)
        assert torch.allclose(out.float(), ref.float(), atol=_ATOL[dtype], rtol=_RTOL[dtype])

    def test_forward_fp32_matches_reference(self, dtype):
        op = _get_op()
        hidden, weight, _ = _make_inputs((3, 5), 129, 1000, dtype)
        out = op.forward_fp32(hidden, weight)
        ref = NativeLMHeadOp().forward_fp32(hidden, weight)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=_ATOL[dtype], rtol=_RTOL[dtype])

    def test_forward_with_bias(self, dtype):
        op = _get_op()
        hidden, weight, bias = _make_inputs((3, 5), 129, 1000, dtype, with_bias=True)
        out = op(hidden, weight, bias=bias)
        ref = NativeLMHeadOp().forward_fp32(hidden, weight, bias=bias)
        assert torch.allclose(out.float(), ref.float(), atol=_ATOL[dtype], rtol=_RTOL[dtype])

    def test_output_shape_leading_dims(self, dtype):
        op = _get_op()
        hidden, weight, _ = _make_inputs((2, 4, 3), 17, 64, dtype)
        out = op(hidden, weight)
        assert out.shape == (2, 4, 3, 17)

    def test_backward_matches_native_reference(self, dtype):
        # The native reference runs the fp32 path: torch.mv on NPU rejects
        # bf16, and the fp32 path keeps both VJPs in the same accumulation
        # dtype so the comparison isolates the matmul-tree drift.
        op = _get_op()
        hidden, weight, _ = _make_inputs((3, 5), 129, 1000, dtype)
        grad_out = torch.randn(3, 5, 129, device="npu", dtype=dtype)

        h_a = hidden.clone().requires_grad_()
        w_a = weight.clone().requires_grad_()
        op(h_a, w_a).backward(grad_out)

        h_n = hidden.clone().requires_grad_()
        w_n = weight.clone().requires_grad_()
        NativeLMHeadOp().forward_fp32(h_n, w_n).backward(grad_out)

        assert torch.allclose(
            h_a.grad.float(), h_n.grad.float(), atol=_GRAD_ATOL[dtype], rtol=_GRAD_RTOL[dtype]
        )
        assert torch.allclose(
            w_a.grad.float(), w_n.grad.float(), atol=_GRAD_ATOL[dtype], rtol=_GRAD_RTOL[dtype]
        )

    def test_backward_with_bias(self, dtype):
        op = _get_op()
        hidden, weight, bias = _make_inputs((3, 5), 129, 1000, dtype, with_bias=True)
        grad_out = torch.randn(3, 5, 129, device="npu", dtype=dtype)

        h_a = hidden.clone().requires_grad_()
        w_a = weight.clone().requires_grad_()
        b_a = bias.clone().requires_grad_()
        op(h_a, w_a, bias=b_a).backward(grad_out)
        assert b_a.grad is not None
        assert b_a.grad.shape == (129,)
        assert torch.isfinite(b_a.grad).all()

        b_n = bias.clone().requires_grad_()
        NativeLMHeadOp().forward_fp32(hidden, weight, bias=b_n).backward(grad_out)
        assert torch.allclose(
            b_a.grad.float(), b_n.grad.float(), atol=_GRAD_ATOL[dtype], rtol=_GRAD_RTOL[dtype]
        )


# ---------------------------------------------------------------------------
# Batch invariance
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendLMHeadBatchInvariance:
    def test_batch_size_1_vs_n(self):
        # One fixed hidden row embedded in batches of growing size: its logits
        # must be bitwise identical regardless of batch size.
        dtype = torch.bfloat16
        op = _get_op()
        alone_hidden, weight, _ = _make_inputs((1,), 129, 1000, dtype, seed=7)
        alone = op(alone_hidden, weight)[0]
        for batch in (2, 4, 16, 300):  # 300 rows -> > MAX_BLOCKS strided blocks
            hidden, _, _ = _make_inputs((batch,), 129, 1000, dtype, seed=7)
            hidden[0].copy_(alone_hidden[0])
            in_batch = op(hidden, weight)[0]  # same weight table throughout
            assert torch.equal(alone, in_batch), f"drift at batch_size={batch}"

    def test_different_positions_in_batch(self):
        # The same row content copied to every position projects bitwise
        # identically regardless of where it lands.
        dtype = torch.float16
        op = _get_op()
        hidden, weight, _ = _make_inputs((8,), 129, 1000, dtype, seed=11)
        base = hidden[0].clone()
        for pos in range(1, 8):
            hidden[pos].copy_(base)
        out = op(hidden, weight)
        for pos in range(1, 8):
            assert torch.equal(out[pos], out[0]), f"drift at position={pos}"

    def test_multi_tile_rows(self):
        # hidden > TILE_LENGTH (4096): the reduction spans multiple tiles.
        dtype = torch.float32
        op = _get_op()
        hidden, weight, _ = _make_inputs((4,), 129, 10000, dtype, seed=5)
        out = op(hidden, weight)
        assert torch.equal(out, op(hidden, weight))

    def test_repeated_runs_deterministic(self):
        dtype = torch.bfloat16
        hidden, weight, _ = _make_inputs((3, 5), 129, 1000, dtype, seed=5)
        op = _get_op()
        first = op(hidden, weight)
        for _ in range(3):
            again = op(hidden, weight)
            assert torch.equal(first, again)


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendRegistryDispatch:
    def test_get_op_lm_head(self):
        from rl_engine.kernels.registry import kernel_registry

        op = kernel_registry.get_op("lm_head", device="npu")
        assert type(op).__name__ == "AscendLMHeadOp"
