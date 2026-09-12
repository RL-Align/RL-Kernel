# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the Ascend C SiLU kernel (WS1 #266 C5/C8, ascend_bf16 chain node).

SiLU is a required C2 chain node, so the Ascend profile needs its own kernel
rather than borrowing SwiGLU with a ones operand. The properties checked are
the ones the contract judges:

1. **Accuracy** - forward and backward match the FP32 PyTorch reference within
   the elementwise tolerances.
2. **Batch invariance** - an element's value and gradient are bitwise identical
   regardless of tensor size, its position, or how many AI-core blocks ran.
3. **Consistency with SwiGLU** - ``silu(x)`` equals ``swiglu(x, ones)`` bitwise,
   since both evaluate the same FP32 sigmoid sequence.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.activation.swiglu import NativeSiLUOp

# Elementwise tolerances from the gtest contract, bf16.
_ATOL = 5.0e-2
_RTOL = 2.0e-2

# Deliberately spans tile boundaries: TILE_LENGTH is 2048 and MAX_BLOCKS is 32,
# so 2047/2048/2049 and a size beyond one full strided sweep exercise the tail
# path, the exact-tile path, and the multi-pass loop.
_SIZES = (1, 31, 32, 2047, 2048, 2049, 4096, 70000)


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
        from rl_engine.kernels.ops.ascend.activation.silu import _C_npu
    except Exception:
        return False
    return _C_npu is not None and hasattr(_C_npu, "silu_forward")


requires_ascend = pytest.mark.skipif(
    not _ascend_kernel_available(),
    reason="silu Ascend kernel not compiled "
    "(needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).",
)


def _get_op():
    from rl_engine.kernels.ops.ascend.activation.silu import SiLUAscendOp

    return SiLUAscendOp()


def _rand(*shape, seed=0, dtype=torch.bfloat16):
    # Independent generator per call so a different tensor size cannot shift
    # the content of the leading elements the invariance checks compare.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=dtype).to("npu")


@requires_ascend
class TestAscendSiLUCorrectness:
    @pytest.mark.parametrize("n", _SIZES)
    def test_forward_matches_fp32_reference(self, n):
        x = _rand(n)
        out = _get_op().forward(x)
        expected = NativeSiLUOp()(x.float().cpu()).to(torch.bfloat16)
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out.cpu(), expected, atol=_ATOL, rtol=_RTOL)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_supported_dtypes_round_trip(self, dtype):
        x = _rand(4096, dtype=dtype)
        out = _get_op().forward(x)
        assert out.dtype == dtype
        expected = NativeSiLUOp()(x.float().cpu()).to(dtype)
        torch.testing.assert_close(out.cpu(), expected, atol=_ATOL, rtol=_RTOL)

    def test_forward_fp32_returns_fp32(self):
        x = _rand(1024)
        out = _get_op().forward_fp32(x)
        assert out.dtype == torch.float32

    def test_empty_tensor_is_supported(self):
        out = _get_op().forward(_rand(0))
        assert out.numel() == 0

    def test_multi_dimensional_shapes_are_preserved(self):
        x = _rand(3, 17, 128)
        assert _get_op().forward(x).shape == x.shape

    def test_backward_matches_fp32_reference(self):
        x = _rand(4096)
        xa = x.clone().requires_grad_(True)
        _get_op().forward(xa).backward(torch.ones_like(xa))

        ref = x.float().cpu().requires_grad_(True)
        NativeSiLUOp()(ref).backward(torch.ones_like(ref))
        torch.testing.assert_close(
            xa.grad.float().cpu(), ref.grad.to(torch.bfloat16).float(), atol=_ATOL, rtol=_RTOL
        )


@requires_ascend
class TestAscendSiLUInvariance:
    @pytest.mark.parametrize("n", [2047, 2048, 4096, 70000])
    def test_forward_is_batch_invariant(self, n):
        """The first 1024 elements must not change when more elements join.

        Different sizes launch a different number of AI-core blocks and a
        different number of strided passes; a batch-invariant elementwise op
        evaluates each element identically regardless.
        """
        op = _get_op()
        small = _rand(1024, seed=7)
        large = torch.cat([small, _rand(n, seed=8)])
        assert torch.equal(op.forward(small), op.forward(large)[:1024])

    def test_backward_is_batch_invariant(self):
        op = _get_op()
        base = _rand(1024, seed=11)

        def grad_of(x):
            xa = x.clone().requires_grad_(True)
            op.forward(xa).backward(torch.ones_like(xa))
            return xa.grad

        large = torch.cat([base, _rand(4096, seed=12)])
        assert torch.equal(grad_of(base), grad_of(large)[:1024])

    def test_matches_swiglu_with_unit_up_bitwise(self):
        """silu(x) == swiglu(x, ones): both run the same FP32 sigmoid sequence."""

        from rl_engine.kernels.ops.ascend.activation.swiglu import SwiGLUAscendOp

        x = _rand(4096, seed=3)
        ones = torch.ones_like(x)
        assert torch.equal(_get_op().forward(x), SwiGLUAscendOp().forward(x, ones))


@requires_ascend
class TestAscendSiLUContract:
    def test_cpu_tensors_are_rejected(self):
        with pytest.raises(RuntimeError, match="NPU"):
            _get_op().forward(torch.randn(16, dtype=torch.bfloat16))

    def test_unsupported_dtype_is_rejected(self):
        with pytest.raises(TypeError):
            _get_op().forward(_rand(16).to(torch.int32))
