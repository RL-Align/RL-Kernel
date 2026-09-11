# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Ascend NPU batch-invariant deterministic GEMM (WS1 #146).

Forward: hand-written Ascend C kernel (`_C_npu.det_gemm_ascend_*`). Every
output row-tile is reduced end-to-end by exactly one AI-core block with a
fixed ascending 32-element leaf order and the CUDA mid-split BF16-add tree,
so per-element numerics are batch-invariant (the same algorithm as the CUDA
`det_gemm_kernel.cu` and the Triton tree reference).

Backward: reuses the forward kernel on transposed operands, exactly like the
CUDA op: dA = dC @ B^T, dB = A^T @ dC (with the canonical [N,K] layout
variant for native weights).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.utils.logger import logger

_C_npu: Any = None
try:
    from rl_engine import _C_npu

    _NPU_EXT_AVAILABLE = True
except ImportError:  # pragma: no cover - Ascend extension not built
    _NPU_EXT_AVAILABLE = False

_REQUIRED = (
    "det_gemm_ascend_fwd",
    "det_gemm_ascend_fwd_rhs_transposed",
    "det_gemm_ascend_fwd_fp32",
    "det_gemm_ascend_da",
    "det_gemm_ascend_db",
    "det_gemm_ascend_db_transposed",
)


class _DetGemmAscendFn(Function):
    @staticmethod
    def forward(ctx, a, b, output_fp32=False):
        ctx.save_for_backward(a, b)
        if output_fp32:
            return _C_npu.det_gemm_ascend_fwd_fp32(a, b)
        return _C_npu.det_gemm_ascend_fwd(a, b)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        if grad_out.dtype != torch.bfloat16:
            grad_out = grad_out.to(torch.bfloat16)
        da = _C_npu.det_gemm_ascend_da(grad_out, b) if ctx.needs_input_grad[0] else None
        db = _C_npu.det_gemm_ascend_db(a, grad_out) if ctx.needs_input_grad[1] else None
        record_backward(
            "det_gemm",
            kernel_id=(
                "rl_engine._C_npu.det_gemm_ascend_da+rl_engine._C_npu.det_gemm_ascend_db"
            ),
            impl="ascend_det_gemm",
            family="ascend",
        )
        return da, db, None


class _DetLinearAscendFn(Function):
    @staticmethod
    def forward(ctx, a, weight):
        ctx.save_for_backward(a, weight)
        return _C_npu.det_gemm_ascend_fwd_rhs_transposed(a, weight)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        a, weight = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        if grad_out.dtype != torch.bfloat16:
            grad_out = grad_out.to(torch.bfloat16)
        # weight is physical [N,K]: reading it as logical [K'=N, N'=K] yields
        # dA = dC @ weight, the same trick the CUDA linear backward uses.
        da = (
            _C_npu.det_gemm_ascend_fwd(grad_out, weight)
            if ctx.needs_input_grad[0]
            else None
        )
        dweight = (
            _C_npu.det_gemm_ascend_db_transposed(a, grad_out)
            if ctx.needs_input_grad[1]
            else None
        )
        record_backward(
            "det_gemm",
            kernel_id=(
                "rl_engine._C_npu.det_gemm_ascend_fwd+"
                "rl_engine._C_npu.det_gemm_ascend_db_transposed"
            ),
            impl="ascend_det_gemm_linear",
            family="ascend",
        )
        return da, dweight


class DetGemmAscendOp:
    """Batch-invariant deterministic GEMM on Ascend NPU.

    a:[M,K] bf16, b:[K,N] bf16 -> [M,N] bf16. Strict backend: out-of-domain
    inputs are rejected up front and no non-strict fallback exists (the same
    refusal contract as the CUDA DetGemmOp).
    """

    def __init__(self) -> None:
        if not _NPU_EXT_AVAILABLE or _C_npu is None:
            raise RuntimeError(
                "strict RL-Kernel Ascend GEMM requires the compiled _C_npu "
                "extension; rebuild with KERNEL_ALIGN_FORCE_ASCEND=1 on an "
                "Ascend NPU host: 'pip install -e .'"
            )
        missing = [name for name in _REQUIRED if not hasattr(_C_npu, name)]
        if missing:
            raise RuntimeError(
                f"missing {', '.join(missing)} in _C_npu; rebuild the extension"
            )
        self.has_hardware_op = True
        logger.info("Successfully linked to precompiled _C_npu.det_gemm_ascend kernels.")

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "BF16 only"
        assert a.device.type == "npu" and b.device.type == "npu", "Inputs must be on NPU"
        return _DetGemmAscendFn.apply(a.contiguous(), b.contiguous(), False)

    def forward_fp32(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "BF16 only"
        assert a.device.type == "npu" and b.device.type == "npu", "Inputs must be on NPU"
        return _DetGemmAscendFn.apply(a.contiguous(), b.contiguous(), True)

    def linear(self, a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply a native [N,K] linear weight without materializing weight.T."""
        assert a.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16, "BF16 only"
        assert a.device.type == "npu" and weight.device.type == "npu", "Inputs must be on NPU"
        return _DetLinearAscendFn.apply(a.contiguous(), weight.contiguous())


def deterministic_gemm_ascend(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Functional entry. a:[M,K] bf16, b:[K,N] bf16 -> [M,N] bf16."""
    return _DetGemmAscendFn.apply(a, b, False)
