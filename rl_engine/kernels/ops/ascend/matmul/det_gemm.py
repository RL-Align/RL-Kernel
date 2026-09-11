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
        # FP32-accumulation rowwise backward (the canonical row-fold VJP).
        # The BF16 mid-split tree grads round at every node, which the
        # gradient-accuracy judgment compares against unrounded FP32
        # reference grads (a structural 2.0-4.0 residual at near-cancellation
        # outputs); the rowwise kernels reduce each output row in one fixed
        # per-row order with FP32 accumulation, so the gradients are both
        # batch-invariant and ULP-close to the FP32 reference.
        a, b = ctx.saved_tensors
        grad_fp32 = grad_out.contiguous().float()
        da = (
            _rowwise_fp32(grad_fp32, b.float().t().contiguous()).to(a.dtype)
            if ctx.needs_input_grad[0]
            else None
        )
        db = (
            _rowwise_fp32(a.float().t().contiguous(), grad_fp32).to(b.dtype)
            if ctx.needs_input_grad[1]
            else None
        )
        record_backward(
            "det_gemm",
            kernel_id="rl_engine._C_npu.det_gemm_rowwise_ascend_fwd_fp32",
            impl="ascend_rowwise_fp32_accum_det_gemm",
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
        da = _C_npu.det_gemm_ascend_fwd(grad_out, weight) if ctx.needs_input_grad[0] else None
        dweight = (
            _C_npu.det_gemm_ascend_db_transposed(a, grad_out) if ctx.needs_input_grad[1] else None
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


def _rowwise_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not hasattr(_C_npu, "det_gemm_rowwise_ascend_fwd_fp32"):
        raise RuntimeError(
            "FP32 rowwise deterministic GEMM requires a rebuilt Ascend extension; "
            "rebuild with KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host"
        )
    return _C_npu.det_gemm_rowwise_ascend_fwd_fp32(a.float().contiguous(), b.float().contiguous())


class _DetGemmAscendAccumFn(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _rowwise_fp32(a, b)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_fp32 = grad_out.contiguous().float()
        da = (
            _rowwise_fp32(grad_fp32, b.float().t().contiguous()).to(a.dtype)
            if ctx.needs_input_grad[0]
            else None
        )
        db = (
            _rowwise_fp32(a.float().t().contiguous(), grad_fp32).to(b.dtype)
            if ctx.needs_input_grad[1]
            else None
        )
        record_backward(
            "det_gemm",
            kernel_id="rl_engine._C_npu.det_gemm_rowwise_ascend_fwd_fp32",
            impl="ascend_rowwise_fp32_accum_det_gemm",
            family="ascend",
        )
        return da, db


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
            raise RuntimeError(f"missing {', '.join(missing)} in _C_npu; rebuild the extension")
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

    def forward_accum_fp32(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """FP32-accumulation rowwise GEMM, the twin of the CUDA op's entry.

        The canonical row-fold VJP drives its matmuls through this path. It
        does not round intermediate nodes to BF16, so gradients keep the FP32
        accumulation the contract requires; determinism comes from the fixed
        per-row reduction order of the underlying kernel rather than from the
        BF16 mid-split tree used by the BF16 forward.
        """
        if a.dtype not in (torch.bfloat16, torch.float32) or b.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise TypeError("FP32-accumulation GEMM requires BF16 or FP32 inputs")
        assert a.device.type == "npu" and b.device.type == "npu", "Inputs must be on NPU"
        return _DetGemmAscendAccumFn.apply(a.contiguous(), b.contiguous())

    def linear(self, a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply a native [N,K] linear weight without materializing weight.T."""
        assert a.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16, "BF16 only"
        assert a.device.type == "npu" and weight.device.type == "npu", "Inputs must be on NPU"
        return _DetLinearAscendFn.apply(a.contiguous(), weight.contiguous())

    def parameter_vjp_contributions_fp32(
        self, *, a: torch.Tensor, b: torch.Tensor, grad_output: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Canonical row-fold parameter contribution (the CUDA twin).

        dW[k,n] = sum_tokens a[t,k] * dC[t,n]: each token's FP32 outer
        product is returned per row, and the C4 harness accumulates the
        per-row contributions in FP32 across call spans, so chunked /
        padded / permuted layouts sum the same row contributions in the
        same order and produce a bitwise-identical weight gradient.
        """
        del b
        rows_a = a.float()
        rows_g = grad_output.float()
        return {"b": rows_a[:, :, None] * rows_g[:, None, :]}


def deterministic_gemm_ascend(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Functional entry. a:[M,K] bf16, b:[K,N] bf16 -> [M,N] bf16."""
    return _DetGemmAscendFn.apply(a, b, False)
