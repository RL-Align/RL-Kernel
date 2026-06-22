# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch-invariant deterministic GEMM, CUDA path (WS1 #146).

Hand-written kernel (csrc/cuda/gemm/det_gemm_kernel.cu): fixed K-accumulation
order, FP32 accumulation, no split-K. A row's output is invariant to batch size,
chunked-prefill, and padding. No PyTorch fallback -- a generic matmul (cuBLAS)
would silently break invariance (see NativeGemmOp). Tensor-parallel GEMM is WS2.
"""
import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger


class _DetGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _C.det_gemm_fwd(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        da = _C.det_gemm_da(grad_out, b) if ctx.needs_input_grad[0] else None
        db = _C.det_gemm_db(a, grad_out) if ctx.needs_input_grad[1] else None
        return da, db


class DetGemmOp:
    """Hand-written batch-invariant GEMM. a:[M,K] bf16, b:[K,N] bf16 -> [M,N] bf16."""

    def __init__(self):
        self.has_hardware_op = False
        if _EXT_AVAILABLE and hasattr(_C, "det_gemm_fwd"):
            self.op = _C.det_gemm_fwd
            self.has_hardware_op = True
            logger.info("Successfully linked to RL-Kernel _C.det_gemm_fwd.")
        else:
            logger.warning(
                "RL-Kernel _C.det_gemm_fwd unavailable; DetGemmOp requires the "
                "compiled CUDA extension and has no batch-invariant fallback."
            )

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "BF16 only"
        assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA device"
        if not self.has_hardware_op:
            raise RuntimeError(
                "DetGemmOp: compiled _C.det_gemm kernel unavailable; no "
                "batch-invariant fallback exists. Build the extension first."
            )
        return _DetGemmFn.apply(a.contiguous(), b.contiguous())


def deterministic_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Functional entry. a:[M,K] bf16, b:[K,N] bf16 -> [M,N] bf16."""
    return _DetGemmFn.apply(a, b)