# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE


class _MusaDetGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, output_fp32: bool):
        ctx.save_for_backward(a, b)
        ctx.output_fp32 = bool(output_fp32)
        if ctx.output_fp32:
            return _C.det_gemm_fwd_fp32(a, b)
        return _C.det_gemm_fwd(a, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b = ctx.saved_tensors
        grad_native = grad_output.contiguous()
        if ctx.output_fp32:
            grad_native = grad_native.to(torch.bfloat16)
        grad_a = _C.det_gemm_da(grad_native, b) if ctx.needs_input_grad[0] else None
        grad_b = _C.det_gemm_db(a, grad_native) if ctx.needs_input_grad[1] else None
        return grad_a, grad_b, None


class _MusaDetGemmAccumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        ctx.save_for_backward(a, b)
        return _C.det_gemm_accum_fwd_fp32(a, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b = ctx.saved_tensors
        grad_fp32 = grad_output.contiguous().float()
        grad_a = (
            _C.det_gemm_accum_fwd_fp32(grad_fp32, b.t().contiguous()).to(a.dtype)
            if ctx.needs_input_grad[0]
            else None
        )
        grad_b = (
            _C.det_gemm_accum_fwd_fp32(a.t().contiguous(), grad_fp32).to(b.dtype)
            if ctx.needs_input_grad[1]
            else None
        )
        return grad_a, grad_b


class _MusaDetLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, weight: torch.Tensor):
        ctx.save_for_backward(a, weight)
        return _C.det_gemm_fwd_rhs_transposed(a, weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, weight = ctx.saved_tensors
        grad_a = _C.det_gemm_fwd(grad_output, weight) if ctx.needs_input_grad[0] else None
        grad_weight = _C.det_gemm_db_transposed(a, grad_output) if ctx.needs_input_grad[1] else None
        return grad_a, grad_weight


class MusaDetGemmOp:
    """Deterministic fixed-order GEMM for MUSA BF16 tensors."""

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or _C is None:
            raise RuntimeError("MUSA det_gemm requires the compiled extension")
        required = (
            "det_gemm_fwd",
            "det_gemm_fwd_fp32",
            "det_gemm_accum_fwd_fp32",
            "det_gemm_fwd_rhs_transposed",
            "det_gemm_da",
            "det_gemm_db",
            "det_gemm_db_transposed",
        )
        missing = [name for name in required if not hasattr(_C, name)]
        if missing:
            raise RuntimeError(f"MUSA det_gemm extension is missing: {', '.join(missing)}")

    @staticmethod
    def _check(a: torch.Tensor, b: torch.Tensor) -> None:
        if a.device.type != "musa" or b.device.type != "musa":
            raise ValueError("MUSA det_gemm requires MUSA tensors")
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise TypeError("MUSA det_gemm currently supports BF16 tensors")

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._check(a, b)
        if a.ndim != 2 or b.ndim != 2 or a.size(1) != b.size(0):
            raise ValueError("det_gemm expects A[M,K] and B[K,N]")
        return _MusaDetGemmFunction.apply(a.contiguous(), b.contiguous(), False)

    def forward_fp32(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._check(a, b)
        if a.ndim != 2 or b.ndim != 2 or a.size(1) != b.size(0):
            raise ValueError("det_gemm expects A[M,K] and B[K,N]")
        return _MusaDetGemmFunction.apply(a.contiguous(), b.contiguous(), True)

    def forward_accum_fp32(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.device.type != "musa" or b.device.type != "musa":
            raise ValueError("MUSA det_gemm requires MUSA tensors")
        supported = (torch.bfloat16, torch.float32)
        if a.dtype not in supported or b.dtype not in supported:
            raise TypeError("FP32-accumulation GEMM requires BF16 or FP32 inputs")
        if a.ndim != 2 or b.ndim != 2 or a.size(1) != b.size(0):
            raise ValueError("det_gemm expects A[M,K] and B[K,N]")
        return _MusaDetGemmAccumFunction.apply(a.contiguous(), b.contiguous())

    def linear(self, a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        self._check(a, weight)
        if a.ndim != 2 or weight.ndim != 2 or a.size(1) != weight.size(1):
            raise ValueError("linear expects A[M,K] and weight[N,K]")
        return _MusaDetLinearFunction.apply(a.contiguous(), weight.contiguous())


def deterministic_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return MusaDetGemmOp()(a, b)
