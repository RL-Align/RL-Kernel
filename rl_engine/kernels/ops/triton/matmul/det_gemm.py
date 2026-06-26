# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch-invariant deterministic GEMM, Triton path (WS1).

Portable implementation with the SAME invariance guarantees as the CUDA path:
autotune disabled, BLOCK sizes pinned, no split-K, fixed K-loop order, FP32
accumulation, no TF32. Used as the cross-backend reference and the ROCm/portable
fallback. Slower than a tuned GEMM by design.
"""
import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

from rl_engine.utils.logger import logger

# Pinned. NOT autotuned (autotune picks per-shape configs -> breaks invariance).
_BLOCK_M, _BLOCK_N, _BLOCK_K = 64, 64, 32


if _TRITON_AVAILABLE:

    @triton.jit
    def _det_gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # One program = one output tile, walks the whole K in fixed order.
        # No split-K -> K-accumulation order independent of M -> batch-invariant.
        pid_m, pid_n = tl.program_id(0), tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_rem = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
            acc += tl.dot(a, b, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c = acc.to(c_ptr.dtype.element_ty)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)


def _triton_gemm(a, b):
    a, b = a.contiguous(), b.contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
    _det_gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
    )
    return c


class _TritonDetGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _triton_gemm(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        da = _triton_gemm(grad_out, b.t().contiguous()) if ctx.needs_input_grad[0] else None
        db = _triton_gemm(a.t().contiguous(), grad_out) if ctx.needs_input_grad[1] else None
        return da, db


class TritonDetGemmOp:
    """Batch-invariant deterministic GEMM, Triton path."""

    def __init__(self):
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton not available for TritonDetGemmOp")
        logger.info("TritonDetGemmOp ready (deterministic, autotune disabled).")

    def __call__(self, a, b):
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "BF16 only"
        assert a.is_cuda and b.is_cuda, "CUDA only"
        return _TritonDetGemmFn.apply(a, b)


def deterministic_gemm_triton(a, b):
    return _TritonDetGemmFn.apply(a, b)
