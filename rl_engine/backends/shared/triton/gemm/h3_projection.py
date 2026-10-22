# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fixed-order Triton GEMM for the H3 projection rows (contract ``h3-projection-gemm-v1``).

One program owns one output tile and walks the whole reduction axis in ascending
``BLOCK_K`` steps, accumulating into FP32 registers. There is no split-K, no
stream-K, no atomics, no autotuning and no dependence of the tile shape, the
instruction sequence or the order of the reduction on ``M``, ``N`` or the grid:

* a token's output bytes are independent of how many tokens share the call and of
  its position (``M`` only decides how many programs run, not what each computes);
* a tensor-parallel output shard (a block of whole rows of ``W``) equals the same columns of
  the full product, because ``N`` only selects which tiles exist;
* the reduction axis is always ``K`` in ascending order (``T`` ascending for ``d_weight``).

Arithmetic: BF16 operands use the tensor-core ``tl.dot`` with an FP32 accumulator;
FP32 operands use ``input_precision="ieee"`` (no TF32, no reduced-precision reductions).
The bias is added once in FP32 and the result is rounded once to the output dtype.
The accumulation order inside one MMA instruction is the architecture's, so the contract
is bitwise per architecture family (Hopper wgmma, Ampere/Ada mma.sync), not across them;
tile configurations below are fixed per dtype and therefore part of the contract.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - import guard mirrors the other Triton backends
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

CONTRACT = "h3-projection-gemm-v1"
# Fixed tile configurations (part of the contract; changing one changes the bytes).
CONFIG_BF16 = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=3)
CONFIG_FP32 = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3)

if HAS_TRITON:

    @triton.jit
    def _projection_kernel(
        a_ptr,
        b_ptr,
        bias_ptr,
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
        HAS_BIAS: tl.constexpr,
        IEEE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_rem = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)
            if IEEE:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(
            c_ptrs,
            acc.to(c_ptr.dtype.element_ty),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def available() -> bool:
    return bool(HAS_TRITON and torch.cuda.is_available())


def _require(*tensors: torch.Tensor) -> torch.dtype:
    if not HAS_TRITON:
        raise RuntimeError("triton is required for the H3 projection rows (no cuBLAS fallback)")
    dtype = tensors[0].dtype
    if dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"h3 projection GEMM supports float32 and bfloat16, got {dtype}")
    for t in tensors:
        if not t.is_cuda or t.dtype != dtype or t.device != tensors[0].device:
            raise ValueError("h3 projection GEMM needs same-dtype tensors on one CUDA device")
    return dtype


def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    a_t: bool = False,
    b_t: bool = False,
) -> torch.Tensor:
    """``C = A @ B (+ bias)`` with reduction in ascending index order.

    ``a`` is ``(M, K)`` (``a_t``: given as ``(K, M)``), ``b`` is ``(K, N)`` (``b_t``: given as
    ``(N, K)``). Operands may be strided; nothing is copied. FP32 accumulate, one rounding.
    """

    dtype = _require(a, b) if bias is None else _require(a, b, bias)
    config = CONFIG_BF16 if dtype == torch.bfloat16 else CONFIG_FP32
    if a_t:
        k, m = a.shape
        sam, sak = a.stride(1), a.stride(0)
    else:
        m, k = a.shape
        sam, sak = a.stride(0), a.stride(1)
    if b_t:
        n, kb = b.shape
        sbk, sbn = b.stride(1), b.stride(0)
    else:
        kb, n = b.shape
        sbk, sbn = b.stride(0), b.stride(1)
    if k != kb:
        raise ValueError(f"reduction axes differ: {k} vs {kb}")
    if m == 0 or n == 0 or k == 0:
        raise ValueError("h3 projection GEMM needs non-empty operands")
    c = torch.empty((m, n), device=a.device, dtype=out_dtype or dtype)
    grid = (triton.cdiv(m, config["BLOCK_M"]), triton.cdiv(n, config["BLOCK_N"]))
    _projection_kernel[grid](
        a,
        b,
        bias if bias is not None else c,
        c,
        m,
        n,
        k,
        sam,
        sak,
        sbk,
        sbn,
        c.stride(0),
        c.stride(1),
        HAS_BIAS=bias is not None,
        IEEE=dtype == torch.float32,
        BLOCK_M=config["BLOCK_M"],
        BLOCK_N=config["BLOCK_N"],
        BLOCK_K=config["BLOCK_K"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return c


def linear_forward(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """``y = x @ W^T (+ bias)`` for ``x (T, K)`` and ``W (N, K)``."""

    return gemm(x, weight, bias=bias, b_t=True)


def linear_backward_input(grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``dx = g @ W`` for ``g (T, N)`` and ``W (N, K)``; reduction over ``N`` ascending."""

    return gemm(grad, weight)


def linear_backward_weight(
    grad: torch.Tensor, x: torch.Tensor, *, with_bias: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``dW = g^T @ x`` and ``db = sum_t g`` with the token axis reduced in ascending order."""

    dweight = gemm(grad, x, a_t=True)
    if not with_bias:
        return dweight, None
    ones = torch.ones(grad.shape[0], 1, device=grad.device, dtype=grad.dtype)
    dbias = gemm(grad, ones, a_t=True).reshape(-1)
    return dweight, dbias
