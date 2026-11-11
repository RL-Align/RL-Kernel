# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P1-2 fp32_gemm_rms Triton backend (numeric profile ``oracle-fp32-mhc-v1``).

Same determinism-by-construction schedule as the CUDA backend: every reduced
output element is produced by a single FP32 accumulator walking its reduction
axis in ascending order, and parallelism comes only from independent output
elements. No ``tl.sum`` / ``tl.dot`` anywhere: those pick their own reduction
tree.

Rounding is pinned as follows (all measured on-device against torch):

- Plain Triton ``+`` / ``*`` lower to round-to-nearest, non-FTZ PTX
  (``add.rn.f32`` / ``mul.rn.f32``); every kernel launches with
  ``enable_fp_fusion=False`` so mul-then-add cannot contract into an FMA.
  ``libdevice.*_rn`` is deliberately NOT used: Triton links libdevice with
  FTZ enabled, so ``libdevice.mul_rn`` flushes subnormal results.
- Division and square root use inline PTX ``div.rn.f32`` / ``sqrt.rn.f32``
  (correctly rounded, non-FTZ), byte-matching torch's tensor ops.
- The oracle's ``q = norm / sqrt(K)`` divides by a Python scalar, which torch
  executes as a multiply by the FP32-rounded double reciprocal; the kernel
  reproduces exactly that.
"""

from __future__ import annotations

import math

import torch

try:  # pragma: no cover - exercised only where triton is installed
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = hasattr(tl, "inline_asm_elementwise")
    _TRITON_IMPORT_ERROR = (
        None if _TRITON_AVAILABLE else "triton.language lacks inline_asm_elementwise"
    )
except ImportError as exc:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False
    _TRITON_IMPORT_ERROR = str(exc)


def triton_available() -> bool:
    return _TRITON_AVAILABLE


def _require_triton() -> None:
    if not _TRITON_AVAILABLE:
        raise NotImplementedError(
            f"Triton backend unavailable (fail-closed, no silent fallback): "
            f"{_TRITON_IMPORT_ERROR}"
        )


if _TRITON_AVAILABLE:

    @triton.jit
    def _div_rn(a, b):
        """Correctly rounded non-FTZ FP32 division (torch-byte-equal)."""
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _sqrt_rn(a):
        """Correctly rounded non-FTZ FP32 square root (torch-byte-equal)."""
        return tl.inline_asm_elementwise(
            "sqrt.rn.f32 $0, $1;", "=f,f", [a], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _gemm_rms_fwd_kernel(
        x_ptr,
        w_ptr,
        p_ptr,
        s_ptr,
        norm_ptr,
        q_ptr,
        r_ptr,
        K,
        N,
        recip_sqrt_k,
        eps,
        HAS_RMS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        VEC4: tl.constexpr,
    ):
        """One program per output row: lanes over n, k walked serially ascending.

        VEC4 unrolls four consecutive k steps (static_range keeps the exact
        ascending accumulation order; only loop overhead changes)."""
        t = tl.program_id(0)
        n_offs = tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        ss = tl.zeros([1], dtype=tl.float32)
        if VEC4:
            for k4 in range(0, K, 4):
                for i in tl.static_range(4):
                    x_k = tl.load(x_ptr + t * K + k4 + i)
                    w_k = tl.load(w_ptr + n_offs * K + k4 + i, mask=n_mask, other=0.0)
                    acc = acc + x_k * w_k
                    if HAS_RMS:
                        ss = ss + x_k * x_k
        else:
            for k in range(0, K):
                x_k = tl.load(x_ptr + t * K + k)
                w_k = tl.load(w_ptr + n_offs * K + k, mask=n_mask, other=0.0)
                acc = acc + x_k * w_k
                if HAS_RMS:
                    ss = ss + x_k * x_k
        tl.store(p_ptr + t * N + n_offs, acc, mask=n_mask)
        if HAS_RMS:
            # Oracle epilogue: norm = sqrt(s); q = norm * (1/sqrt(K)) (torch's
            # scalar-division decomposition); r = 1 / (q + eps). Controller
            # RMS, NOT rsqrt(mean + eps).
            one = tl.full([1], 1.0, dtype=tl.float32)
            norm = _sqrt_rn(ss)
            q = norm * recip_sqrt_k
            r = _div_rn(one, q + eps)
            lane = tl.arange(0, 1)
            tl.store(s_ptr + t + lane, ss)
            tl.store(norm_ptr + t + lane, norm)
            tl.store(q_ptr + t + lane, q)
            tl.store(r_ptr + t + lane, r)

    @triton.jit
    def _gemm_rms_bwd_dx_kernel(
        dp_ptr,
        dr_ptr,
        x_ptr,
        w_ptr,
        q_ptr,
        r_ptr,
        dx_ptr,
        K,
        N,
        k_float,
        HAS_RMS: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """dX[t, k] = fold_n dP[t, n] * W[n, k] (+ RMS leg), n ascending."""
        t = tl.program_id(0)
        kb = tl.program_id(1)
        k_offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for n in range(0, N):
            dp_n = tl.load(dp_ptr + t * N + n)
            w_blk = tl.load(w_ptr + n * K + k_offs, mask=k_mask, other=0.0)
            acc = acc + dp_n * w_blk
        if HAS_RMS:
            # dX_rms = dr * ((-(r*r) * x) / (K * q)), the oracle's association.
            r_t = tl.load(r_ptr + t)
            q_t = tl.load(q_ptr + t)
            dr_t = tl.load(dr_ptr + t)
            x_blk = tl.load(x_ptr + t * K + k_offs, mask=k_mask, other=0.0)
            neg_r2 = -(r_t * r_t)
            # Broadcast the scalar denominator to the block for the inline
            # asm; 0 + x is exact for every finite x here (denom >= +0).
            denom_blk = tl.zeros([BLOCK_K], dtype=tl.float32) + (k_float * q_t)
            scaled = _div_rn(neg_r2 * x_blk, denom_blk)
            acc = acc + dr_t * scaled
        tl.store(dx_ptr + t * K + k_offs, acc, mask=k_mask)

    @triton.jit
    def _gemm_rms_bwd_dw_kernel(
        dp_ptr,
        x_ptr,
        dw_ptr,
        T,
        K,
        N,
        BLOCK_K: tl.constexpr,
    ):
        """dW[n, k] = fold_t dP[t, n] * X[t, k], t ascending."""
        n = tl.program_id(0)
        kb = tl.program_id(1)
        k_offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for t in range(0, T):
            dp_tn = tl.load(dp_ptr + t * N + n)
            x_blk = tl.load(x_ptr + t * K + k_offs, mask=k_mask, other=0.0)
            acc = acc + dp_tn * x_blk
        tl.store(dw_ptr + n * K + k_offs, acc, mask=k_mask)

    @triton.jit
    def _fixed_k_gemm_kernel(
        a_ptr,
        b_ptr,
        out_ptr,
        K,
        N,
        BLOCK_N: tl.constexpr,
    ):
        m = tl.program_id(0)
        nb = tl.program_id(1)
        n_offs = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k in range(0, K):
            a_k = tl.load(a_ptr + m * K + k)
            b_k = tl.load(b_ptr + n_offs * K + k, mask=n_mask, other=0.0)
            acc = acc + a_k * b_k
        tl.store(out_ptr + m * N + n_offs, acc, mask=n_mask)


def _check_pair(x: torch.Tensor, w: torch.Tensor) -> tuple[int, int, int]:
    for name, t, dim in (("x", x, 2), ("w", w, 2)):
        if not t.is_cuda:
            raise RuntimeError(f"{name} must be a CUDA tensor (fail-closed)")
        if t.dtype != torch.float32:
            raise TypeError(f"{name} must be FP32, got {t.dtype}")
        if t.dim() != dim:
            raise ValueError(f"{name} must be {dim}-D, got {t.dim()}-D")
    if w.shape[1] != x.shape[1]:
        raise ValueError(f"w K {w.shape[1]} != x K {x.shape[1]}")
    return int(x.shape[0]), int(w.shape[0]), int(x.shape[1])


def triton_gemm_rms_forward(
    x: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton()
    tokens, n_dim, k_dim = _check_pair(x, w)
    if k_dim < 1:
        raise ValueError("K must be >= 1")
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError("eps must be positive and finite")
    x = x.contiguous()
    w = w.contiguous()
    p = torch.empty((tokens, n_dim), dtype=torch.float32, device=x.device)
    s = torch.empty((tokens,), dtype=torch.float32, device=x.device)
    norm = torch.empty_like(s)
    q = torch.empty_like(s)
    r = torch.empty_like(s)
    if tokens > 0:
        block_n = max(triton.next_power_of_2(n_dim), 2)
        if block_n > 32:
            raise ValueError(
                f"fp32_gemm_rms Triton kernel supports N <= 32, got {n_dim} (fail-closed)"
            )
        _gemm_rms_fwd_kernel[(tokens,)](
            x,
            w,
            p,
            s,
            norm,
            q,
            r,
            k_dim,
            n_dim,
            float(1.0 / math.sqrt(float(k_dim))),
            float(eps),
            HAS_RMS=True,
            BLOCK_N=block_n,
            VEC4=(k_dim % 4 == 0),
            num_warps=1,
            enable_fp_fusion=False,
        )
    return p, r, s, norm, q


def triton_gemm_rms_backward(
    dp: torch.Tensor,
    dr: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_triton()
    tokens, n_dim, k_dim = _check_pair(x, w)
    dp = dp.contiguous()
    dr = dr.contiguous()
    x = x.contiguous()
    w = w.contiguous()
    q = q.contiguous()
    r = r.contiguous()
    if tuple(dp.shape) != (tokens, n_dim):
        raise ValueError(f"dp shape {tuple(dp.shape)} != {(tokens, n_dim)}")
    if dr.shape[0] != tokens or q.shape[0] != tokens or r.shape[0] != tokens:
        raise ValueError("dr/q/r must be [T]")
    dx = torch.empty((tokens, k_dim), dtype=torch.float32, device=x.device)
    dw = torch.zeros((n_dim, k_dim), dtype=torch.float32, device=x.device)
    if tokens > 0:
        block_k = min(max(triton.next_power_of_2(k_dim), 16), 1024)
        k_blocks = triton.cdiv(k_dim, block_k)
        _gemm_rms_bwd_dx_kernel[(tokens, k_blocks)](
            dp,
            dr,
            x,
            w,
            q,
            r,
            dx,
            k_dim,
            n_dim,
            float(k_dim),
            HAS_RMS=True,
            BLOCK_K=block_k,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _gemm_rms_bwd_dw_kernel[(n_dim, k_blocks)](
            dp,
            x,
            dw,
            tokens,
            k_dim,
            n_dim,
            BLOCK_K=block_k,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return dx, dw


def triton_fixed_k_gemm_forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Generic deterministic ``x @ w.T`` (fold over k ascending), any N."""
    _require_triton()
    m_dim, n_dim, k_dim = _check_pair(x, w)
    x = x.contiguous()
    w = w.contiguous()
    out = torch.empty((m_dim, n_dim), dtype=torch.float32, device=x.device)
    if m_dim > 0 and n_dim > 0:
        block_n = min(max(triton.next_power_of_2(n_dim), 2), 32)
        n_blocks = triton.cdiv(n_dim, block_n)
        _fixed_k_gemm_kernel[(m_dim, n_blocks)](
            x,
            w,
            out,
            k_dim,
            n_dim,
            BLOCK_N=block_n,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return out


def triton_fixed_k_gemm_backward(
    dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_triton()
    tokens, n_dim, k_dim = _check_pair(x, w)
    dy = dy.contiguous()
    x = x.contiguous()
    w = w.contiguous()
    if tuple(dy.shape) != (tokens, n_dim):
        raise ValueError(f"dy shape {tuple(dy.shape)} != {(tokens, n_dim)}")
    dx = torch.empty((tokens, k_dim), dtype=torch.float32, device=x.device)
    dw = torch.zeros((n_dim, k_dim), dtype=torch.float32, device=x.device)
    if tokens > 0:
        block_k = min(max(triton.next_power_of_2(k_dim), 16), 1024)
        k_blocks = triton.cdiv(k_dim, block_k)
        _gemm_rms_bwd_dx_kernel[(tokens, k_blocks)](
            dy,
            dy,  # unused when HAS_RMS=False
            x,
            w,
            dy,  # unused
            dy,  # unused
            dx,
            k_dim,
            n_dim,
            float(k_dim),
            HAS_RMS=False,
            BLOCK_K=block_k,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _gemm_rms_bwd_dw_kernel[(n_dim, k_blocks)](
            dy,
            x,
            dw,
            tokens,
            k_dim,
            n_dim,
            BLOCK_K=block_k,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return dx, dw


__all__ = [
    "triton_available",
    "triton_fixed_k_gemm_backward",
    "triton_fixed_k_gemm_forward",
    "triton_gemm_rms_backward",
    "triton_gemm_rms_forward",
]
