# SPDX-License-Identifier: Apache-2.0
"""V4.1 RMSNorm with explicit epsilon and deterministic parameter VJP."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X, W, Y, R, D: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, B)
    x = tl.load(X + t * D + d, d < D, other=0).to(tl.float32)
    w = tl.load(W + d, d < D, other=0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    tl.store(Y + t * D + d, (x * r) * w, d < D)
    tl.store(R + t, r)


@triton.jit
def _norm_bwd(DY, X, W, R, DX, DW_PART, D: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, B)
    x = tl.load(X + t * D + d, d < D, other=0).to(tl.float32)
    w = tl.load(W + d, d < D, other=0).to(tl.float32)
    dy = tl.load(DY + t * D + d, d < D, other=0).to(tl.float32)
    r = tl.load(R + t)
    v = dy * w
    dot = tl.sum(v * x, 0) / D
    tl.store(DX + t * D + d, v * r - x * (r * r * r * dot), d < D)
    tl.store(DW_PART + t * D + d, (dy * x) * r, d < D)


@triton.jit
def _norm_dw(P, DW, T: tl.constexpr, D: tl.constexpr):
    d = tl.program_id(0) * 256 + tl.arange(0, 256)
    total = tl.full((256,), 0, tl.float32)
    for t in range(T):
        total = total + tl.load(P + t * D + d, d < D, other=0)
    tl.store(DW + d, total, d < D)


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        if not torch.version.hip or x.device.type != "cuda":
            raise RuntimeError("V4.1 RMSNorm requires AMD ROCm")
        if x.ndim < 1 or weight.shape != (x.shape[-1],) or not 0 < x.shape[-1] <= 8192:
            raise ValueError("expected x[...,D], weight[D], 0 < D <= 8192")
        if x.dtype not in (torch.bfloat16, torch.float32) or weight.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise TypeError("RMSNorm supports BF16/FP32 operands")
        if weight.device != x.device or not math.isfinite(eps) or eps <= 0:
            raise ValueError("device mismatch or invalid RMS epsilon")
        x, weight = x.contiguous(), weight.contiguous()
        d = x.shape[-1]
        t = x.numel() // d
        out = torch.empty_like(x)
        r = torch.empty(t, device=x.device, dtype=torch.float32)
        with torch.cuda.device(x.device):
            if t:
                _norm[(t,)](
                    x, weight, out, r, d, eps, triton.next_power_of_2(d), enable_fp_fusion=False
                )
        ctx.save_for_backward(x, weight, r)
        return out

    @staticmethod
    def backward(ctx, dy):
        x, w, r = ctx.saved_tensors
        d = x.shape[-1]
        t = x.numel() // d
        dx = torch.empty_like(x)
        dw = torch.empty_like(w)
        partial = torch.empty_like(x, dtype=torch.float32)
        with torch.cuda.device(x.device):
            if t:
                _norm_bwd[(t,)](
                    dy.contiguous(),
                    x,
                    w,
                    r,
                    dx,
                    partial,
                    d,
                    triton.next_power_of_2(d),
                    enable_fp_fusion=False,
                )
            _norm_dw[(triton.cdiv(d, 256),)](partial, dw, t, d, enable_fp_fusion=False)
        return dx, dw, None


def rmsnorm(x, weight, *, eps):
    return _Norm.apply(x, weight, float(eps))
