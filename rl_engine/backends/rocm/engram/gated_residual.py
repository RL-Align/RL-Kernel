# SPDX-License-Identifier: Apache-2.0
"""V4.1 Engram normalized gate, with deterministic non-atomic backward.

Implements the gate after the embedding/wkv projection, not the hash lookup or
whole Engram module. The FP32 reduction tree and no-FMA profile are shared by
training and inference. No V4 epsilon or joint-HC normalization is substituted.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gate(
    X,
    K,
    V,
    QW,
    KW,
    MASK,
    Y,
    SAVED,
    D: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    t, h = row // H, row % H
    d = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + d, d < D, other=0).to(tl.float32)
    k = tl.load(K + row * D + d, d < D, other=0).to(tl.float32)
    w = tl.load(QW + h * D + d, d < D, other=0).to(tl.float32)
    w = w * tl.load(KW + h * D + d, d < D, other=0).to(tl.float32)
    rx = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    rk = tl.rsqrt(tl.sum(k * k, 0) / D + EPS)
    dot = tl.sum((x * w) * k, 0) * (rx * rk) * (D**-0.5)
    root = tl.sqrt(tl.maximum(tl.abs(dot), 1.0e-6))
    signed = tl.where(dot < 0, -root, root)
    gate = 1.0 / (1.0 + tl.exp(-signed))
    active = tl.load(MASK + t)
    gate = tl.where(active, gate, 0.0)
    v = tl.load(V + t * D + d, d < D, other=0).to(tl.float32)
    tl.store(Y + row * D + d, x + gate * v, d < D)
    tl.store(SAVED + row * 4 + 0, rx)
    tl.store(SAVED + row * 4 + 1, rk)
    tl.store(SAVED + row * 4 + 2, dot)
    tl.store(SAVED + row * 4 + 3, gate)


@triton.jit
def _gate_bwd(
    DY,
    X,
    K,
    V,
    QW,
    KW,
    SAVED,
    DX,
    DK,
    DP,
    DV_PART,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    t, h = row // H, row % H
    d = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + d, d < D, other=0).to(tl.float32)
    k = tl.load(K + row * D + d, d < D, other=0).to(tl.float32)
    dy = tl.load(DY + row * D + d, d < D, other=0).to(tl.float32)
    v = tl.load(V + t * D + d, d < D, other=0).to(tl.float32)
    qw = tl.load(QW + h * D + d, d < D, other=0).to(tl.float32)
    kw = tl.load(KW + h * D + d, d < D, other=0).to(tl.float32)
    rx = tl.load(SAVED + row * 4 + 0)
    rk = tl.load(SAVED + row * 4 + 1)
    dot = tl.load(SAVED + row * 4 + 2)
    gate = tl.load(SAVED + row * 4 + 3)
    ddot = tl.sum(dy * v, 0) * gate * (1.0 - gate)
    ddot = ddot * tl.where(tl.abs(dot) >= 1e-6, 0.5 / tl.sqrt(tl.maximum(tl.abs(dot), 1e-6)), 0.0)
    factor = rx * rk * (D**-0.5)
    dx = dy + ddot * ((qw * kw) * k * factor - dot * x * rx * rx / D)
    dk = ddot * ((qw * kw) * x * factor - dot * k * rk * rk / D)
    tl.store(DX + row * D + d, dx, d < D)
    tl.store(DK + row * D + d, dk, d < D)
    tl.store(DP + row * D + d, ddot * ((x * k) * factor), d < D)
    tl.store(DV_PART + row * D + d, dy * gate, d < D)


@triton.jit
def _weights_bwd(DP, QW, KW, DQW, DKW, T: tl.constexpr, D: tl.constexpr, H: tl.constexpr):
    h = tl.program_id(0)
    d = tl.program_id(1) * 256 + tl.arange(0, 256)
    acc = tl.full((256,), 0, tl.float32)
    for t in range(T):
        acc = acc + tl.load(DP + (t * H + h) * D + d, d < D, other=0)
    q = tl.load(QW + h * D + d, d < D, other=0).to(tl.float32)
    k = tl.load(KW + h * D + d, d < D, other=0).to(tl.float32)
    tl.store(DQW + h * D + d, acc * k, d < D)
    tl.store(DKW + h * D + d, acc * q, d < D)


@triton.jit
def _value_bwd(PART, DV, D: tl.constexpr, H: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * 256 + tl.arange(0, 256)
    h = tl.arange(0, triton.next_power_of_2(H))
    p = tl.load(
        PART + (t * H + h[:, None]) * D + d[None, :], (h[:, None] < H) & (d[None, :] < D), other=0
    )
    tl.store(DV + t * D + d, tl.sum(p, 0), d < D)


def _validate(x, key, value, qw, kw, mask, eps):
    if not torch.version.hip or x.device.type != "cuda":
        raise RuntimeError("Engram gate requires an AMD ROCm GPU")
    if x.ndim < 3 or key.shape != x.shape or value.shape != (*x.shape[:-2], x.shape[-1]):
        raise ValueError("expected x/key [...,HC,D] and value [...,D]")
    if qw.shape != x.shape[-2:] or kw.shape != qw.shape:
        raise ValueError("q/k weights must be [HC,D]")
    if not 0 < x.shape[-1] <= 8192 or not 0 < x.shape[-2] <= 16:
        raise ValueError("supported HC in [1,16], D in [1,8192]")
    if mask is not None and (mask.shape != x.shape[:-2] or mask.dtype != torch.bool):
        raise ValueError("token mask must be bool with the token shape")
    for tensor in (x, key, value, qw, kw):
        if tensor.device != x.device or tensor.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("operands must share the device and be BF16 or FP32")
    if mask is not None and mask.device != x.device:
        raise ValueError("mask device must match x")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be finite and positive")


class _Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, key, value, qw, kw, mask, eps):
        _validate(x, key, value, qw, kw, mask, eps)
        x, key, value, qw, kw = (t.contiguous() for t in (x, key, value, qw, kw))
        h, d = x.shape[-2:]
        tokens = x.numel() // (h * d)
        if mask is None:
            mask = torch.ones(x.shape[:-2], device=x.device, dtype=torch.bool)
        mask = mask.contiguous()
        out = torch.empty_like(x)
        saved = torch.empty((tokens * h, 4), device=x.device, dtype=torch.float32)
        with torch.cuda.device(x.device):
            if tokens:
                _gate[(tokens * h,)](
                    x,
                    key,
                    value,
                    qw,
                    kw,
                    mask,
                    out,
                    saved,
                    d,
                    h,
                    eps,
                    triton.next_power_of_2(d),
                    enable_fp_fusion=False,
                )
        ctx.save_for_backward(x, key, value, qw, kw, saved)
        return out

    @staticmethod
    def backward(ctx, dy):
        x, key, value, qw, kw, saved = ctx.saved_tensors
        h, d = x.shape[-2:]
        tokens = x.numel() // (h * d)
        dx, dk = torch.empty_like(x), torch.empty_like(key)
        dp = torch.empty_like(x, dtype=torch.float32)
        part = torch.empty_like(dp)
        dv, dqw, dkw = torch.empty_like(value), torch.empty_like(qw), torch.empty_like(kw)
        with torch.cuda.device(x.device):
            if tokens:
                _gate_bwd[(tokens * h,)](
                    dy.contiguous(),
                    x,
                    key,
                    value,
                    qw,
                    kw,
                    saved,
                    dx,
                    dk,
                    dp,
                    part,
                    d,
                    h,
                    triton.next_power_of_2(d),
                    enable_fp_fusion=False,
                )
                _value_bwd[(tokens, triton.cdiv(d, 256))](part, dv, d, h)
            _weights_bwd[(h, triton.cdiv(d, 256))](
                dp, qw, kw, dqw, dkw, tokens, d, h, enable_fp_fusion=False
            )
        return dx, dk, dv, dqw, dkw, None, None


def engram_gated_residual(x, key, value, q_weight, k_weight, token_mask=None, *, eps=1e-20):
    """Use identical gate kernels in autograd training and inference_mode."""
    return _Gate.apply(x, key, value, q_weight, k_weight, token_mask, float(eps))
