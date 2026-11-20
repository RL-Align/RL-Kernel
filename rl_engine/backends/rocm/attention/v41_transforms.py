# SPDX-License-Identifier: Apache-2.0
"""V4.1 interleaved RoPE and FP8/FP4 activation codecs with declared identity STE."""

import torch
import triton
import triton.language as tl

from rl_engine.backends.rocm.gemm.v41_fp8 import quantize_fp8_block32
from rl_engine.kernels.ops.triton.moe.fused_mlp import _decode_e4m3, _e8m0_scale


@triton.jit
def _dequant(C, S, Y, D: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    d = block * 32 + tl.arange(0, 32)
    code = tl.load(C + row * D + d)
    scale = _e8m0_scale(tl.load(S + row * (D // 32) + block))
    tl.store(Y + row * D + d, _decode_e4m3(code) * scale)


class _FP8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        quant = quantize_fp8_block32(flat)
        y = torch.empty_like(x)
        with torch.cuda.device(x.device):
            if flat.shape[0]:
                _dequant[(flat.shape[0], shape[-1] // 32)](
                    quant.codes, quant.scales, y, shape[-1], enable_fp_fusion=False
                )
        return y

    @staticmethod
    def backward(ctx, dy):
        return dy


@triton.jit
def _fake4(X, Y, D: tl.constexpr, G: tl.constexpr, E4S: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    d = block * G + tl.arange(0, G)
    x = tl.load(X + row * D + d).to(tl.float32)
    amax = tl.max(tl.abs(x), 0)
    if E4S:
        scale = (tl.maximum(amax, 6.0 * (2.0**-9)) / 6.0).to(tl.float8e4nv).to(tl.float32)
    else:
        ratio = tl.maximum(amax, 6.0 * (2.0**-126)) * (1.0 / 6.0)
        bits = ratio.to(tl.uint32, bitcast=True)
        code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
        scale = (code << 23).to(tl.float32, bitcast=True)
    value = tl.minimum(tl.abs(x / scale), 6.0)
    rounded = tl.where(
        value > 5,
        6.0,
        tl.where(
            value >= 3.5,
            4.0,
            tl.where(
                value > 2.5,
                3.0,
                tl.where(
                    value >= 1.75,
                    2.0,
                    tl.where(
                        value > 1.25,
                        1.5,
                        tl.where(value >= 0.75, 1.0, tl.where(value > 0.25, 0.5, 0.0)),
                    ),
                ),
            ),
        ),
    )
    sign = (x.to(tl.uint32, bitcast=True) & 0x80000000) != 0
    rounded = tl.where(sign, -rounded, rounded)
    tl.store(Y + row * D + d, rounded * scale)


class _FP4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, e4scale):
        if x.device.type != "cuda" or not torch.version.hip or x.dtype != torch.bfloat16:
            raise ValueError("V4.1 FP4 codec requires ROCm BF16 activations")
        if group not in (16, 32) or x.shape[-1] % group:
            raise ValueError("FP4 activation group must divide the last dimension")
        x = x.contiguous()
        y = torch.empty_like(x)
        with torch.cuda.device(x.device):
            _fake4[(x.numel() // x.shape[-1], x.shape[-1] // group)](
                x, y, x.shape[-1], group, e4scale, enable_fp_fusion=False
            )
        return y

    @staticmethod
    def backward(ctx, dy):
        return dy, None, None


@triton.jit
def _rotate(
    X,
    C,
    S,
    Y,
    D: tl.constexpr,
    RD: tl.constexpr,
    H: tl.constexpr,
    INVERSE: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // H
    d = tl.arange(0, B)
    rope = (d >= D - RD) & (d < D)
    v = tl.load(X + row * D + d, d < D, other=0).to(tl.float32)
    partner = tl.load(X + row * D + (d ^ 1), d < D, other=0).to(tl.float32)
    pair = (d - (D - RD)) // 2
    c = tl.load(C + token * (RD // 2) + pair, rope, other=1.0)
    s = tl.load(S + token * (RD // 2) + pair, rope, other=0.0)
    if INVERSE:
        s = -s
    a = tl.where(d % 2 == 0, v, partner)
    b = tl.where(d % 2 == 0, partner, v)
    out = tl.where(d % 2 == 0, a * c - b * s, a * s + b * c)
    tl.store(Y + row * D + d, tl.where(rope, out, v), d < D)


class _RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, inverse):
        if x.dtype != torch.bfloat16 or cos.dtype != torch.float32 or sin.dtype != torch.float32:
            raise ValueError("RoPE requires BF16 input and FP32 frequencies")
        if (
            x.device.type != "cuda"
            or not torch.version.hip
            or cos.device != x.device
            or sin.device != x.device
        ):
            raise ValueError("RoPE tensors must share an AMD ROCm GPU")
        if x.ndim not in (3, 4) or cos.shape != sin.shape or cos.shape[:2] != x.shape[:2]:
            raise ValueError("expected X[B,S,D] or X[B,S,H,D] with frequencies[B,S,RD/2]")
        rd = cos.shape[-1] * 2
        d = x.shape[-1]
        if rd > d or d % 2 or not rd:
            raise ValueError("invalid rotary width")
        h = x.shape[-2] if x.ndim == 4 else 1
        x, cos, sin = (t.contiguous() for t in (x, cos, sin))
        y = torch.empty_like(x)
        with torch.cuda.device(x.device):
            _rotate[(x.numel() // d,)](
                x, cos, sin, y, d, rd, h, inverse, triton.next_power_of_2(d), enable_fp_fusion=False
            )
        ctx.save_for_backward(cos, sin)
        ctx.h, ctx.inverse, ctx.shape = h, inverse, x.shape
        return y

    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        d = ctx.shape[-1]
        dx = torch.empty(ctx.shape, device=dy.device, dtype=dy.dtype)
        with torch.cuda.device(dy.device):
            _rotate[(dy.numel() // d,)](
                dy.contiguous(),
                cos,
                sin,
                dx,
                d,
                cos.shape[-1] * 2,
                ctx.h,
                not ctx.inverse,
                triton.next_power_of_2(d),
                enable_fp_fusion=False,
            )
        return dx, None, None, None


def fp8_roundtrip(x):
    return _FP8.apply(x)


def fp4_roundtrip(x, *, group=32, e4m3_scale=False):
    return _FP4.apply(x, group, e4m3_scale)


def rotary(x, cos, sin, *, inverse=False):
    return _RoPE.apply(x, cos, sin, inverse)
