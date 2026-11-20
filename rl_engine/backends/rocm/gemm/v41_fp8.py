# SPDX-License-Identifier: Apache-2.0
"""V4.1 block32 OCP FP8 projection and input STE, packed weights frozen.

MI300X decodes OCP bytes to exactly representable BF16 tile values before MFMA;
no FNUZ reinterpretation or persistent dequantized weight cache is used. Each
32-wide product is scaled then accumulated in ascending K-block order.
"""

import torch
import triton
import triton.language as tl

from rl_engine.contracts.operators.expert_training import QuantizedExpertActivation
from rl_engine.kernels.ops.triton.moe.fused_mlp import _decode_e4m3, _e8m0_scale


@triton.jit
def _quant32(X, C, S, K: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    k = block * 32 + tl.arange(0, 32)
    x = tl.load(X + row * K + k).to(tl.float32)
    scaled = tl.maximum(tl.max(tl.abs(x), 0), 1e-4) * (1.0 / 448.0)
    bits = scaled.to(tl.uint32, bitcast=True)
    code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    inv = ((254 - code) << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(x * inv, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(C + row * K + k, q.to(tl.uint8, bitcast=True))
    tl.store(S + row * (K // 32) + block, code.to(tl.uint8))


def quantize_fp8_block32(x):
    if not torch.version.hip or x.device.type != "cuda":
        raise RuntimeError("V4.1 FP8 quantization requires AMD ROCm")
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[-1] == 0 or x.shape[-1] % 32:
        raise ValueError("expected BF16 [M,K] with positive block32-aligned K")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("activation quantization requires finite inputs")
    x = x.contiguous()
    codes = torch.empty_like(x, dtype=torch.uint8)
    scales = torch.empty((x.shape[0], x.shape[1] // 32), device=x.device, dtype=torch.uint8)
    with torch.cuda.device(x.device):
        if x.shape[0]:
            _quant32[(x.shape[0], x.shape[1] // 32)](
                x, codes, scales, x.shape[1], enable_fp_fusion=False
            )
    return QuantizedExpertActivation(codes, scales)


@triton.jit
def _gemm(AC, AS, WC, WS, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.program_id(0) * 16 + tl.arange(0, 16)
    n = tl.program_id(1) * 32 + tl.arange(0, 32)
    kk = tl.arange(0, 32)
    acc = tl.zeros((16, 32), tl.float32)
    for block in range(K // 32):
        k = block * 32 + kk
        a = _decode_e4m3(tl.load(AC + m[:, None] * K + k[None, :], m[:, None] < M, other=0)).to(
            tl.bfloat16
        )
        w = _decode_e4m3(tl.load(WC + n[None, :] * K + k[:, None], n[None, :] < N, other=0)).to(
            tl.bfloat16
        )
        sa = _e8m0_scale(tl.load(AS + m * (K // 32) + block, m < M, other=127))
        sw = _e8m0_scale(tl.load(WS + (n // 32) * (K // 32) + block, n < N, other=127))
        prod = tl.dot(a, w)
        acc = acc + (prod * sa[:, None]) * sw[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _vjp(DY, WC, WS, DX, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.program_id(0) * 16 + tl.arange(0, 16)
    k = tl.program_id(1) * 32 + tl.arange(0, 32)
    nn = tl.arange(0, 32)
    acc = tl.zeros((16, 32), tl.float32)
    for block in range(triton.cdiv(N, 32)):
        n = block * 32 + nn
        dy = tl.load(
            DY + m[:, None] * N + n[None, :], (m[:, None] < M) & (n[None, :] < N), other=0
        ).to(tl.bfloat16)
        w = _decode_e4m3(tl.load(WC + n[:, None] * K + k[None, :], n[:, None] < N, other=0))
        sw = _e8m0_scale(
            tl.load(
                WS + (n[:, None] // 32) * (K // 32) + k[None, :] // 32, n[:, None] < N, other=127
            )
        )
        weight = (w * sw).to(tl.bfloat16)
        acc = tl.dot(dy, weight, acc)
    tl.store(DX + m[:, None] * K + k[None, :], acc, m[:, None] < M)


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codes, scales):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        q = quantize_fp8_block32(flat)
        n, k = codes.shape
        out = torch.empty((flat.shape[0], n), device=x.device, dtype=torch.bfloat16)
        with torch.cuda.device(x.device):
            if flat.shape[0]:
                _gemm[(triton.cdiv(flat.shape[0], 16), triton.cdiv(n, 32))](
                    q.codes,
                    q.scales,
                    codes,
                    scales,
                    out,
                    flat.shape[0],
                    n,
                    k,
                    enable_fp_fusion=False,
                    num_warps=4,
                )
        ctx.save_for_backward(codes, scales)
        ctx.shape = shape
        return out.reshape(*shape[:-1], n)

    @staticmethod
    def backward(ctx, dy):
        c, s = ctx.saved_tensors
        n, k = c.shape
        dy = dy.reshape(-1, n).contiguous()
        dx = torch.empty((dy.shape[0], k), device=dy.device, dtype=torch.bfloat16)
        with torch.cuda.device(dy.device):
            if dy.shape[0]:
                _vjp[(triton.cdiv(dy.shape[0], 16), triton.cdiv(k, 32))](
                    dy, c, s, dx, dy.shape[0], n, k, enable_fp_fusion=False, num_warps=4
                )
        return dx.reshape(ctx.shape), None, None


class V41FP8Linear(torch.nn.Module):
    profile = "dsv41-ocp-fp8-block32-fixed-k-bf16-output-input-ste-v1"

    def __init__(self, weight, scales):
        super().__init__()
        if weight.ndim != 2 or weight.shape[1] == 0 or weight.shape[1] % 32:
            raise ValueError("expected block32-aligned FP8 weight[N,K]")
        n, k = weight.shape
        if scales.shape != (triton.cdiv(n, 32), k // 32):
            raise ValueError("expected original 32x32 weight scales")
        if weight.dtype != torch.float8_e4m3fn or scales.dtype != torch.float8_e8m0fnu:
            raise TypeError("expected original OCP E4M3/E8M0 weights/scales")
        if not torch.version.hip or weight.device.type != "cuda" or scales.device != weight.device:
            raise ValueError("weights/scales must be colocated on AMD ROCm")
        if not torch.isfinite(weight.float()).all() or not torch.isfinite(scales.float()).all():
            raise ValueError("nonfinite checkpoint weights or scales")
        self.register_buffer("codes", weight.view(torch.uint8).contiguous())
        self.register_buffer("scales", scales.view(torch.uint8).contiguous())

    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.codes.shape[1] or x.device != self.codes.device:
            raise ValueError("projection input shape/device mismatch")
        return _Linear.apply(x, self.codes, self.scales)
