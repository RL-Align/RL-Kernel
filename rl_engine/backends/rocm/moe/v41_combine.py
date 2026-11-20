# SPDX-License-Identifier: Apache-2.0
"""Top-six combine in ascending original expert ID, shared expert added last."""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(X, S, Y, D: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * B + tl.arange(0, B)
    out = tl.full((B,), 0, tl.float32)
    for slot in range(6):
        out = out + tl.load(X + (t * 6 + slot) * D + d, d < D, other=0).to(tl.float32)
    out = out + tl.load(S + t * D + d, d < D, other=0).to(tl.float32)
    tl.store(Y + t * D + d, out, d < D)


@triton.jit
def _backward(DY, DX, DS, D: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * B + tl.arange(0, B)
    dy = tl.load(DY + t * D + d, d < D, other=0)
    for slot in range(6):
        tl.store(DX + (t * 6 + slot) * D + d, dy, d < D)
    tl.store(DS + t * D + d, dy, d < D)


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shared):
        if x.ndim != 3 or x.shape[1] != 6 or shared.shape != (x.shape[0], x.shape[2]):
            raise ValueError("expected expert outputs[T,6,D] ordered by original expert ID")
        if (
            x.dtype != torch.bfloat16
            or shared.dtype != x.dtype
            or x.device != shared.device
            or not torch.version.hip
            or x.device.type != "cuda"
        ):
            raise ValueError("combine requires BF16 expert outputs on one AMD GPU")
        x, shared = x.contiguous(), shared.contiguous()
        y = torch.empty_like(shared)
        ctx.shape = x.shape
        with torch.cuda.device(x.device):
            _forward[(x.shape[0], triton.cdiv(x.shape[2], 256))](
                x, shared, y, x.shape[2], 256, enable_fp_fusion=False
            )
        return y

    @staticmethod
    def backward(ctx, dy):
        dx = torch.empty(ctx.shape, device=dy.device, dtype=dy.dtype)
        ds = torch.empty_like(dy)
        with torch.cuda.device(dy.device):
            _backward[(dy.shape[0], triton.cdiv(dy.shape[1], 256))](
                dy.contiguous(), dx, ds, dy.shape[1], 256
            )
        return dx, ds


def combine(expert_outputs, shared):
    return _Combine.apply(expert_outputs, shared)
