# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""One row per program AdaLN; shared gradients fold tokens in ascending order."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from rl_engine.kernels.ops.pytorch.norm.adaln_modulation import _validate


@triton.jit
def _sum_hidden(values, BLOCK: tl.constexpr, LOG: tl.constexpr):
    col = tl.arange(0, BLOCK)
    for level in tl.static_range(LOG - 1, -1, -1):
        half = 1 << level
        upper = tl.gather(values, tl.minimum(col + half, BLOCK - 1), 0)
        values = tl.where(col < half, values + upper, values)
    return tl.sum(tl.where(col == 0, values, 0.0), 0)


@triton.jit
def _fwd(X, M, Y, G, S: tl.constexpr, H: tl.constexpr, EPS: tl.constexpr,
         BLOCK: tl.constexpr, LOG: tl.constexpr):
    row = tl.program_id(0)
    batch = row // S
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + col, col < H, other=0).to(tl.float32)
    mean = tl.div_rn(_sum_hidden(x, BLOCK, LOG), H)
    centered = tl.where(col < H, x - mean, 0.0)
    variance = tl.div_rn(_sum_hidden(centered * centered, BLOCK, LOG), H)
    rstd = tl.div_rn(1.0, libdevice.sqrt_rn(variance + EPS))
    shift = tl.load(M + batch * 3 * H + col, col < H, other=0).to(tl.float32)
    scale = tl.load(M + batch * 3 * H + H + col, col < H, other=0).to(tl.float32)
    y = (centered * rstd) * (1.0 + scale) + shift
    tl.store(Y + row * H + col, y, col < H)
    if row % S == 0:
        gate = tl.load(M + batch * 3 * H + 2 * H + col, col < H, other=0)
        tl.store(G + batch * H + col, gate, col < H)


@triton.jit
def _bwd_rows(
    X, M, DY, DX, PART_SHIFT, PART_SCALE,
    S: tl.constexpr, H: tl.constexpr, EPS: tl.constexpr,
    BLOCK: tl.constexpr, LOG: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // S
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + col, col < H, other=0).to(tl.float32)
    dy = tl.load(DY + row * H + col, col < H, other=0).to(tl.float32)
    mean = tl.div_rn(_sum_hidden(x, BLOCK, LOG), H)
    centered = tl.where(col < H, x - mean, 0.0)
    variance = tl.div_rn(_sum_hidden(centered * centered, BLOCK, LOG), H)
    rstd = tl.div_rn(1.0, libdevice.sqrt_rn(variance + EPS))
    norm = centered * rstd
    scale = tl.load(M + batch * 3 * H + H + col, col < H, other=0).to(tl.float32)
    dnorm = dy * (1.0 + scale)
    sum_dnorm = tl.div_rn(_sum_hidden(tl.where(col < H, dnorm, 0.0), BLOCK, LOG), H)
    sum_dnorm_norm = tl.div_rn(
        _sum_hidden(tl.where(col < H, dnorm * norm, 0.0), BLOCK, LOG), H
    )
    dx = (dnorm - sum_dnorm - norm * sum_dnorm_norm) * rstd
    tl.store(DX + row * H + col, dx, col < H)
    tl.store(PART_SHIFT + row * H + col, dy, col < H)
    tl.store(PART_SCALE + row * H + col, dy * norm, col < H)


@triton.jit
def _bwd_reduce(PART_SHIFT, PART_SCALE, DG, DM, S: tl.constexpr, H: tl.constexpr,
                TILE: tl.constexpr):
    batch = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    shift = tl.full((TILE,), 0.0, tl.float32)
    scale = tl.full((TILE,), 0.0, tl.float32)
    # ponytail: O(S) left fold; replace with a documented fixed tree if profiling demands it.
    for token in range(S):
        offset = (batch * S + token) * H + col
        shift += tl.load(PART_SHIFT + offset, col < H, other=0)
        scale += tl.load(PART_SCALE + offset, col < H, other=0)
    gate = tl.load(DG + batch * H + col, col < H, other=0).to(tl.float32)
    tl.store(DM + batch * 3 * H + col, shift, col < H)
    tl.store(DM + batch * 3 * H + H + col, scale, col < H)
    tl.store(DM + batch * 3 * H + 2 * H + col, gate, col < H)


class _AdaLNTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, modulation, eps):
        batch, seq, hidden = x.shape
        y = torch.empty_like(x)
        gate = torch.empty((batch, 1, hidden), device=x.device, dtype=x.dtype)
        _fwd[(batch * seq,)](
            x, modulation, y, gate, seq, hidden, eps,
            triton.next_power_of_2(hidden), triton.next_power_of_2(hidden).bit_length() - 1,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(x, modulation)
        ctx.eps = eps
        return y, gate

    @staticmethod
    def backward(ctx, grad_y, grad_gate):
        x, modulation = ctx.saved_tensors
        batch, seq, hidden = x.shape
        dx = torch.empty_like(x)
        partial_shift = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        partial_scale = torch.empty_like(partial_shift)
        dm = torch.empty_like(modulation)
        _bwd_rows[(batch * seq,)](
            x, modulation, grad_y.contiguous(), dx, partial_shift, partial_scale,
            seq, hidden, ctx.eps, triton.next_power_of_2(hidden),
            triton.next_power_of_2(hidden).bit_length() - 1, enable_fp_fusion=False,
        )
        _bwd_reduce[(batch, triton.cdiv(hidden, 128))](
            partial_shift, partial_scale, grad_gate.contiguous(), dm, seq, hidden, 128,
            enable_fp_fusion=False,
        )
        return dx, dm, None


class TritonAdaLNModulationOp:
    def __call__(self, x, modulation, *, eps=1e-6):
        return self.forward(x, modulation, eps=eps)

    def forward(self, x, modulation, *, eps=1e-6):
        _validate(x, modulation, eps)
        if not x.is_cuda:
            raise RuntimeError("Triton AdaLN requires CUDA tensors")
        return _AdaLNTriton.apply(x.contiguous(), modulation.contiguous(), float(eps))
