# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Indexed NVIDIA AdaLN: fixed H tree, FP32 selected-token left folds."""

import torch
import triton
import triton.language as tl

from rl_engine.reference.norm.adaln_modulation import (
    _indexed_domain_reason,
    _int32_addressable,
    _validate_select01,
    _validate_select01_metadata,
)

from .adaln_modulation import _center_and_rstd, _sum_hidden


@triton.jit
def _rows(
    X,
    M,
    INDEX,
    Y,
    G,
    DY,
    DX,
    PS,
    PC,
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BROADCAST: tl.constexpr,
    BACKWARD: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG: tl.constexpr,
):
    row = tl.program_id(0)
    bank = tl.load(INDEX + (row % S if BROADCAST else row))
    bank_row = row // S + bank * B
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + col, col < H, other=0).to(tl.float32)
    centered, rstd = _center_and_rstd(x, H, EPS, BLOCK, LOG)
    norm = centered * rstd
    offset = bank_row * 3 * H + col
    scale = tl.load(M + offset + H, col < H, other=0).to(tl.float32)
    if BACKWARD:
        dy = tl.load(DY + row * H + col, col < H, other=0).to(tl.float32)
        dnorm = dy * (1.0 + scale)
        mean = tl.div_rn(_sum_hidden(tl.where(col < H, dnorm, 0.0), BLOCK, LOG), H)
        mean_norm = tl.div_rn(_sum_hidden(tl.where(col < H, dnorm * norm, 0.0), BLOCK, LOG), H)
        dx = (dnorm - mean - norm * mean_norm) * rstd
        tl.store(DX + row * H + col, dx, col < H)
        tl.store(PS + row * H + col, dy, col < H)
        # The FP32 store fixes the product's rounding before the token fold.
        tl.store(PC + row * H + col, dy * norm, col < H)
    else:
        shift = tl.load(M + offset, col < H, other=0).to(tl.float32)
        tl.store(Y + row * H + col, norm * (1.0 + scale) + shift, col < H)
        # Integer loads/stores preserve signaling NaNs without a floating conversion.
        if M.dtype.element_ty == tl.bfloat16:
            raw = tl.load((M + offset + 2 * H).to(tl.pointer_type(tl.uint16)), col < H, other=0)
            tl.store((G + row * H + col).to(tl.pointer_type(tl.uint16)), raw, col < H)
        else:
            raw = tl.load((M + offset + 2 * H).to(tl.pointer_type(tl.uint32)), col < H, other=0)
            tl.store((G + row * H + col).to(tl.pointer_type(tl.uint32)), raw, col < H)


@triton.jit
def _fold(
    PS,
    PC,
    DG,
    INDEX,
    DM,
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BROADCAST: tl.constexpr,
    HAS_Y: tl.constexpr,
    HAS_GATE: tl.constexpr,
    TILE: tl.constexpr,
):
    sample = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    # Both banks use independent +0 FP32 accumulators; no masked zero additions.
    for bank in tl.static_range(2):
        shift = tl.full((TILE,), 0.0, tl.float32)
        scale = tl.full((TILE,), 0.0, tl.float32)
        gate = tl.full((TILE,), 0.0, tl.float32)
        for token in range(S):
            row = sample * S + token
            selected = tl.load(INDEX + (token if BROADCAST else row))
            if selected == bank:
                offset = row * H + col
                if HAS_Y:
                    shift = shift + tl.load(PS + offset, col < H, other=0)
                    scale = scale + tl.load(PC + offset, col < H, other=0)
                if HAS_GATE:
                    gate = gate + tl.load(DG + offset, col < H, other=0).to(tl.float32)
        offset = (sample + bank * B) * 3 * H + col
        tl.store(DM + offset, shift, col < H)
        tl.store(DM + offset + H, scale, col < H)
        tl.store(DM + offset + 2 * H, gate, col < H)


class _IndexedTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, m, index, eps, num_warps, tile):
        ctx.set_materialize_grads(False)
        b, s, h = x.shape
        y, gate = torch.empty_like(x), torch.empty_like(x)
        block = triton.next_power_of_2(h)
        _rows[(b * s,)](
            x,
            m,
            index,
            y,
            gate,
            None,
            None,
            None,
            None,
            b,
            s,
            h,
            eps,
            index.shape[0] == 1,
            False,
            block,
            block.bit_length() - 1,
            num_warps=num_warps,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
        ctx.save_for_backward(x, m, index)
        ctx.eps, ctx.num_warps, ctx.tile = eps, num_warps, tile
        return y, gate

    @staticmethod
    def backward(ctx, dy, dg):
        x, m, index = ctx.saved_tensors
        # Reject upstream spans before any copies or temporary allocations.
        for grad in (dy, dg):
            if grad is not None and not _int32_addressable(grad):
                raise ValueError("indexed Triton upstream requires int32 indexing")
        dy = None if dy is None else dy.contiguous()
        dg = None if dg is None else dg.contiguous()
        b, s, h = x.shape
        dx = ps = pc = None
        if dy is not None:
            dx = torch.empty_like(x)
            ps = torch.empty(x.shape, dtype=torch.float32, device=x.device)
            pc = torch.empty_like(ps)
            block = triton.next_power_of_2(h)
            _rows[(b * s,)](
                x,
                m,
                index,
                None,
                None,
                dy,
                dx,
                ps,
                pc,
                b,
                s,
                h,
                ctx.eps,
                index.shape[0] == 1,
                True,
                block,
                block.bit_length() - 1,
                num_warps=ctx.num_warps,
                enable_fp_fusion=False,
                enable_reflect_ftz=False,
            )
        dm = torch.empty_like(m)
        _fold[(b, triton.cdiv(h, ctx.tile))](
            ps,
            pc,
            dg,
            index,
            dm,
            b,
            s,
            h,
            index.shape[0] == 1,
            dy is not None,
            dg is not None,
            ctx.tile,
            num_warps=ctx.num_warps,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
        return dx, dm, None, None, None, None


def indexed_forward(x, modulation, index, eps, num_warps, tile):
    _validate_select01_metadata(x, modulation, index, eps)
    reason = _indexed_domain_reason(x, modulation, index, eps)
    if reason in {"int32 indexing", "B <= 65535"}:
        raise ValueError(f"indexed Triton requires {reason}")
    if reason is not None:
        raise RuntimeError(f"indexed Triton requires {reason}")
    _validate_select01(x, modulation, index, eps)
    return _IndexedTriton.apply(
        x.contiguous(),
        modulation.contiguous(),
        index.to(torch.int32).contiguous(),
        float(eps),
        num_warps,
        tile,
    )
