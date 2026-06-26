# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton fused masking + variable-length packing (pack-and-pad) op (issue #42).

Packs the active rows of a dense ``[B, S, *tail]`` tensor (selected by a
``[B, S]`` mask) into a contiguous ``[Total_Active, *tail]`` tensor, and scatters
gradients back to the dense layout on the backward pass. The packing order is
row-major over the flattened ``[B, S]`` grid, matching ``NativePackOp`` (the
numerical contract for this op).

The active-token destination indices are computed with a cheap exclusive
prefix-sum over the flattened mask (small ``[B*S]`` tensor); the heavy
``tail``-vector movement runs in a Triton gather kernel, and the backward is a
symmetric scatter kernel. ``cu_seqlens`` (per-row prefix-sum) is returned for
varlen consumers, identical to the native op.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

# Tail-vector tile width.
_BLOCK_T = 1024


@triton.jit
def _pack_gather_kernel(
    src_ptr,  # [n_rows, T] dense, flattened over (B, S)
    dst_ptr,  # [n_active, T] packed
    dest_ptr,  # [n_rows] int64: packed row index for an active row, else -1
    T,
    BLOCK_T: tl.constexpr,
):
    """One program per (source row, tail-tile). Active rows copy their tail
    vector into the packed buffer at ``dest_ptr[row]``; inactive rows are skipped."""
    row = tl.program_id(0)
    dest = tl.load(dest_ptr + row)
    if dest >= 0:
        t0 = tl.program_id(1) * BLOCK_T
        cols = t0 + tl.arange(0, BLOCK_T)
        cmask = cols < T
        src = tl.load(src_ptr + row.to(tl.int64) * T + cols, mask=cmask, other=0.0)
        tl.store(dst_ptr + dest.to(tl.int64) * T + cols, src, mask=cmask)


@triton.jit
def _pack_scatter_kernel(
    grad_packed_ptr,  # [n_active, T]
    grad_src_ptr,  # [n_rows, T], pre-zeroed
    dest_ptr,  # [n_rows] int64
    T,
    BLOCK_T: tl.constexpr,
):
    """Backward: scatter the packed gradient back to the active source rows.
    Inactive rows stay zero (grad_src is pre-zeroed)."""
    row = tl.program_id(0)
    dest = tl.load(dest_ptr + row)
    if dest >= 0:
        t0 = tl.program_id(1) * BLOCK_T
        cols = t0 + tl.arange(0, BLOCK_T)
        cmask = cols < T
        g = tl.load(grad_packed_ptr + dest.to(tl.int64) * T + cols, mask=cmask, other=0.0)
        tl.store(grad_src_ptr + row.to(tl.int64) * T + cols, g, mask=cmask)


def _dest_index(flat_mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Map each flattened row to its packed destination index (active rows get an
    exclusive prefix-sum position; inactive rows get -1). Returns (dest, n_active)."""
    active = flat_mask.to(torch.bool)
    counts = active.to(torch.int64)
    # Exclusive prefix sum: position of each active row in the packed buffer.
    excl = torch.cumsum(counts, dim=0) - counts
    n_active = int(counts.sum().item())
    dest = torch.where(active, excl, torch.full_like(excl, -1))
    return dest.contiguous(), n_active


class _PackFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, mask: torch.Tensor):
        lead = mask.dim()
        tail_shape = x.shape[lead:]
        n_rows = 1
        for s in mask.shape:
            n_rows *= int(s)
        T = 1
        for s in tail_shape:
            T *= int(s)

        src = x.reshape(n_rows, T).contiguous()
        flat_mask = mask.reshape(-1)
        dest, n_active = _dest_index(flat_mask)

        packed = torch.empty(n_active, T, device=x.device, dtype=x.dtype)
        if n_active > 0:
            grid = (n_rows, triton.cdiv(T, _BLOCK_T))
            _pack_gather_kernel[grid](src, packed, dest, T, BLOCK_T=_BLOCK_T)

        per_row_active = flat_mask.to(torch.bool).reshape(mask.shape[0], -1).to(torch.int64).sum(1)
        cu_seqlens = torch.zeros(mask.shape[0] + 1, dtype=torch.int64, device=x.device)
        torch.cumsum(per_row_active, dim=0, out=cu_seqlens[1:])

        ctx.save_for_backward(dest)
        ctx.n_rows = n_rows
        ctx.T = T
        ctx.x_shape = tuple(x.shape)
        ctx.x_dtype = x.dtype
        out_tail = tuple(tail_shape)
        packed_out = packed.reshape(n_active, *out_tail) if out_tail else packed.reshape(n_active)
        return packed_out, cu_seqlens

    @staticmethod
    def backward(ctx, grad_packed: torch.Tensor, grad_cu_seqlens):
        (dest,) = ctx.saved_tensors
        n_rows, T = ctx.n_rows, ctx.T
        n_active = grad_packed.shape[0]

        gp = grad_packed.reshape(n_active, T).contiguous()
        grad_src = torch.zeros(n_rows, T, device=grad_packed.device, dtype=grad_packed.dtype)
        if n_active > 0:
            grid = (n_rows, triton.cdiv(T, _BLOCK_T))
            _pack_scatter_kernel[grid](gp, grad_src, dest, T, BLOCK_T=_BLOCK_T)

        grad_x = grad_src.reshape(ctx.x_shape)
        return grad_x, None


class TritonPackOp:
    """Triton fused masking + variable-length packing (pack-and-pad).

    Forward packs the active rows of ``x`` (selected by ``mask``) into a
    contiguous ``[Total_Active, *tail]`` tensor and returns the per-row
    ``cu_seqlens`` prefix-sum. Backward scatters the upstream gradient back to the
    original ``[*mask.shape, *tail]`` layout, leaving zeros at inactive positions.
    Numerically identical to ``NativePackOp``; CUDA & ROCm via Triton.
    """

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.device.type not in ("cuda", "xpu", "hip"):
            raise RuntimeError(
                "TritonPackOp requires a GPU tensor (CUDA / ROCm / XPU), got "
                f"device '{x.device}'."
            )
        if mask.dim() < 1:
            raise ValueError("mask must have at least one dimension.")
        if mask.shape != x.shape[: mask.dim()]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match the leading dims of "
                f"x.shape {tuple(x.shape)} (expected {tuple(x.shape[: mask.dim()])})."
            )
        return _PackFunction.apply(x, mask)
