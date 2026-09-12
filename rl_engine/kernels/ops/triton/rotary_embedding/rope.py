# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton RoPE kernel (GPT-NeoX / HF rotate-half), matching NativeRoPEOp.

For a token at absolute position ``p`` and head_dim index ``d`` (half = D // 2)::

    inv_freq[i] = theta ** (-i / half)              i in [0, half)
    angle       = p * inv_freq[d % half]
    out[d<half]  = x[d] * cos(angle) - x[d+half] * sin(angle)
    out[d>=half] = x[d] * cos(angle) + x[d-half] * sin(angle)

cos/sin are built in fp32 with the *exact* reference math (``theta ** x``) so the
fp32 path stays bit-close to the gold even at large positions where cos/sin are
numerically sensitive; the Triton kernel does the elementwise rotation in fp32
and rounds back to the input dtype on store.

RoPE is a per-position orthogonal rotation, so the input gradient is the same
rotation with the sine negated::

    grad_x = grad_out * cos - rotate_half(grad_out) * sin

which the same kernel produces when called with ``sin_sign = -1``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rope_kernel(
    x_ptr,  # [n_rows, D] flattened input
    cos_ptr,  # [S, HALF] fp32 cosine cache
    sin_ptr,  # [S, HALF] fp32 sine cache
    out_ptr,  # [n_rows, D] output
    n_rows,
    S,
    SIN_SIGN: tl.constexpr,  # +1.0 forward, -1.0 backward
    HALF: tl.constexpr,  # D // 2, power-of-two block width
    stride_row,
    stride_d,
):
    """One program per row (a single [B, H, S] token vector of width D)."""
    row = tl.program_id(0)
    if row >= n_rows:
        return

    # Row layout is [..., S, D] contiguous, so the sequence index is row % S.
    seq_idx = row % S
    d = tl.arange(0, HALF)
    cos = tl.load(cos_ptr + seq_idx * HALF + d)
    sin = tl.load(sin_ptr + seq_idx * HALF + d) * SIN_SIGN

    base = row * stride_row
    x1 = tl.load(x_ptr + base + d * stride_d).to(tl.float32)
    x2 = tl.load(x_ptr + base + (d + HALF) * stride_d).to(tl.float32)

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + base + d * stride_d, out1.to(out_dtype))
    tl.store(out_ptr + base + (d + HALF) * stride_d, out2.to(out_dtype))


def _build_cos_sin(positions: Tensor, half: int, theta: float, device: torch.device):
    """fp32 cos/sin caches of shape [S, half], identical math to NativeRoPEOp."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    pos = positions.to(device=device, dtype=torch.float32).reshape(-1, 1)
    freqs = pos * inv_freq  # [S, half]
    return freqs.cos().contiguous(), freqs.sin().contiguous()


def _launch_rope(x: Tensor, cos: Tensor, sin: Tensor, S: int, sin_sign: float) -> Tensor:
    D = x.shape[-1]
    half = D // 2
    x_2d = x.contiguous().reshape(-1, D)
    n_rows = x_2d.shape[0]

    out = torch.empty_like(x_2d)
    grid = (n_rows,)
    _rope_kernel[grid](
        x_2d,
        cos,
        sin,
        out,
        n_rows,
        S,
        SIN_SIGN=float(sin_sign),
        HALF=half,
        stride_row=x_2d.stride(0),
        stride_d=x_2d.stride(1),
    )
    return out.reshape(x.shape)


def _rope_table(x: Tensor, positions: Tensor, theta: float) -> tuple[Tensor, Tensor, Tensor, int]:
    """Build (x_2d, cos, sin, table_len) for [S] or [B, S] positions.

    ``[B, H, S, D]`` + ``[B, S]`` is permuted to ``[H, B, S, D]`` so the existing
    ``row % table_len`` index equals ``b * S + s``.
    """
    D = x.shape[-1]
    if D % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {D}")
    if positions.dim() == 1:
        table_len = int(positions.shape[0])
        n_rows = x.numel() // D
        if n_rows % table_len != 0:
            raise ValueError(
                f"row count {n_rows} not divisible by seq length {table_len}; "
                "expected a [..., S, D] contiguous layout."
            )
        x_2d = x.contiguous().reshape(-1, D)
        cos, sin = _build_cos_sin(positions, D // 2, float(theta), x.device)
        return x_2d, cos, sin, table_len
    if positions.dim() != 2:
        raise ValueError(f"positions must be [S] or [B, S], got shape {tuple(positions.shape)}")
    batch, seq = positions.shape
    if x.shape[0] != batch or x.shape[-2] != seq:
        raise ValueError(
            f"positions {tuple(positions.shape)} is incompatible with x {tuple(x.shape)}; "
            "expected x [B, ..., S, D]"
        )
    if x.dim() == 4:
        x_2d = x.permute(1, 0, 2, 3).contiguous().reshape(-1, D)
    elif x.dim() == 3:
        x_2d = x.contiguous().reshape(-1, D)
    else:
        raise ValueError(
            f"RoPE [B, S] positions require x [B, S, D] or [B, H, S, D], got {x.dim()}D"
        )
    table_len = batch * seq
    if x_2d.shape[0] % table_len != 0:
        raise ValueError(f"row count {x_2d.shape[0]} not divisible by B*S={table_len}")
    cos, sin = _build_cos_sin(positions.reshape(-1), D // 2, float(theta), x.device)
    return x_2d, cos, sin, table_len


def _restore_rope(out_2d: Tensor, x: Tensor, positions: Tensor) -> Tensor:
    if positions.dim() == 1 or x.dim() != 4:
        return out_2d.reshape(x.shape)
    heads, batch, seq, dim = x.shape[1], x.shape[0], x.shape[2], x.shape[3]
    return out_2d.reshape(heads, batch, seq, dim).permute(1, 0, 2, 3).contiguous()


class _RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, positions: Tensor, theta: float) -> Tensor:
        x_2d, cos, sin, table_len = _rope_table(x, positions, theta)
        ctx.save_for_backward(cos, sin)
        ctx.seq_len = table_len
        ctx.x_shape = tuple(x.shape)
        ctx.pos_dim = positions.dim()
        out_2d = _launch_rope(x_2d, cos, sin, table_len, sin_sign=1.0)
        return _restore_rope(out_2d, x, positions)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        cos, sin = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            if ctx.pos_dim == 2 and len(ctx.x_shape) == 4:
                grad_2d = grad_out.permute(1, 0, 2, 3).contiguous().reshape(-1, ctx.x_shape[-1])
                out_2d = _launch_rope(grad_2d, cos, sin, ctx.seq_len, sin_sign=-1.0)
                heads, batch, seq, dim = (
                    ctx.x_shape[1],
                    ctx.x_shape[0],
                    ctx.x_shape[2],
                    ctx.x_shape[3],
                )
                grad_x = out_2d.reshape(heads, batch, seq, dim).permute(1, 0, 2, 3).contiguous()
            else:
                grad_x = _launch_rope(grad_out, cos, sin, ctx.seq_len, sin_sign=-1.0)
        return grad_x, None, None


class TritonRoPEOp:
    """Triton RoPE op (GPT-NeoX rotate-half), differentiable w.r.t. ``x``.

    Qwen3 defaults: theta=1e6, head_dim=128, full-dimension rotation. cos/sin are
    computed in fp32 from ``positions`` and ``theta`` (matching the reference) and
    the rotation runs in a Triton kernel -- no external cos/sin cache is accepted.
    """

    op_class = "elementwise"

    def __call__(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        return self.forward(x, positions, theta=theta)

    def forward(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        if x.device.type not in ("cuda", "hip", "xpu", "musa"):
            raise RuntimeError(f"TritonRoPEOp requires a GPU tensor, got device '{x.device}'.")
        return _RoPEFunction.apply(x, positions, theta)
