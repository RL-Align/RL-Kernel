# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""FP32 golden of one MiniMax-H3 transformer block (RFC #420 ``ws1_one_h3_block``).

Every node takes FP32 inputs (BF16 weights and activations upcast exactly),
computes in FP32 with TF32 off, and returns FP32 without rounding back to
BF16, so the golden is the block's arithmetic without the BF16 storage the
provider and the candidate both have. It is the yardstick their errors are
measured against, not a bitwise target. The one declared cast kept from the
checkpoint contract is the AdaLN input: ``silu(temb)`` is rounded to BF16
before the projection, as every implementation must do it.

The partial MM-RoPE uses exact FP32 ``cos``/``sin`` tables on the leading 96
of 128 head channels (the provider rounds the tables to BF16 first).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

H3_HEADS = 56
H3_HEAD_DIM = 128
H3_ROTARY_DIM = 96
H3_NORM_EPS = 1e-5


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32)


def golden_adaln_tables(temb, weight, bias, hidden_size: int = 5376) -> tuple[torch.Tensor, ...]:
    """Six ``(3T, H)`` FP32 tables; ``silu(temb)`` rounded to the weight dtype first."""

    act = F.silu(_f32(temb)).to(weight.dtype)
    table = F.linear(_f32(act), _f32(weight), _f32(bias))
    return table.view(-1, 6 * hidden_size).chunk(6, dim=-1)


def golden_rmsnorm(x, weight, eps: float = H3_NORM_EPS) -> torch.Tensor:
    x = _f32(x)
    rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return x * rstd * _f32(weight)


def golden_norm_modulate(x, weight, shift, scale, index, eps: float = H3_NORM_EPS):
    n = golden_rmsnorm(x, weight, eps)
    return n * (1.0 + _f32(scale).index_select(0, index)) + _f32(shift).index_select(0, index)


def golden_linear(x, weight) -> torch.Tensor:
    return F.linear(_f32(x), _f32(weight))


def golden_rope_tables(position_ids, rope_freq_dim: int = 16, rope_theta: float = 10000.0):
    """FP32 ``(S, 96)`` cos/sin, the same angles as the provider's rotary module."""

    exponent = torch.arange(
        0, 2 * rope_freq_dim, 2, dtype=torch.float32, device=position_ids.device
    )
    inv_freq = 1.0 / (rope_theta ** (exponent / (2 * rope_freq_dim)))
    freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)
    freqs = torch.cat(freqs.unbind(dim=1), dim=-1)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()


def golden_apply_rotary(x, cos, sin) -> torch.Tensor:
    """Rotate-half on the leading ``cos.shape[-1]`` channels of ``(B, S, H, D)``; FP32 tables."""

    x = _f32(x)
    rot = cos.shape[-1]
    head, tail = x[..., :rot], x[..., rot:]
    x1, x2 = head.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    head = head * cos[None, :, None, :] + rotated * sin[None, :, None, :]
    return torch.cat((head, tail), dim=-1)


def golden_attention(q, k, v) -> torch.Tensor:
    """Non-causal full attention over one packed document, ``(B, S, H, D)`` -> ``(B, S, H*D)``."""

    q, k, v = (_f32(t).permute(0, 2, 1, 3) for t in (q, k, v))
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    out = torch.matmul(torch.softmax(scores, dim=-1), v)
    return out.permute(0, 2, 1, 3).flatten(2, 3)


def golden_swiglu(projected) -> torch.Tensor:
    hidden, gate = _f32(projected).chunk(2, dim=-1)
    return hidden * F.silu(gate)


def golden_gate_residual(residual, gate, index, y) -> torch.Tensor:
    return _f32(residual) + _f32(gate).index_select(0, index) * _f32(y)
