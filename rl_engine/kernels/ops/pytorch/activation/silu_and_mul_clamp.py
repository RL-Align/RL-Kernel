# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Clamped SwiGLU of the DSv4 shared expert, matching vLLM's rollout bytes.

vLLM's ``SiluAndMulWithClamp(swiglu_limit)`` (CUDA op
``silu_and_mul_with_clamp``, alpha 1, beta 0) on ``z = [gate | up]``, BF16:

    g   = fminf(gate, limit)                 # NaN gate -> limit
    u   = fmaxf(fminf(up, limit), -limit)
    act = bf16(g / (1 + expf(-g)))           # first rounding
    out = bf16(act * (u + 0.0f))             # second rounding; FP32 product exact

DeepSeek-V4-Flash uses ``swiglu_limit = 10``. The forward is written with the
same FP32 operations as the CUDA kernel; on CUDA, ``torch.exp`` is libdevice
``expf`` like vLLM's, so the bytes match there (on CPU ``exp`` can differ in
the last bit). Division is tensor / tensor because CUDA PyTorch multiplies by
the reciprocal when dividing by a Python scalar.

Note: the P5 oracle's unweighted shared-expert SwiGLU applies no clamp and
rounds once from FP32; this operator follows the rollout instead.

vLLM has no backward. Ours is deterministic and elementwise, in FP32:

    dgate = dh * u * dsilu(g)     where gate < limit, else 0
    dup   = dh * act              where -limit < up < limit, else 0
    dsilu = sig * (1 + g * (1 - sig)),  sig = 1 / (1 + exp(-g))

and returns ``dz = [dgate | dup]`` in BF16 for the dgrad GEMM.
"""

from __future__ import annotations

import torch
from torch import Tensor

SWIGLU_LIMIT = 10.0


def _split(z: Tensor) -> tuple[Tensor, Tensor]:
    if type(z) is not Tensor or z.ndim != 2:
        raise ValueError("z must be a 2-D tensor [T, 2F].")
    if z.dtype != torch.bfloat16:
        raise TypeError(f"z must be BF16, got {z.dtype}.")
    if z.shape[1] % 2:
        raise ValueError("z's last dimension must be even ([gate | up]).")
    half = z.shape[1] // 2
    return z[:, :half].float(), z[:, half:].float()


def _clamped(gate: Tensor, up: Tensor, limit: float) -> tuple[Tensor, Tensor]:
    bound = gate.new_tensor(limit)
    g = torch.fmin(gate, bound)
    u = torch.fmax(torch.fmin(up, bound), -bound)
    return g, u


def _silu(g: Tensor) -> Tensor:
    denominator = torch.exp(-g) + 1.0
    return g / denominator


@torch.no_grad()
def silu_and_mul_clamp(z: Tensor, limit: float = SWIGLU_LIMIT) -> Tensor:
    """``z [T, 2F]`` BF16 -> ``[T, F]`` BF16, vLLM ``silu_and_mul_with_clamp``."""
    gate, up = _split(z)
    g, u = _clamped(gate, up, limit)
    act = _silu(g).to(torch.bfloat16)
    return (act.float() * (u + 0.0)).to(torch.bfloat16)


@torch.no_grad()
def silu_and_mul_clamp_backward(dh: Tensor, z: Tensor, limit: float = SWIGLU_LIMIT) -> Tensor:
    """``dh [T, F]``, ``z [T, 2F]`` -> ``dz [T, 2F]`` BF16."""
    gate, up = _split(z)
    if dh.shape != gate.shape:
        raise ValueError(f"dh shape {tuple(dh.shape)} != {tuple(gate.shape)}.")
    g, u = _clamped(gate, up, limit)
    dh32 = dh.float()
    sig = 1.0 / (torch.exp(-g) + 1.0)
    dsilu = sig * (1.0 + g * (1.0 - sig))
    act = _silu(g).to(torch.bfloat16).float()
    zero = torch.zeros_like(dh32)
    dgate = torch.where(gate < limit, (dh32 * u) * dsilu, zero)
    dup = torch.where((up > -limit) & (up < limit), dh32 * act, zero)
    return torch.cat([dgate, dup], dim=1).to(torch.bfloat16)
