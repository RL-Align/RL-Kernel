# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4.1 Engram gate: ``csrc/cuda/engram/dsv41_engram.cu`` (profile ``dsv41-engram-v1``).

The forward is byte-equal to ``rl_engine.kernels.ops.pytorch.engram`` and batch-invariant;
the autograd function returns dX of the streams only (tables, ``wkv``, ``q`` and ``k``
frozen, so ``kv`` gets no gradient).
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor
from torch.version import hip as torch_hip

from rl_engine.kernels.ops.pytorch.engram import CLAMP, NORM_EPS

PROFILE = "dsv41-engram-v1"
_SYMBOLS = ("dsv41_engram_gate", "dsv41_engram_gate_backward")


def _extension():
    if torch_hip is not None or not torch.cuda.is_available():
        raise RuntimeError("DSv4.1 Engram requires NVIDIA CUDA")
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError as exc:
        raise RuntimeError("Build the CUDA extension first") from exc
    missing = [name for name in _SYMBOLS if not hasattr(ext, name)]
    if missing:
        raise RuntimeError(f"rl_engine._C lacks {missing}; rebuild the CUDA extension")
    return ext


def _bf16(w: Tensor) -> Tensor:
    return w.to(torch.bfloat16).contiguous()


def engram_gate_cuda(
    x: Tensor,
    kv: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    mask: Tensor | None = None,
    eps: float = NORM_EPS,
    clamp: float = CLAMP,
) -> tuple[Tensor, Tensor]:
    """streams BF16 [T, 4, H], kv FP32 [T, 5 H] -> (out BF16 [T, 4, H], gate FP32 [T, 4])."""
    out, gate = _extension().dsv41_engram_gate(
        x.contiguous(),
        kv.float(),
        _bf16(q_weight),
        _bf16(k_weight),
        None if mask is None else mask.bool().contiguous(),
        float(eps),
        float(clamp),
    )
    return out, gate


def engram_gate_backward_cuda(
    dout: Tensor,
    x: Tensor,
    kv: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    mask: Tensor | None = None,
    eps: float = NORM_EPS,
    clamp: float = CLAMP,
    out_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """dX [T, 4, H] in ``out_dtype`` (BF16 or FP32)."""
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"out_dtype must be BF16 or FP32, got {out_dtype}")
    g = dout if dout.dtype in (torch.bfloat16, torch.float32) else dout.float()
    return _extension().dsv41_engram_gate_backward(
        g.contiguous(),
        x.contiguous(),
        kv.float(),
        _bf16(q_weight),
        _bf16(k_weight),
        None if mask is None else mask.bool().contiguous(),
        float(eps),
        float(clamp),
        out_dtype == torch.bfloat16,
    )


class EngramGateFunction(torch.autograd.Function):
    """(x, kv, q, k, mask) -> out; dX only."""

    @staticmethod
    def forward(ctx, x, kv, q_weight, k_weight, mask, eps):
        out, _ = engram_gate_cuda(x, kv, q_weight, k_weight, mask, eps)
        ctx.eps = eps
        ctx.has_mask = mask is not None
        ctx.save_for_backward(x, kv, q_weight, k_weight, mask)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, kv, q_weight, k_weight, mask = ctx.saved_tensors
        dx = engram_gate_backward_cuda(
            dout,
            x,
            kv,
            q_weight,
            k_weight,
            mask if ctx.has_mask else None,
            ctx.eps,
            out_dtype=x.dtype,
        )
        return dx, None, None, None, None, None


def engram_gate(
    x: Tensor,
    kv: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    mask: Tensor | None = None,
    eps: float = NORM_EPS,
) -> Tensor:
    """Autograd Engram gate: BF16 [T, 4, H] (dX only)."""
    return EngramGateFunction.apply(x, kv, q_weight, k_weight, mask, eps)
