# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4 selected-token log-probability from the final hidden state (``dsv4-logprob-v1``).

    logits = hidden @ W.T        DeepGEMM bf16_gemm_nt, FP32 output (batch-invariant and
                                 vocab-shard invariant; scripts/probe_dsv4_head_gemm.py)
    (logp, lse) = tile stats + fixed-order merge (csrc/cuda/logprob_dsv4/dsv4_logprob.cu),
                  byte-equal to the golden rl_engine.kernels.ops.pytorch.logprob_dsv4

Tokens are processed in chunks of ``chunk_rows`` so the FP32 logits never exist for the
whole batch; every row is computed alone, so the chunking does not change any byte.

Autograd (:func:`selected_logprob`): ``dL/dhidden`` only (the LM head is frozen); the
LSE output carries no gradient. ``dL/dlogp`` is one scalar ``g`` per row, so::

    P    = (onehot(target) - p) [/ T]        BF16 [n, V]   (golden backward with g = 1)
    Q    = P @ W                             DeepGEMM bf16_gemm_nn, FP32 [n, H]
    dh   = BF16(g * Q)                       one FP32 multiply per element

When the hidden state requires grad, the forward computes ``Q`` while the chunk's logits
are still there and keeps only ``Q``; the backward is the multiply. Two GEMMs in all,
as for a naive implementation that keeps the logits. Without grad (rollout, recompute)
nothing extra runs. Deterministic and batch-invariant.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor
from torch.version import hip as torch_hip

from rl_engine.kernels.ops.cuda.deepgemm import require_deepgemm
from rl_engine.kernels.ops.pytorch.logprob_dsv4 import NUM_TILES, PROFILE

_SYMBOLS = ("dsv4_logprob_tile_stats", "dsv4_logprob_merge", "dsv4_logprob_backward")
DEFAULT_CHUNK_ROWS = 4096


def _extension():
    if torch_hip is not None or not torch.cuda.is_available():
        raise RuntimeError("DSv4 logprob requires NVIDIA CUDA")
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError as exc:
        raise RuntimeError("Build the CUDA extension first") from exc
    missing = [name for name in _SYMBOLS if not hasattr(ext, name)]
    if missing:
        raise RuntimeError(f"rl_engine._C lacks {missing}; rebuild the CUDA extension")
    return ext


def _check_temperature(temperature: float) -> float:
    t = float(temperature)
    if not t > 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return t


def head_logits(hidden: Tensor, weight: Tensor) -> Tensor:
    """FP32 ``[n, V]`` logits ``hidden @ weight.T`` (BF16 operands, DeepGEMM)."""
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("hidden and weight must be BF16")
    dg, _ = require_deepgemm(hidden.device)
    out = torch.empty(hidden.shape[0], weight.shape[0], dtype=torch.float32, device=hidden.device)
    if hidden.shape[0]:
        dg.bf16_gemm_nt(hidden.contiguous(), weight.contiguous(), out)
    return out


def tile_stats_cuda(z: Tensor, tile_len: int, temperature: float = 1.0) -> tuple[Tensor, Tensor]:
    """``(m, s)`` FP32 ``[n, tiles]`` for logits covering whole tiles of ``tile_len``."""
    m, s = _extension().dsv4_logprob_tile_stats(z, int(tile_len), _check_temperature(temperature))
    return m, s


def merge_cuda(m: Tensor, s: Tensor, zt: Tensor, temperature: float = 1.0):
    """``(logp, lse)`` from the 64 tile partials and the targets' raw logits ``zt``."""
    logp, lse = _extension().dsv4_logprob_merge(
        m.contiguous(), s.contiguous(), zt.float().contiguous(), _check_temperature(temperature)
    )
    return logp, lse


def logprob_from_logits_cuda(z: Tensor, targets: Tensor, temperature: float = 1.0):
    """``(logp, lse)`` FP32 ``[n]`` from FP32 logits ``[n, V]`` (``V % 64 == 0``)."""
    if z.shape[1] % NUM_TILES:
        raise ValueError(f"vocabulary {z.shape[1]} is not a multiple of {NUM_TILES}")
    m, s = tile_stats_cuda(z, z.shape[1] // NUM_TILES, temperature)
    zt = z.gather(1, targets.long().view(-1, 1)).view(-1)
    return merge_cuda(m, s, zt, temperature)


def logprob_backward_cuda(
    z: Tensor,
    targets: Tensor,
    lse: Tensor,
    grad: Tensor,
    temperature: float = 1.0,
    col_offset: int = 0,
) -> Tensor:
    """``dL/dz`` BF16 ``[n, cols]``; ``z`` holds columns ``[col_offset, col_offset + cols)``."""
    return _extension().dsv4_logprob_backward(
        z,
        targets.long().contiguous(),
        lse.float().contiguous(),
        grad.float().contiguous(),
        _check_temperature(temperature),
        int(col_offset),
    )


def _rows(n: int, chunk: int):
    for lo in range(0, n, chunk):
        yield lo, min(n, lo + chunk)


class SelectedLogprobFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, temperature, chunk_rows):
        n = hidden.shape[0]
        logp = torch.empty(n, dtype=torch.float32, device=hidden.device)
        lse = torch.empty(n, dtype=torch.float32, device=hidden.device)
        grad = ctx.needs_input_grad[0]
        if grad:
            dg, _ = require_deepgemm(hidden.device)
            q = torch.empty(n, hidden.shape[1], dtype=torch.float32, device=hidden.device)
            ones = torch.ones(min(n, chunk_rows), dtype=torch.float32, device=hidden.device)
        for lo, hi in _rows(n, chunk_rows):
            z = head_logits(hidden[lo:hi], weight)
            logp[lo:hi], lse[lo:hi] = logprob_from_logits_cuda(z, targets[lo:hi], temperature)
            if grad:
                p = logprob_backward_cuda(
                    z, targets[lo:hi], lse[lo:hi], ones[: hi - lo], temperature
                )
                del z
                dg.bf16_gemm_nn(p, weight.contiguous(), q[lo:hi])
                del p
            else:
                del z
        ctx.mark_non_differentiable(lse)
        if grad:
            ctx.save_for_backward(q)
        ctx.hidden_dtype = hidden.dtype
        return logp, lse

    @staticmethod
    def backward(ctx, grad_logp, grad_lse):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        (q,) = ctx.saved_tensors
        dh = (grad_logp.float().unsqueeze(1) * q).to(ctx.hidden_dtype)
        return dh, None, None, None, None


def selected_logprob(
    hidden: Tensor,
    weight: Tensor,
    targets: Tensor,
    temperature: float = 1.0,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[Tensor, Tensor]:
    """``(logp, lse)`` FP32 ``[n]`` of ``targets`` under ``softmax(hidden @ weight.T / T)``.

    ``hidden`` BF16 ``[n, H]``, ``weight`` BF16 ``[V, H]`` (frozen; ``V % 64 == 0``),
    ``targets`` int ``[n]`` in ``[0, V)``. Differentiable in ``hidden``.
    """
    if hidden.dim() != 2 or weight.dim() != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError(f"shapes {tuple(hidden.shape)} x {tuple(weight.shape)} do not match")
    if weight.shape[0] % NUM_TILES:
        raise ValueError(f"vocabulary {weight.shape[0]} is not a multiple of {NUM_TILES}")
    if targets.numel() != hidden.shape[0]:
        raise ValueError("one target per row is required")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    temperature = _check_temperature(temperature)
    return SelectedLogprobFunction.apply(
        hidden, weight, targets.long().contiguous(), temperature, int(chunk_rows)
    )


__all__ = [
    "DEFAULT_CHUNK_ROWS",
    "NUM_TILES",
    "PROFILE",
    "head_logits",
    "logprob_backward_cuda",
    "logprob_from_logits_cuda",
    "merge_cuda",
    "selected_logprob",
    "tile_stats_cuda",
]
