# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton WS2 vocab-parallel logprob backend (``triton-vocab-parallel-logp-ws2``).

Same contract, TP transport, and fixed tile-order merge as the PyTorch
reference; the two per-shard passes are Triton kernels, so the backend runs on
both CUDA and ROCm from one source:

* ``_vocab_tile_stats_kernel``: one program per ``(row, tile)`` computes the
  FP32 ``(max, sumexp)`` partial over the real-vocabulary part of the tile.  The
  tile is walked in ``BLOCK_V`` chunks measured from the tile start, and masked
  lanes contribute the reduction identity, so the reduction order depends only
  on ``BLOCK_V`` (not on the tile's position in the shard, the padding, or the
  storage dtype) and TP=n stays bitwise equal to TP=1.
* ``_vocab_logp_backward_kernel``: fused elementwise
  ``coef_logp * (onehot - p) + coef_lse * p`` with ``p = exp(z - lse)`` on
  finite rows, ``0`` on non-finite rows and padding columns.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from rl_engine.contracts.operators.logprob import LogprobContract
from rl_engine.reference.logprob.vocab_parallel_logp import (
    DEFAULT_NUM_VOCAB_TILES,
    VocabParallelLogprobOp,
    apply_with_kernels,
)

BACKEND_ID = "triton-vocab-parallel-logp-ws2"
_BLOCK_V: int = 1024
_NUM_WARPS: int = 4
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _vocab_tile_stats_kernel(
    logits_ptr,  # [rows, local_vocab] shard, any of fp16/bf16/fp32
    max_ptr,  # [rows, local_tiles] fp32
    sum_ptr,  # [rows, local_tiles] fp32
    local_vocab,
    vocab_start,
    real_vocab,
    tile_size,
    local_tiles,
    stride_row,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    col_begin = tile * tile_size
    col_end = tl.minimum(col_begin + tile_size, local_vocab)
    # Columns at or beyond the real vocabulary are padding and contribute nothing.
    real_end = tl.minimum(col_end, tl.maximum(real_vocab - vocab_start, 0))
    row_base = row.to(tl.int64) * stride_row

    tile_max = tl.full((), float("-inf"), dtype=tl.float32)
    for start in range(col_begin, col_end, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        values = tl.load(
            logits_ptr + row_base + cols, mask=cols < real_end, other=float("-inf")
        ).to(tl.float32)
        tile_max = tl.maximum(tile_max, tl.max(values))

    empty = tile_max == float("-inf")
    max_safe = tl.where(empty, 0.0, tile_max)
    tile_sum = tl.zeros((), dtype=tl.float32)
    for start in range(col_begin, col_end, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        values = tl.load(
            logits_ptr + row_base + cols, mask=cols < real_end, other=float("-inf")
        ).to(tl.float32)
        tile_sum += tl.sum(tl.exp(values - max_safe))
    tile_sum = tl.where(empty, 0.0, tile_sum)

    out = row.to(tl.int64) * local_tiles + tile
    tl.store(max_ptr + out, tile_max)
    tl.store(sum_ptr + out, tile_sum)


@triton.jit
def _vocab_logp_backward_kernel(
    logits_ptr,  # [rows, local_vocab] shard
    lse_ptr,  # [rows] fp32 merged vocabulary LSE
    coef_logp_ptr,  # [rows] fp32, upstream grad of selected logp (0 on inactive rows)
    coef_lse_ptr,  # [rows] fp32, upstream grad of LSE (ignored unless HAS_LSE_GRAD)
    target_ptr,  # [rows] int64 local column of the owned target, or -1
    grad_ptr,  # [rows, local_vocab] output in the shard dtype
    local_vocab,
    vocab_start,
    real_vocab,
    stride_row,
    HAS_LSE_GRAD: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = cols < local_vocab
    real_end = tl.minimum(local_vocab, tl.maximum(real_vocab - vocab_start, 0))
    row_base = row.to(tl.int64) * stride_row

    lse = tl.load(lse_ptr + row)
    finite = (lse == lse) & (lse != float("inf")) & (lse != float("-inf"))
    lse_safe = tl.where(finite, lse, 0.0)
    g_logp = tl.load(coef_logp_ptr + row)
    hit = tl.load(target_ptr + row)

    values = tl.load(logits_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
    p = tl.where(finite, tl.exp(values - lse_safe), 0.0)
    onehot = tl.where(cols == hit, 1.0, 0.0)
    grad = g_logp * (onehot - p)
    if HAS_LSE_GRAD:
        g_lse = tl.load(coef_lse_ptr + row)
        grad = grad + g_lse * p
    grad = tl.where(cols < real_end, grad, 0.0)
    tl.store(grad_ptr + row_base + cols, grad.to(grad_ptr.dtype.element_ty), mask=mask)


def _check_shard(shard: torch.Tensor) -> torch.Tensor:
    if shard.device.type not in ("cuda", "hip", "xpu"):
        raise RuntimeError(f"{BACKEND_ID} requires a GPU tensor, got device {shard.device}")
    if shard.dim() != 2:
        raise ValueError("logits must be 2D [tokens, local_vocab]")
    if shard.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("logits must be float16, bfloat16, or float32")
    return shard.contiguous()


def triton_vocab_tile_stats(
    logits: torch.Tensor, vocab_start: int, real_vocab: int, num_tiles: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row, per-tile FP32 ``(max, sumexp)`` partials over the real vocabulary."""

    shard = _check_shard(logits)
    if vocab_start < 0 or real_vocab <= 0 or num_tiles <= 0:
        raise ValueError("invalid vocabulary metadata")
    rows, local_vocab = shard.shape
    if local_vocab <= 0 or local_vocab % num_tiles != 0:
        raise ValueError("local_vocab must be divisible by num_tiles")
    tile_max = torch.empty(rows, num_tiles, device=shard.device, dtype=torch.float32)
    tile_sum = torch.empty(rows, num_tiles, device=shard.device, dtype=torch.float32)
    if rows == 0:
        return tile_max, tile_sum
    _vocab_tile_stats_kernel[(rows, num_tiles)](
        shard,
        tile_max,
        tile_sum,
        local_vocab,
        int(vocab_start),
        int(real_vocab),
        local_vocab // num_tiles,
        num_tiles,
        shard.stride(0),
        BLOCK_V=_BLOCK_V,
        num_warps=_NUM_WARPS,
    )
    return tile_max, tile_sum


def triton_vocab_logp_backward(
    logits: torch.Tensor,
    lse: torch.Tensor,
    coef_logp: torch.Tensor,
    coef_lse: torch.Tensor,
    target_local: torch.Tensor,
    vocab_start: int,
    real_vocab: int,
    has_lse_grad: bool,
) -> torch.Tensor:
    """Fused ``grad_logits`` for the selected logprob and LSE outputs."""

    shard = _check_shard(logits)
    rows, local_vocab = shard.shape
    for name, tensor, dtype in (
        ("lse", lse, torch.float32),
        ("coef_logp", coef_logp, torch.float32),
        ("coef_lse", coef_lse, torch.float32),
        ("target_local", target_local, torch.long),
    ):
        if tensor.shape != (rows,) or tensor.dtype != dtype or tensor.device != shard.device:
            raise ValueError(f"{name} must be a [{rows}] {dtype} tensor on the logits device")
    grad = torch.empty_like(shard)
    if rows == 0 or local_vocab == 0:
        return grad
    grid = (rows, triton.cdiv(local_vocab, _BLOCK_V))
    _vocab_logp_backward_kernel[grid](
        shard,
        lse.contiguous(),
        coef_logp.contiguous(),
        coef_lse.contiguous(),
        target_local.contiguous(),
        grad,
        local_vocab,
        int(vocab_start),
        int(real_vocab),
        shard.stride(0),
        HAS_LSE_GRAD=bool(has_lse_grad),
        BLOCK_V=_BLOCK_V,
        num_warps=_NUM_WARPS,
    )
    return grad


class _TritonKernels:
    """``VocabParallelLogprobKernels`` over the Triton launchers above."""

    tile_stats = staticmethod(triton_vocab_tile_stats)
    backward = staticmethod(triton_vocab_logp_backward)


def _entropy_tile_stats(
    z_masked: torch.Tensor, tile: int, *, vocab_start: int, real_vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tile statistics for the shared entropy path (same kernel as ``apply``)."""

    return triton_vocab_tile_stats(
        z_masked, vocab_start, real_vocab_size, z_masked.shape[1] // tile
    )


class TritonVocabParallelLogprobOp(VocabParallelLogprobOp):
    """Contract-preserving Triton implementation of the WS2 vocab-parallel logprob."""

    op_class = "logprob"
    is_batch_invariant = True
    backend_id = BACKEND_ID
    use_native_tile_stats = staticmethod(_entropy_tile_stats)

    def apply(
        self,
        local_logits: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        contract: LogprobContract,
        tp_group: Any = None,
        num_vocab_tiles: int = DEFAULT_NUM_VOCAB_TILES,
        validate: bool = True,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not deterministic:
            return super().apply(
                local_logits,
                target_ids,
                contract=contract,
                tp_group=tp_group,
                num_vocab_tiles=num_vocab_tiles,
                validate=validate,
                deterministic=False,
            )
        return apply_with_kernels(
            local_logits,
            target_ids,
            contract=contract,
            tp_group=tp_group,
            num_vocab_tiles=num_vocab_tiles,
            validate=validate,
            kernels=_TritonKernels,
        )


__all__ = [
    "BACKEND_ID",
    "TritonVocabParallelLogprobOp",
    "triton_vocab_logp_backward",
    "triton_vocab_tile_stats",
]
