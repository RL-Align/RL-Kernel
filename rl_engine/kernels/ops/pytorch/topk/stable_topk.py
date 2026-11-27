# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Stable per-row top-k for the DSv4 sparse-attention indexer (golden).

Reproduces the selection of vLLM's batch-invariant DSv4 selector
(``torch.ops.ds4_bi.top_k_per_row_prefill``, ``ISEEKYAN/vllm@1a6542c``), which
the rollout uses for both prefill and decode under ``VLLM_BATCH_INVARIANT=1``:

* Row ``r`` considers the columns ``[row_starts[r], row_ends[r])`` of
  ``logits[r]``; a candidate's id is its request-local index
  ``column - row_starts[r]``.
* Candidates are ranked by the unique composite key
  ``(ordered_bits(score), local_index)`` in descending order: higher score
  first, and on equal score bits the LARGER local index first.
  ``ordered_bits`` maps the FP32 bit pattern onto an unsigned total order, so
  every input has a fixed place: ``-0.0`` ranks just below ``+0.0``, NaNs with
  the sign bit clear rank above ``+inf`` and NaNs with it set below ``-inf``.
* The output is ``int32 [rows, top_k]``: the selected local indices in key
  order, then ``-1`` once a row runs out of candidates.

The key never involves the row's position in a packed batch or any other row,
so the selection is batch-invariant by construction.
"""

from __future__ import annotations

import torch
from torch import Tensor

MAX_TOP_K = 2048
_LOW32 = 0xFFFFFFFF


def ordered_bits(x: Tensor) -> Tensor:
    """FP32 -> int64 in ``[0, 2**32)``, monotone in vLLM's total order."""
    if x.dtype != torch.float32:
        raise TypeError(f"expected FP32, got {x.dtype}")
    bits = x.contiguous().view(torch.int32).to(torch.int64) & _LOW32
    negative = bits >= 0x80000000
    return torch.where(negative, (~bits) & _LOW32, bits ^ 0x80000000)


def _check(logits: Tensor, row_starts: Tensor, row_ends: Tensor, top_k: int) -> None:
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise TypeError(f"logits must be 2-D FP32, got {tuple(logits.shape)} {logits.dtype}")
    rows = logits.shape[0]
    for name, t in (("row_starts", row_starts), ("row_ends", row_ends)):
        if t.dtype != torch.int32 or t.ndim != 1 or t.shape[0] != rows:
            raise TypeError(f"{name} must be int32 [{rows}], got {tuple(t.shape)} {t.dtype}")
        if t.device != logits.device:
            raise ValueError(f"{name} must be on {logits.device}")
    if not isinstance(top_k, int) or not 0 < top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be in (0, {MAX_TOP_K}], got {top_k!r}")
    if rows:
        starts, ends = row_starts.long(), row_ends.long()
        nonempty = ends > starts
        width = logits.shape[1]
        if bool(((starts < 0) & nonempty).any()) or bool(((ends > width) & nonempty).any()):
            raise ValueError(f"row ranges must lie within [0, {width}]")


def stable_topk_per_row(logits: Tensor, row_starts: Tensor, row_ends: Tensor, top_k: int) -> Tensor:
    """``logits`` FP32 ``[R, W]`` (any strides) -> int32 ``[R, top_k]`` local indices."""
    _check(logits, row_starts, row_ends, top_k)
    rows, width = logits.shape
    device = logits.device
    out = torch.full((rows, top_k), -1, dtype=torch.int32, device=device)
    if rows == 0 or width == 0:
        return out
    cols = torch.arange(width, device=device, dtype=torch.int64)
    starts = row_starts.long()[:, None]
    ends = row_ends.long()[:, None]
    inside = (cols >= starts) & (cols < ends)
    local = (cols - starts).clamp(min=0)
    # (ordered_bits - 2**31) * 2**32 + local is the vLLM key minus 2**63, so
    # signed int64 order equals its unsigned order (and nothing overflows).
    key = (ordered_bits(logits) - 0x80000000) * (1 << 32) + local
    key = torch.where(inside, key, torch.iinfo(torch.int64).min)
    taken = min(top_k, width)
    best = torch.sort(key, dim=1, descending=True).values[:, :taken]
    count = inside.sum(dim=1, keepdim=True)
    slots = torch.arange(taken, device=device)
    out[:, :taken] = torch.where(slots < count, best & _LOW32, -1).to(torch.int32)
    return out


class StableTopkOp:
    """PyTorch golden for ``stable_topk_per_row``."""

    def __call__(self, logits: Tensor, row_starts: Tensor, row_ends: Tensor, top_k: int) -> Tensor:
        return stable_topk_per_row(logits, row_starts, row_ends, top_k)
