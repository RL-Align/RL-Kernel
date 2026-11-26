# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Deterministic indexer Top-K shared by the DSV4 training actor and the vLLM rollout.

Rule: per row, over the valid columns, pick K = min(topk, n_valid) columns with the
largest score; exact ties go to the LOWER column index. Indices are written relative
to the row's first valid column, ascending, padded with -1 up to ``topk``.
Implemented with a stable descending ``torch.sort`` (stable => lower index first on
ties), so the result does not depend on which aiter / vLLM ``_C`` kernel is picked.

Invalid columns are -inf. Valid -inf scores (candidate masking) are mapped to the
lowest finite value so they still rank above invalid columns. NaN in a valid column
fails an async device assert (no host sync).
"""

import torch

PROFILE = "rlk-deterministic-indexer-topk-stable-sort-lower-index-ascending-relative-v1"

# Rows per sort chunk are capped so values + int64 indices stay bounded.
_CHUNK_ELEMENTS = 1 << 24
_PAD = torch.iinfo(torch.int64).max


def _select(logits, starts, ends, indices, topk):
    """Core: ``starts``/``ends`` are int64 [rows]; valid columns are [start, end)."""
    rows, cols = logits.shape
    if indices.shape != (rows, topk):
        raise ValueError(f"indices must be [{rows}, {topk}], got {tuple(indices.shape)}")
    if not logits.dtype.is_floating_point:
        raise ValueError("logits must be floating point")
    if rows == 0:
        return
    k = min(topk, cols)
    column = torch.arange(cols, device=logits.device, dtype=torch.int64)
    rank = torch.arange(k, device=logits.device, dtype=torch.int64)
    lowest = torch.finfo(logits.dtype).min
    step = max(1, _CHUNK_ELEMENTS // max(cols, 1))
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        lg = logits[r0:r1]
        start = starts[r0:r1].clamp(min=0, max=cols)[:, None]
        end = ends[r0:r1].clamp(min=0, max=cols)[:, None]
        valid = (column[None, :] >= start) & (column[None, :] < end)
        # vLLM warm-up runs on uninitialized caches and may produce NaN/-inf logits; rank them
        # last (deterministically, lower index first). Real NaNs are caught downstream by the
        # finite-logprob verifier; a device assert here would abort the engine during warm-up.
        key = torch.where(valid, torch.nan_to_num(lg, nan=lowest, neginf=lowest), float("-inf"))
        _, order = torch.sort(key, dim=-1, descending=True, stable=True)
        order = order[:, :k]
        n_valid = (end - start).clamp(min=0)
        relative = torch.where(rank[None, :] < n_valid, order - start, _PAD)
        relative = torch.sort(relative, dim=-1).values
        relative = torch.where(relative == _PAD, -1, relative)
        indices[r0:r1, :k] = relative.to(indices.dtype)
        if k < topk:
            indices[r0:r1, k:] = -1


def deterministic_topk_prefill(logits, row_starts, row_ends, indices, topk):
    """Prefill: row i's valid columns are [row_starts[i], row_ends[i]) of ``logits``.

    Output is relative to row_starts[i] (same space as vLLM after
    ``_localize_aiter_prefill_topk`` / ``_C.top_k_per_row_prefill``).
    """
    _select(logits, row_starts.to(torch.int64), row_ends.to(torch.int64), indices, topk)


def deterministic_topk_decode(logits, next_n, seq_lens, indices, topk):
    """Decode: row r's valid columns are [0, len_r).

    ``seq_lens`` with one entry per row is used as-is (vLLM's aiter call passes
    next_n=1 with per-row lens). With one entry per request and next_n>1 (the
    ``_C.top_k_per_row_decode`` / paged-MQA-logits convention), row r = b*next_n + n
    has len = seq_lens[b] - next_n + n + 1. Lens are in compressed-KV index space,
    so the compress ratio does not enter.
    """
    rows = logits.shape[0]
    lens = seq_lens.reshape(-1).to(torch.int64)
    if lens.numel() != rows:
        if next_n < 1 or lens.numel() * next_n != rows:
            raise ValueError(f"seq_lens {lens.numel()} x next_n {next_n} != rows {rows}")
        n = torch.arange(rows, device=logits.device, dtype=torch.int64) % next_n
        lens = lens.repeat_interleave(next_n) - next_n + n + 1
    zeros = torch.zeros_like(lens)
    _select(logits, zeros, lens, indices, topk)
