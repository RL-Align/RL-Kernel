# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""A serving engine's paged caches as :class:`Dsv41Model` caches, one forward at a time.

The rollout plugin (design doc ``dsv41-design.md`` 10.1) keeps vLLM's scheduler, cache
allocation and block tables and runs every computation as :class:`Dsv41Model`. vLLM's
caches differ from :meth:`Dsv41Model.new_caches` only in ways the kernels take as
arguments, except one:

- Block sizes (32 window rows, ``64 / ratio`` compressed entries) and block strides (one
  hybrid allocation, every layer's block a slice of a larger page): the kernels take both
  (``attention.block_sizes``, the tensors' ``stride(0)``).
- The ratio-2 compressor state. vLLM keeps a ring of ``capacity`` rows per request (row
  ``p % capacity``), enough for its kernels, which pool a chunk's complete groups straight
  from the chunk's scores. :class:`Dsv41Attention` saves every token's state first and then
  pools, so a prefill chunk needs a row per token. Each step gets a scratch state paged per
  request; the open group's rows (``p >= ratio * (start // ratio)``) come from the ring
  before the forward, and the rows of a group still open after it go back
  (:meth:`StepCaches.finish`). The ring's row layout is RL-Kernel's (``kv | score``):
  nothing of vLLM's reads it.

Nothing here imports vLLM; ``vllm_model.py`` fills :class:`EngineLayer` from vLLM's forward
context.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_engine.models.deepseek_v41.attention import LayerCaches

__all__ = ["EngineLayer", "StepCaches", "engram_windows"]

STATE_BLOCK = 16  # rows per block of the step's scratch state


@dataclass
class EngineLayer:
    """One layer's caches in the engine for this step; tables have a row per request.

    ``swa`` uint8 ``[blocks, swa_block, 584]``; a KV source also has ``kv``
    ``[blocks, kv_block, 584]`` and ``idx_k`` ``[blocks, idx_block, 132]`` (other layers of
    the group alias the source's), and at ratio 2 ``ring`` FP32
    ``[blocks, capacity, 2 * head_dim]`` with ``ring_blocks`` int64 ``[R]`` (each request's
    ring block). Any block stride."""

    swa: Tensor
    swa_table: Tensor
    kv: Tensor | None = None
    kv_table: Tensor | None = None
    idx_k: Tensor | None = None
    idx_k_table: Tensor | None = None
    ring: Tensor | None = None
    ring_blocks: Tensor | None = None


def _flat(cache: Tensor) -> Tensor:
    """``[blocks, rows, bytes]`` -> ``[blocks, rows * bytes]``, keeping the block stride."""
    return cache.view(cache.shape[0], -1)


class StepCaches:
    """:class:`LayerCaches` for every layer of one forward over the engine's caches.

    ``seq_lens`` / ``query_lens``: host lists, one per request in step order (the tables'
    row order). Call :meth:`finish` after the forward."""

    def __init__(
        self,
        config,
        layers: list[EngineLayer],
        seq_lens: list[int],
        query_lens: list[int],
    ):
        self.layers = layers
        self.caches: list[LayerCaches] = []
        self._ring_out: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = []
        reqs = len(seq_lens)
        for i, el in enumerate(layers):
            role = config.layer_role(i)
            c = LayerCaches(swa=_flat(el.swa), swa_table=el.swa_table[:reqs])
            c.extra["swa_block"] = el.swa.shape[1]
            if role.ratio:
                src = layers[role.kv_source]
                if src.kv is None or src.idx_k is None:
                    raise ValueError(f"layer {i}: KV source {role.kv_source} has no caches")
                c.kv, c.kv_table = _flat(src.kv), src.kv_table[:reqs]
                c.idx_k, c.idx_k_table = _flat(src.idx_k), src.idx_k_table[:reqs]
                c.extra["kv_block"] = src.kv.shape[1]
                c.extra["idx_block"] = src.idx_k.shape[1]
                if role.kv_source == i and role.ratio > 1:
                    self._scratch_state(c, el, role.ratio, seq_lens, query_lens)
            self.caches.append(c)

    def _scratch_state(self, c, el: EngineLayer, ratio, seq_lens, query_lens) -> None:
        if el.ring is None or el.ring_blocks is None:
            raise ValueError("a ratio-2 KV source needs its compressor ring")
        cap = el.ring.shape[1]
        if cap < ratio:
            raise ValueError(f"compressor ring of {cap} rows < ratio {ratio}")
        dev = el.ring.device
        width = max(-(-s // STATE_BLOCK) for s in seq_lens) if seq_lens else 1
        table = torch.zeros(len(seq_lens), width, dtype=torch.int32)
        n = 1  # block 0: scratch for the table's unused entries
        carry_req, carry_pos, out_req, out_pos = [], [], [], []
        for r, (s, q) in enumerate(zip(seq_lens, query_lens)):
            start = s - q
            first = start - start % ratio  # the open group's first position
            lo, hi = first // STATE_BLOCK, -(-s // STATE_BLOCK)
            if hi > lo:
                table[r, lo:hi] = torch.arange(n, n + hi - lo, dtype=torch.int32)
                n += hi - lo
            carry_req += [r] * (start - first)
            carry_pos += list(range(first, start))
            end = s - s % ratio
            out_req += [r] * (s - end)
            out_pos += list(range(end, s))
        state = torch.full(
            (n, STATE_BLOCK, el.ring.shape[2]), float("nan"), dtype=torch.float32, device=dev
        )
        table = table.to(dev)
        c.state, c.state_table = state, table
        flat = state.view(-1, state.shape[2])

        def slots(req: list[int], pos: list[int]):
            rq = torch.tensor(req, dtype=torch.long, device=dev)
            p = torch.tensor(pos, dtype=torch.long, device=dev)
            scratch = table[rq, p // STATE_BLOCK].long() * STATE_BLOCK + p % STATE_BLOCK
            return el.ring_blocks[rq].long(), p % cap, scratch

        if carry_req:
            rb, row, scratch = slots(carry_req, carry_pos)
            flat[scratch] = el.ring[rb, row]
        if out_req:
            self._ring_out.append((el.ring, flat, *slots(out_req, out_pos)))

    def finish(self) -> None:
        """Copy the rows of groups still open after the forward into the rings."""
        for ring, flat, rb, row, scratch in self._ring_out:
            ring[rb, row] = flat[scratch]
        self._ring_out.clear()


def engram_windows(
    input_ids: Tensor, query_start_loc: Tensor, lookback: Tensor, max_ngram: int
) -> Tensor:
    """int64 ``[T, max_ngram]`` n-gram windows (``engram.ngram_windows``) of a step's tokens.

    ``input_ids`` ``[T]`` grouped by request; ``query_start_loc`` int ``[R + 1]``;
    ``lookback`` int ``[>= R, depth >= max_ngram - 1]``: row ``r`` column ``k`` is the token
    ``k + 1`` places before request ``r``'s first token of the step, ``-1`` before the
    sequence (vLLM's ``lookback_token_ids``). Column ``k`` of a window is the token ``k``
    places back."""
    dev = input_ids.device
    tokens = input_ids.shape[0]
    qsl = query_start_loc.to(dev).long()
    reqs = qsl.shape[0] - 1
    req = torch.repeat_interleave(torch.arange(reqs, device=dev), qsl[1:] - qsl[:-1])
    j = torch.arange(tokens, device=dev) - qsl[req]  # index within the request's chunk
    k = torch.arange(max_ngram, device=dev)
    back = j[:, None] - k[None]  # >= 0: inside the chunk
    inside = input_ids.long()[(qsl[req][:, None] + back).clamp(min=0, max=max(tokens - 1, 0))]
    before = lookback.long()[req[:, None], (-back - 1).clamp(min=0, max=lookback.shape[1] - 1)]
    return torch.where(back >= 0, inside, before)
