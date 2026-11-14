# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The full H3 model under TP / CP / SP (RFC #420 WS2 closeouts ``ws2_tp2_tp4_tp8`` and friends).

``parallel_model_execution`` binds every transformer block of the real model to the
``h3-parallel-v1`` block (``h3_tp.run_tp_block_rows``) on top of the full-model harness: the
boundary rows (audio / text projections, token refiner) keep their strict single-GPU bindings and
run replicated, each block takes the packed sequence from the previous block, slices the rows its
rank owns (CP rows, or their TP share with sequence parallelism), runs the parallel block and
all-gathers the rows back, so the next block and the heads see the same packed tensor on every rank.
Slicing and gathering are exact data movement with exact adjoints (a gather's backward is a slice,
a slice's backward is a gather of disjoint rows), so no rounding is introduced between blocks.

The canonical reference is the same model with ``Grid(1, 1, 0, 0)``: every other layout must equal
it bit for bit, block by block. :func:`first_drift` names the first block and logical row that
differs, with the owning rank and the row's modality, which is what the WS2 exit criterion asks of
first-drift tooling across rank boundaries.

This is a validation harness: every rank keeps the full weights of the model and slices the shards
of the projections out of them on each call. It does not shard the weight storage.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
from rl_engine.validation.models import h3_block as hb
from rl_engine.validation.models import h3_tp
from rl_engine.validation.models.h3_full_model import full_model_execution

CONTRACT = "h3-parallel-model-v1"
# Weight shards read back after a backward: name -> axis of the unsharded weight.
SHARDED_WEIGHTS = {
    "attn.to_q.weight": 0,
    "attn.to_k.weight": 0,
    "attn.to_v.weight": 0,
    "attn.to_out.0.weight": 1,
    "ff.hidden": 0,
    "ff.gate": 0,
    "ff.net.2.weight": 1,
}


def rank_spans(tp: int, cp: int, length: int, sp: bool) -> list[tuple[int, int]]:
    """Per global rank, the packed rows it holds between blocks.

    Rank ``r`` is ``(cp_rank, tp_rank) = divmod(r, tp)``: its CP rows, or with ``sp`` its TP share
    of them. TP ranks of one CP rank hold the same rows when ``sp`` is off.
    """

    spans = []
    for r in range(tp * cp):
        cp_rank, tp_rank = divmod(r, tp)
        shard = par.SeqShard(length, cp, cp_rank)
        lo, hi = shard.lo, shard.hi
        if sp:
            rows = hi - lo
            if rows % tp:
                raise ValueError(
                    f"sequence parallelism needs the {rows} rows of CP rank {cp_rank} "
                    f"divisible by tp={tp} (length {length}, cp={cp})"
                )
            share = rows // tp
            lo, hi = lo + tp_rank * share, lo + (tp_rank + 1) * share
        spans.append((lo, hi))
    return spans


def _assemble(rows: torch.Tensor, spans: list[tuple[int, int]], length: int) -> torch.Tensor:
    """All-gather every rank's rows and place them at their spans (copies only)."""

    if len(spans) == 1:  # the canonical single-GPU layout: nothing to exchange
        return rows
    sizes = [hi - lo for lo, hi in spans]
    widest = max(sizes)
    mine = torch.cat([rows, rows.new_zeros((1, widest - rows.shape[1], rows.shape[2]))], dim=1)
    parts = [torch.empty_like(mine) for _ in spans]
    dist.all_gather(parts, mine.contiguous())
    full = rows.new_empty((1, length, rows.shape[2]))
    for (lo, hi), part in zip(spans, parts):
        full[:, lo:hi] = part[:, : hi - lo]
    return full


class _RowsGather(torch.autograd.Function):
    """Rows held per rank -> the full packed tensor everywhere; backward keeps this rank's rows."""

    @staticmethod
    def forward(ctx, rows, spans, rank, length):
        ctx.span = spans[rank]
        return _assemble(rows, spans, length)

    @staticmethod
    def backward(ctx, grad):
        lo, hi = ctx.span
        return grad[:, lo:hi].contiguous(), None, None, None


class _RowsSlice(torch.autograd.Function):
    """The full packed tensor -> this rank's rows; backward gathers the rows' gradients."""

    @staticmethod
    def forward(ctx, full, spans, rank):
        ctx.spans, ctx.length = spans, full.shape[1]
        lo, hi = spans[rank]
        return full[:, lo:hi].contiguous()

    @staticmethod
    def backward(ctx, grad):
        return _assemble(grad.contiguous(), ctx.spans, ctx.length), None, None


class ParallelState:
    """What a parallel run leaves behind: per-block weight-shard leaves (for their gradients)."""

    def __init__(self, grid: h3_tp.Grid, sp: bool):
        self.grid, self.sp = grid, sp
        self.leaves: dict[int, dict[str, torch.Tensor]] = {}
        self.calls = 0
        # (block, global row, rank): that rank perturbs that row of that block's output, a
        # negative control for the first-drift report. Never set outside such a control.
        self.inject: tuple[int, int, int] | None = None
        # With ``digest_leaf`` set, a weight-shard gradient is reduced to its digest the moment it
        # is accumulated and then freed, and the leaves are not kept: a run then holds one
        # block's shard gradients at a time instead of a full copy of the model's.
        self.digest_leaf: Callable[[str, torch.Tensor], list[int]] | None = None
        self.digests: dict[str, list[int]] = {}


def _digest_on_grad(state: ParallelState, key: str, name: str, leaf: torch.Tensor) -> None:
    """Reduce ``leaf``'s gradient to ``state.digests[key]`` once it is accumulated, then free it."""

    digest_leaf = state.digest_leaf
    assert digest_leaf is not None

    def hook(param: torch.Tensor) -> None:
        grad = param.grad
        assert grad is not None
        state.digests[key] = digest_leaf(name, grad)
        param.grad = None

    leaf.register_post_accumulate_grad_hook(hook)


def _block_forward(index: int, grid: h3_tp.Grid, sp: bool, ops, state: ParallelState):
    def forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
        if attention_mask is not None:
            raise ValueError("the parallel H3 block requires an unmasked packed sequence")
        if hidden_states.shape[0] != 1:
            raise ValueError("the parallel block takes one packed sequence (batch 1)")
        length = hidden_states.shape[1]
        rank = dist.get_rank() if grid.tp * grid.cp > 1 else 0
        spans = rank_spans(grid.tp, grid.cp, length, sp)
        shard = grid.seq_shard(length)
        lo, hi = spans[rank]
        h_res = _RowsSlice.apply(hidden_states, spans, rank)
        cos, sin = par.coshard(shard, rotary_emb[0], rotary_emb[1], dim=0)
        idx = par.coshard(shard, adaln_indices, dim=0)[0]
        idx_res = idx[lo - shard.lo : hi - shard.lo].contiguous()
        weights = {hb.BLOCK + "." + n: p for n, p in self.named_parameters()}
        leaves: dict[str, torch.Tensor] | None = {} if torch.is_grad_enabled() else None
        out = h3_tp.run_tp_block_rows(
            weights, h_res, idx_res, cos, sin, temb, ops, grid, sp=sp, shard=shard, leaves=leaves
        )
        if state.inject is not None and state.inject[0] == index and state.inject[2] == rank:
            row = state.inject[1]
            if lo <= row < hi:
                scale = torch.ones_like(out)
                scale[:, row - lo] = 1.0078125  # one bf16 step on that row
                out = out * scale
        if leaves is not None and state.digest_leaf is not None:
            for name, leaf in leaves.items():  # a recomputation registers the live ones
                _digest_on_grad(state, f"{index}.{name}", name, leaf)
        elif leaves is not None:
            state.leaves[index] = leaves  # a recomputation (checkpointing) replaces the entry
        state.calls += 1
        return _RowsGather.apply(out, spans, rank, length)

    return forward


@contextmanager
def parallel_model_execution(model, grid: h3_tp.Grid, *, sp: bool, trace, ops=None):
    """Run the full model with every block on ``grid`` (``Grid(1, 1, 0, 0)`` is canonical)."""

    ops = ops or hb.CandidateOps()
    state = ParallelState(grid, sp)
    with full_model_execution(model, candidate=True, trace=trace, ops=ops, strict_boundaries=True):
        saved = [(block, block.forward) for block in model.transformer_blocks]
        try:
            for i, block in enumerate(model.transformer_blocks):
                block.forward = MethodType(_block_forward(i, grid, sp, ops, state), block)
            yield state
        finally:
            for block, forward in saved:
                block.forward = forward


# --- Evidence helpers ----------------------------------------------------------------------------


def exact_digest(t: torch.Tensor) -> list[int]:
    """Two wrap-around int64 checksums of a tensor's bytes (on the tensor's device).

    Equal tensors give equal digests; this is for comparing large gradients without copying them
    to the host, not a cryptographic hash.
    """

    flat = t.detach().contiguous().view(torch.uint8).reshape(-1)
    pad = (-flat.numel()) % 2
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    words = flat.view(torch.int16).to(torch.int64)
    weights = (torch.arange(words.numel(), device=words.device, dtype=torch.int64) % 2147483629) + 1
    return [int(words.sum()), int((words * weights).sum())]


def shard_of(grad: torch.Tensor, name: str, tp: int, tp_rank: int) -> torch.Tensor:
    """This TP rank's shard of an unsharded weight gradient, by the sharding of ``name``."""

    if SHARDED_WEIGHTS[name] == 0:
        return par.shard_rows(grad, tp, tp_rank)
    return par.shard_cols(grad, tp, tp_rank)


def first_drift(
    reference: dict[str, torch.Tensor],
    observed: dict[str, torch.Tensor],
    spans: list[tuple[int, int]],
    token_tags: torch.Tensor | None = None,
    order: list[str] | None = None,
) -> dict[str, Any] | None:
    """The first node whose tensors differ, with the first differing logical row and its owner.

    ``reference`` / ``observed`` map node names (``block.0`` ...) to ``(1, S, C)`` tensors (the
    trace snapshots). ``spans`` are the rows each rank holds; the owners are the ranks whose span
    contains the row (all TP ranks of a CP rank when sequence parallelism is off).
    """

    for name in order or list(reference):
        a, b = reference[name], observed[name]
        if a.shape != b.shape:
            return {"node": name, "reason": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"}
        if torch.equal(a, b):
            continue
        rows = (a != b).reshape(a.shape[0], a.shape[1], -1).any(dim=-1).any(dim=0)
        bad = rows.nonzero().flatten()
        row = int(bad[0])
        report: dict[str, Any] = {
            "node": name,
            "first_row": row,
            "rows_differing": int(bad.numel()),
            "owner_ranks": [r for r, (lo, hi) in enumerate(spans) if lo <= row < hi],
        }
        if token_tags is not None:
            report["modality"] = {0: "video", 1: "text", 2: "audio"}.get(
                int(token_tags[row]), "unknown"
            )
        return report
    return None


def grid_readback(grid: h3_tp.Grid, *, sp: bool, ops) -> dict[str, Any]:
    """Runtime identity of this rank's layout: topology, rank order, collectives, kernels."""

    coll = grid.tp_collective
    rank = dist.get_rank() if dist.is_initialized() else 0
    return {
        "contract": CONTRACT,
        "parallel_contract": par.CONTRACT,
        "rank": rank,
        "tp": grid.tp,
        "cp": grid.cp,
        "sp": sp,
        "tp_rank": grid.tp_rank,
        "cp_rank": grid.cp_rank,
        "rank_order": "rank = cp_rank * tp + tp_rank (TP groups are consecutive ranks)",
        "tp_group_ranks": (
            dist.get_process_group_ranks(coll.group) if coll is not None else [rank]
        ),
        "cp_group_ranks": (
            dist.get_process_group_ranks(grid.cp_group) if grid.cp_group is not None else [rank]
        ),
        "tp_collective": (
            None
            if coll is None
            else {
                "backend": getattr(coll, "backend_id", type(coll).__name__),
                "class": type(coll).__name__,
                "world_size": getattr(coll, "world_size", None),
                "local_size": getattr(coll, "local_size", None),
                "groups": getattr(coll, "num_groups", None),
            }
        ),
        "cp_mode": grid.cp_mode,
        "cp_contract": (
            None if grid.cp == 1 else "h3-ring-cp-v1" if grid.cp_mode == "ring" else "ulysses"
        ),
        "attention": grid.attention,
        "kernels": ops.backends(),
        "fallback": None,
    }
