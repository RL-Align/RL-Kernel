# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The full H3 model under TP / CP / SP (RFC #420 WS2 closeouts ``ws2_tp2_tp4_tp8`` and friends).

``parallel_model_execution`` binds every transformer block of the real model to the
``h3-parallel-v2`` block (``h3_tp.run_tp_block_rows``) on top of the full-model harness: the
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

By default every rank keeps the full weights of the model and slices the shards of the
projections out of them on each call. A model loaded by ``h3_sharded_load.load_sharded_model``
holds only this rank's shards of those projections (the full weights stay on the ``meta`` device),
and each block then uses the stored shards: the same bytes, so the same results.
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
# Attribute of a transformer block that holds this rank's stored projection shards (see
# ``h3_sharded_load``): ``{"tp": int, "tp_rank": int, "tensors": {name: shard}}``.
SHARDS_ATTR = "_rlk_tp_shards"
# A block's LoRA shards: {"tp", "tp_rank", "scale", "tensors": {target: {part: tensor}}}
LORA_ATTR = "_rlk_tp_lora"
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


def check_injection(spans: list[tuple[int, int]], row: int, rank: int) -> None:
    """Refuse a negative control whose perturbed row would not reach the assembled output.

    Without sequence parallelism the TP ranks of one CP rank hold the same rows, and
    ``_assemble`` keeps the copy of the last of them. A perturbation on another holder, or on a
    rank that does not hold the row, would leave the output unchanged and the control would
    report no drift. Every rank evaluates the same spans, so every rank raises alike.
    """

    holders = [r for r, (lo, hi) in enumerate(spans) if lo <= row < hi]
    if not holders:
        raise ValueError(f"row {row} is not in the packed sequence")
    if rank != holders[-1]:
        raise ValueError(
            f"rank {rank} cannot perturb row {row}: the assembled output keeps the copy of rank "
            f"{holders[-1]} (holders {holders})"
        )


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
        # With ``grad_sink`` set (training), each shard gradient is handed to
        # ``grad_sink(block, name, grad)`` once it is accumulated and then freed; it takes
        # precedence over ``digest_leaf``. The sink owns ``grad`` (it may keep it).
        self.grad_sink: Callable[[int, str, torch.Tensor], None] | None = None


def _digest_on_grad(state: ParallelState, key: str, name: str, leaf: torch.Tensor) -> None:
    """Reduce ``leaf``'s gradient to ``state.digests[key]`` once it is accumulated, then free it."""

    digest_leaf = state.digest_leaf
    assert digest_leaf is not None

    def hook(param: torch.Tensor) -> None:
        """Digest the accumulated gradient under ``key`` and free it."""
        grad = param.grad
        assert grad is not None
        state.digests[key] = digest_leaf(name, grad)
        param.grad = None

    leaf.register_post_accumulate_grad_hook(hook)


def _sink_on_grad(state: ParallelState, index: int, name: str, leaf: torch.Tensor) -> None:
    """Hand ``leaf``'s gradient to ``state.grad_sink`` once it is accumulated, then free it."""

    sink = state.grad_sink
    assert sink is not None

    def hook(param: torch.Tensor) -> None:
        """Give block ``index``'s gradient of shard ``name`` to the sink."""
        grad = param.grad
        assert grad is not None
        param.grad = None
        sink(index, name, grad)

    leaf.register_post_accumulate_grad_hook(hook)


def _block_forward(index: int, grid: h3_tp.Grid, sp: bool, ops, state: ParallelState):
    def forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
        """Block ``index`` on the grid for this rank's rows; the full rows are gathered back."""
        if attention_mask is not None:
            raise ValueError("the parallel H3 block requires an unmasked packed sequence")
        if hidden_states.shape[0] != 1:
            raise ValueError("the parallel block takes one packed sequence (batch 1)")
        length = hidden_states.shape[1]
        rank = dist.get_rank() if grid.tp * grid.cp > 1 else 0
        spans = rank_spans(grid.tp, grid.cp, length, sp)
        shard = grid.seq_shard(length)
        lo, hi = spans[rank]
        if state.inject is not None and state.inject[0] == index:
            check_injection(spans, state.inject[1], state.inject[2])  # before any collective
        h_res = _RowsSlice.apply(hidden_states, spans, rank)
        cos, sin = par.coshard(shard, rotary_emb[0], rotary_emb[1], dim=0)
        idx = par.coshard(shard, adaln_indices, dim=0)[0]
        idx_res = idx[lo - shard.lo : hi - shard.lo].contiguous()
        weights = {hb.BLOCK + "." + n: p for n, p in self.named_parameters()}
        stored = getattr(self, SHARDS_ATTR, None)
        if stored is not None and (stored["tp"], stored["tp_rank"]) != (grid.tp, grid.tp_rank):
            raise ValueError(
                f"block {index} holds the shards of tp={stored['tp']} rank {stored['tp_rank']}, "
                f"the grid is tp={grid.tp} rank {grid.tp_rank}"
            )
        lora = getattr(self, LORA_ATTR, None)
        if lora is not None and (lora["tp"], lora["tp_rank"]) != (grid.tp, grid.tp_rank):
            raise ValueError(
                f"block {index} holds the LoRA shards of tp={lora['tp']} rank {lora['tp_rank']}, "
                f"the grid is tp={grid.tp} rank {grid.tp_rank}"
            )
        leaves: dict[str, torch.Tensor] | None = {} if torch.is_grad_enabled() else None
        out = h3_tp.run_tp_block_rows(
            weights,
            h_res,
            idx_res,
            cos,
            sin,
            temb,
            ops,
            grid,
            sp=sp,
            shard=shard,
            leaves=leaves,
            shards=None if stored is None else stored["tensors"],
            lora=None if lora is None else lora["tensors"],
            lora_scale=2.0 if lora is None else lora["scale"],
        )
        if state.inject is not None and state.inject[0] == index and state.inject[2] == rank:
            row = state.inject[1]  # this rank holds it (check_injection)
            scale = torch.ones_like(out)
            scale[:, row - lo] = 1.0078125  # one bf16 step on that row
            out = out * scale
        if leaves is not None and state.grad_sink is not None:
            for name, leaf in leaves.items():  # a recomputation registers the live ones
                _sink_on_grad(state, index, name, leaf)
        elif leaves is not None and state.digest_leaf is not None:
            for name, leaf in leaves.items():  # a recomputation registers the live ones
                _digest_on_grad(state, f"{index}.{name}", name, leaf)
        elif leaves is not None:
            state.leaves[index] = leaves  # a recomputation (checkpointing) replaces the entry
        state.calls += 1
        return _RowsGather.apply(out, spans, rank, length)

    return forward


@contextmanager
def parallel_model_execution(
    model, grid: h3_tp.Grid, *, sp: bool, trace, ops=None, strict_boundaries: bool = True
):
    """Run the full model with every block on ``grid`` (``Grid(1, 1, 0, 0)`` is canonical).

    ``strict_boundaries=False`` leaves the boundary modules (patch and audio projections, heads,
    time embedder, final norm) on the provider. That is for hardware the strict boundary rows do
    not support (the video projections need SM90); the blocks run on ``grid`` either way.
    """

    ops = ops or hb.CandidateOps()
    state = ParallelState(grid, sp)
    with full_model_execution(
        model, candidate=True, trace=trace, ops=ops, strict_boundaries=strict_boundaries
    ):
        saved = [(block, block.forward) for block in model.transformer_blocks]
        try:
            for i, block in enumerate(model.transformer_blocks):
                block.forward = MethodType(_block_forward(i, grid, sp, ops, state), block)
            yield state
        finally:
            for block, forward in saved:
                block.forward = forward


# --- Evidence helpers ----------------------------------------------------------------------------


DIGEST_CHUNK = 1 << 24  # int16 words per step: bounds the int64 temporaries to ~384 MiB


def exact_digest(t: torch.Tensor, *, chunk: int = DIGEST_CHUNK) -> list[int]:
    """Two wrap-around int64 checksums of a tensor's bytes (on the tensor's device).

    The bytes are read as int16 words ``w_i``; the checksums are ``sum(w_i)`` and
    ``sum(w_i * (i mod 2147483629 + 1))``, both modulo 2**64. They are accumulated ``chunk``
    words at a time, which gives the same values as one pass (integer addition modulo 2**64 does
    not depend on the order) while the temporaries stay a bounded size, not a multiple of the
    tensor's.

    Equal tensors give equal digests; this is for comparing large gradients without copying them
    to the host, not a cryptographic hash. It is not injective: a change goes unseen only if its
    word differences cancel in both sums. A change of one word is always seen, and so is a change
    of two words in a tensor of fewer than 2147483629 words (two different weights); a change
    needs at least three words in a fixed relation to go unseen. The H3 grids compare the
    velocities, input gradients and block snapshots with ``torch.equal`` and only the weight-shard
    gradients by digest.
    """

    flat = t.detach().contiguous().view(torch.uint8).reshape(-1)
    pad = (-flat.numel()) % 2
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    words = flat.view(torch.int16)
    total = torch.zeros(2, dtype=torch.int64, device=words.device)
    for lo in range(0, words.numel(), chunk):
        part = words[lo : lo + chunk].to(torch.int64)
        positions = torch.arange(lo, lo + part.numel(), device=words.device, dtype=torch.int64)
        total[0] += part.sum()
        total[1] += (part * ((positions % 2147483629) + 1)).sum()
    return [int(total[0]), int(total[1])]


def shard_of(grad: torch.Tensor, name: str, tp: int, tp_rank: int) -> torch.Tensor:
    """This TP rank's shard of an unsharded weight gradient, by the sharding of ``name``.

    ``lora.<target>.<part>`` follows ``h3-parallel-lora-v2``: the ``A`` of a row-parallel target
    and the ``B`` of a column-parallel one are cut; the other is held whole.
    """

    if name.startswith("lora."):
        from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel_lora as pl

        target, part = name[len("lora.") :].rsplit(".", 1)
        if part in ("hidden", "gate"):  # ff.net.0.proj's B.hidden / B.gate
            target, part = target[: -len(".B")], "B"
        kind = pl.TARGETS[target]
        if kind == "row":
            return par.shard_cols(grad, tp, tp_rank) if part == "A" else grad
        return grad if part == "A" else par.shard_rows(grad, tp, tp_rank)
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

    The spans, the tags and every compared snapshot must describe the same packed length:
    otherwise the owners and the modality would be read from the wrong rows, so it raises. The
    two runs must hold the same nodes, and every node of ``order`` (by default all of them) must
    be among them: a node missing from either run is refused, never counted as no drift.
    """

    length = max(hi for _, hi in spans)
    if token_tags is not None and token_tags.numel() != length:
        raise ValueError(f"{token_tags.numel()} token tags for a packed length of {length}")
    if set(reference) != set(observed):
        only = sorted(set(reference) ^ set(observed))
        raise ValueError(f"the two runs hold different nodes: {only}")
    names = list(reference) if order is None else list(order)
    unknown = sorted(set(names) - set(reference))
    if unknown:
        raise ValueError(f"order names nodes neither run holds: {unknown}")
    for name in names:
        a, b = reference[name], observed[name]
        if a.shape != b.shape:
            return {"node": name, "reason": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"}
        if a.shape[1] != length:
            raise ValueError(f"{name} has {a.shape[1]} rows; the spans cover {length}")
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
    """Runtime identity of this rank's layout: topology, rank order, collectives, kernels.

    It has no fallback field, because nothing here measures one. What the grid itself runs
    cannot fall back: the candidate ops are checked to be CUDA (``require_candidate_cuda``; their
    classes are ``kernels``), and the attention path is fixed by ``grid.attention``, whose
    kernels raise if missing. The boundary rows are those of strict ``full_model_execution``,
    the same in the canonical run and on the grid; which op each resolves to is checked there.
    """

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
    }
