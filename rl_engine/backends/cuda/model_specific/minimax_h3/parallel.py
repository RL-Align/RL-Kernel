# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3 tensor-, sequence- and context-parallel building blocks (contract ``h3-parallel-v1``).

Rows: ``tp_attention_7168``, ``tp_ffn_14336``, ``cp_full_attention`` (Ulysses),
``cp_metadata_coshard``, ``cp_arbitrary_length``, ``distributed_packed_io``.

Every reduction that crosses ranks is defined on **8 canonical units** so TP 1/2/4/8 add the
same numbers in the same order:

* *row-parallel* (O and FFN-down, ``K`` = 7168 / 14336): the reduction axis is cut into 8
  equal K-units; a unit is one fixed-order FP32 GEMM (``h3-projection-gemm-v1`` tile, ``K``
  swept ascending). A rank holding ``8/tp`` consecutive units adds them in balanced-tree
  order and the collective adds the ranks' partials in the same tree: the result is the
  balanced tree over the 8 units for every TP size, rounded to BF16 once.
* *column-parallel backward* (``dX`` of Q/K/V and gate/up, reduction over the output axis):
  the same 8 units, where unit ``j`` is output block ``j`` of every weight of the fused
  projection, added in the order given (q, k, v / hidden, gate).

The forward of a column-parallel projection and the backward of a row-parallel one reduce
over axes that do not depend on the sharding (they are plain ``h3-projection-gemm-v1``), so
they are bitwise the unsharded result without any collective.

Context parallelism (Ulysses) moves data only: sequence <-> head all-to-alls are exact, so
attention (per head, full sequence) is the unsharded attention. Weight gradients under
sequence sharding are computed from the gathered rows, so they equal the unsharded GEMM.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from rl_engine.backends.shared.triton.gemm import h3_projection as g
from rl_engine.distributed.algorithms.hierarchical import tree_add

CONTRACT = "h3-parallel-v1"
UNITS = 8  # canonical reduction units; TP sizes are the divisors 1, 2, 4, 8
TP_SIZES = (1, 2, 4, 8)
HEADS, HEAD_DIM = 56, 128
INNER = HEADS * HEAD_DIM  # 7168
FFN = 14336


# --- Sharding helpers ---------------------------------------------------------------------------


def check_tp(tp: int, rank: int) -> None:
    if tp not in TP_SIZES or not 0 <= rank < tp:
        raise ValueError(
            f"tp must be one of {TP_SIZES} and rank in [0, tp); got tp={tp} rank={rank}"
        )


def shard_rows(weight: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """Output-axis shard of a column-parallel weight ``(N, K)``: rows ``[r*N/tp, (r+1)*N/tp)``."""

    check_tp(tp, rank)
    n = weight.shape[0]
    if n % UNITS:
        raise ValueError(f"{n} output rows do not split into {UNITS} units")
    return weight.narrow(0, rank * n // tp, n // tp).contiguous()


def shard_cols(weight: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """Reduction-axis shard of a row-parallel weight ``(N, K)``: columns of ``8/tp`` K-units."""

    check_tp(tp, rank)
    k = weight.shape[1]
    unit = k // UNITS
    if unit * UNITS != k or unit % g.CONFIG_BF16["BLOCK_K"]:
        raise ValueError(f"K={k} must split into {UNITS} units of a multiple of the GEMM K tile")
    return weight.narrow(1, rank * k // tp, k // tp).contiguous()


def shard_heads(x: torch.Tensor, tp: int, rank: int, dim: int = -1) -> torch.Tensor:
    """Head shard of a ``(..., 7168)`` activation or weight axis (whole 128-wide heads)."""

    check_tp(tp, rank)
    return x.narrow(dim, rank * x.shape[dim] // tp, x.shape[dim] // tp).contiguous()


def shard_gate_up(weight: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """Shard of the fused ``(2*FFN, K)`` gate/up weight keeping ``[hidden; gate]`` halves.

    Each half is sharded on its own, so local rows are ``[hidden shard; gate shard]`` and
    ``chunk(2)`` still splits them.
    """

    half = weight.shape[0] // 2
    return torch.cat(
        [shard_rows(weight[:half], tp, rank), shard_rows(weight[half:], tp, rank)]
    ).contiguous()


# --- Row-parallel linear (reduction over the sharded axis) ---------------------------------------


def _units_partial(x: torch.Tensor, w: torch.Tensor, tp: int) -> torch.Tensor:
    """FP32 balanced-tree sum of this rank's ``8/tp`` K-units of ``x @ w.T``."""

    local = UNITS // tp
    unit = x.shape[1] // local
    parts = [
        g.gemm(
            x[:, i * unit : (i + 1) * unit],
            w[:, i * unit : (i + 1) * unit],
            b_t=True,
            out_dtype=torch.float32,
        )
        for i in range(local)
    ]
    return tree_add(parts)


class _RowParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, collective, tp, scatter):
        ctx.save_for_backward(x, w)
        ctx.scatter, ctx.collective = scatter, collective
        partial = _units_partial(x, w, tp)
        if tp > 1:
            partial = (
                collective.reduce_scatter(partial) if scatter else collective.all_reduce(partial)
            )
        return partial.to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        if ctx.scatter:  # sequence-parallel: the rows of dY are sharded; gather them (exact)
            grad = ctx.collective.all_gather(grad.contiguous())
        grad = grad.contiguous()
        dx = g.gemm(grad, w)  # reduction over N: independent of the shard
        dw = g.gemm(grad, x, a_t=True)  # reduction over tokens: independent of the shard
        return dx, dw, None, None, None


def row_parallel_linear(x, w_shard, collective=None, tp: int = 1, scatter: bool = False):
    """``y = x @ W.T`` with ``x (T, K/tp)`` and ``w_shard (N, K/tp)``; ``y`` bitwise TP-invariant.

    ``scatter`` gives this rank's ``T/tp`` rows of ``y`` (sequence parallel) instead of all of it.
    """

    if x.dim() != 2 or w_shard.dim() != 2 or x.shape[1] != w_shard.shape[1]:
        raise ValueError("row_parallel_linear needs x (T, K/tp) and w (N, K/tp)")
    if tp > 1 and collective is None:
        raise ValueError("tp > 1 needs a collective")
    return _RowParallel.apply(x.contiguous(), w_shard, collective, tp, scatter)


# --- Column-parallel linear (backward reduces over the sharded axis) ------------------------------


class _ColumnParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, collective, tp, gather, *weights):
        ctx.save_for_backward(x, *weights)
        ctx.collective, ctx.tp, ctx.gather = collective, tp, gather
        full_x = collective.all_gather(x) if gather else x
        ctx.full_x = full_x if gather else None
        return tuple(g.gemm(full_x, w, b_t=True) for w in weights)

    @staticmethod
    def backward(ctx, *grads):
        x, *weights = ctx.saved_tensors
        full_x = ctx.full_x if ctx.gather else x
        local = UNITS // ctx.tp
        grads = [t.contiguous() for t in grads]
        units = []
        for j in range(local):
            acc = None
            for grad, w in zip(grads, weights):
                width = grad.shape[1] // local
                piece = g.gemm(
                    grad[:, j * width : (j + 1) * width],
                    w[j * width : (j + 1) * width],
                    out_dtype=torch.float32,
                )
                acc = piece if acc is None else acc + piece
            units.append(acc)
        partial = tree_add(units)
        if ctx.tp > 1:
            partial = (
                ctx.collective.reduce_scatter(partial)
                if ctx.gather
                else ctx.collective.all_reduce(partial)
            )
        dx = partial.to(x.dtype)
        dws = tuple(g.gemm(grad, full_x, a_t=True) for grad in grads)
        return (dx, None, None, None, *dws)


def column_parallel_linear(
    x, weights: Sequence[torch.Tensor], collective=None, tp: int = 1, gather: bool = False
):
    """``[x @ w.T for w in weights]`` with each ``w`` a row shard ``(N/tp, K)``.

    The forward is the unsharded projection's columns (no collective). With ``gather``, ``x`` is
    this rank's rows of a sequence-parallel activation, gathered first (exact), and ``dX`` is
    reduce-scattered back to rows. ``dX`` is the canonical 8-unit tree for every ``tp``.
    """

    if tp > 1 and collective is None:
        raise ValueError("tp > 1 needs a collective")
    return _ColumnParallel.apply(x.contiguous(), collective, tp, gather, *weights)


# --- Context parallelism (Ulysses) ---------------------------------------------------------------


@dataclass(frozen=True)
class SeqShard:
    """This rank's contiguous slice of a packed sequence of ``length`` rows over ``cp`` ranks.

    Slices differ by at most one row (first ``length % cp`` ranks get the extra row), so any
    length is legal, including ``length < cp`` (some ranks hold nothing).
    """

    length: int
    cp: int
    rank: int

    @property
    def sizes(self) -> list[int]:
        base, extra = divmod(self.length, self.cp)
        return [base + (r < extra) for r in range(self.cp)]

    @property
    def lo(self) -> int:
        return sum(self.sizes[: self.rank])

    @property
    def hi(self) -> int:
        return self.lo + self.sizes[self.rank]


def coshard(shard: SeqShard, *tensors: torch.Tensor, dim: int = 0) -> list[torch.Tensor]:
    """Slice every per-row tensor (hidden, timestep_indices, token_tags, position_ids, ...) alike.

    The metadata that selects AdaLN rows and rotary angles must follow the rows it describes;
    slicing them in one call with one ``SeqShard`` is what keeps them aligned.
    """

    out = []
    for t in tensors:
        if t.shape[dim] != shard.length:
            raise ValueError(
                f"tensor has {t.shape[dim]} rows along dim {dim}, expected {shard.length}"
            )
        out.append(t.narrow(dim, shard.lo, shard.hi - shard.lo).contiguous())
    return out


def scatter_packed(packed: torch.Tensor, shard: SeqShard, dim: int = 0) -> torch.Tensor:
    """This rank's rows of a packed tensor that every rank holds in full."""

    return coshard(shard, packed, dim=dim)[0]


def gather_packed(local: torch.Tensor, shard: SeqShard, group=None, dim: int = 0) -> torch.Tensor:
    """Reassemble the packed tensor from every rank's rows (exact, any length)."""

    if shard.cp == 1:
        return local
    sizes = shard.sizes
    widest = max(sizes)
    moved = local.movedim(dim, 0).contiguous()
    pad = moved.new_zeros((widest - moved.shape[0], *moved.shape[1:]))
    chunks = [torch.empty_like(torch.cat([moved, pad])) for _ in range(shard.cp)]
    dist.all_gather(chunks, torch.cat([moved, pad]), group=group)
    return torch.cat([c[:n] for c, n in zip(chunks, sizes)]).movedim(0, dim)


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, in_sizes, out_sizes, group):
        ctx.sizes, ctx.group = (in_sizes, out_sizes), group
        out = x.new_empty((sum(out_sizes), *x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), out_sizes, in_sizes, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        in_sizes, out_sizes = ctx.sizes
        back = grad.new_empty((sum(in_sizes), *grad.shape[1:]))
        dist.all_to_all_single(back, grad.contiguous(), in_sizes, out_sizes, group=ctx.group)
        return back, None, None, None


def ulysses_attention(q, k, v, attention, shard: SeqShard, group=None):
    """Full attention of a sequence-sharded ``q, k, v (B, S_local, H, D)`` by head scatter.

    All-to-all makes each rank hold ``H/cp`` heads for the whole sequence, ``attention(q, k, v)``
    (``(B, H', S, D)`` layout, per-head independent) runs, and the result goes back. Both moves
    are exact, so the output rows equal the unsharded attention's rows bitwise as long as
    ``attention`` is batch/head invariant.
    """

    cp = shard.cp
    if cp == 1:
        return attention(q, k, v)
    batch, local_rows, heads, dim = q.shape
    if heads % cp:
        raise ValueError(f"{heads} heads do not divide over cp={cp}")
    sizes, hp = shard.sizes, heads // cp
    if local_rows != sizes[shard.rank]:
        raise ValueError(f"expected {sizes[shard.rank]} local rows, got {local_rows}")

    def to_heads(t):
        # peer r receives head block r of my rows; I receive every peer's rows of my head block
        send = t.reshape(batch, local_rows, cp, hp, dim).permute(2, 1, 0, 3, 4)
        send = send.reshape(cp * local_rows, batch, hp, dim)
        got = _AllToAll.apply(send, [local_rows] * cp, sizes, group)
        return got.permute(1, 0, 2, 3)  # (B, S, hp, D), sequence in rank order

    out = attention(to_heads(q), to_heads(k), to_heads(v))  # (B, S, hp, D)
    send = out.permute(1, 0, 2, 3).contiguous()
    got = _AllToAll.apply(send, sizes, [local_rows] * cp, group)
    got = got.reshape(cp, local_rows, batch, hp, dim)
    return got.permute(2, 1, 0, 3, 4).reshape(batch, local_rows, heads, dim)


# --- Weight gradients under sequence sharding ----------------------------------------------------


class _SeqLinear(torch.autograd.Function):
    """Row-local linear on a sequence shard whose ``dW`` is computed from gathered rows."""

    @staticmethod
    def forward(ctx, x, w, shard, group):
        ctx.save_for_backward(x, w)
        ctx.shard, ctx.group = shard, group
        return g.gemm(x, w, b_t=True)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        grad = grad.contiguous()
        dx = g.gemm(grad, w)
        full_x = gather_packed(x, ctx.shard, ctx.group)
        full_g = gather_packed(grad, ctx.shard, ctx.group)
        return dx, g.gemm(full_g, full_x, a_t=True), None, None


def seq_sharded_linear(x, w, shard: SeqShard, group=None):
    """``x @ W.T`` for this rank's rows; ``dW`` equals the unsharded GEMM (token sweep)."""

    return (
        _SeqLinear.apply(x.contiguous(), w, shard, group)
        if shard.cp > 1
        else g.linear_forward(x, w, None)
    )
