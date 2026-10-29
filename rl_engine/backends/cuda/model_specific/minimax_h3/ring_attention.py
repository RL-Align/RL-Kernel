# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic ring context parallelism for H3 full attention (RFC #420 ``cp_full_attention``).

Ulysses (``parallel.ulysses_attention``) scatters heads, so it needs ``heads % cp == 0``; with
56 heads and TP the local head count is 28 / 14 / 7, which rules out e.g. CP3 or TP8 x CP2. The
ring needs no head split: every rank keeps its own query rows and the key/value blocks travel
around the ring (point-to-point, exact data movement).

Why the result equals the unsharded ``H3FullAttentionOp`` bit for bit (it is not an approximation
and does not merge softmax partials):

* forward: the kernel computes each query row from that row and all keys in ascending order, so
  a row's output does not depend on which other rows share its launch. The ring hands every rank
  the keys and values of all blocks in global sequence order, then runs the same kernel on the
  rank's rows.
* backward: ``dQ`` is per row, so it is local. ``dK`` and ``dV`` are FP32 accumulators that the
  kernel advances **one query row at a time in ascending order** (``h3_staged_dkv``, one chain
  per element, continued across query chunks), then scales and rounds once. The ring runs that
  same chain across ranks: rank ``r`` receives the accumulators after the rows of ranks
  ``0 .. r-1``, advances them with its own rows, and passes them on; the last rank scales,
  rounds to BF16 once and returns each rank its rows. The chain, hence every rounding, is the
  unsharded one for any shard sizes (ragged included), not only for chunk-aligned ones.

The backward pipeline is serial over the ranks by construction (that serial chain *is* the
contract), so it trades speed for exactness. Memory per rank is one full K/V (like an all-gather);
this implementation does not claim the memory saving of an online-softmax ring, which would
change the arithmetic.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from rl_engine.backends.cuda.attention import h3_full_attention as fa
from rl_engine.backends.cuda.model_specific.minimax_h3.parallel import SeqShard, group_peer

CONTRACT = "h3-ring-cp-v1"
_SCALE = fa._SCALE  # the unsharded op's single dK scaling
LEGAL_HEADS = (56, 28, 14, 7)  # 56 / tp for tp = 1, 2, 4, 8


def local_attention(q, k, v, chunk_rows: int = 128) -> torch.Tensor:
    """The unsharded H3 attention on ``(B, S, H, D)`` BF16 tensors for any head count.

    ``H3FullAttentionOp`` itself pins 56 heads; under TP the local count is 56 / tp. The kernels
    are per head, so the arithmetic is the same.
    """

    if q.dim() != 4 or q.shape[2] not in LEGAL_HEADS or q.shape[3] != 128:
        raise ValueError(f"H3 attention takes (B, S, H, 128) with H in {LEGAL_HEADS}")
    out, _ = fa._H3AttentionFn.apply(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        chunk_rows,
    )
    return out.transpose(1, 2).contiguous()


_peer = group_peer


def _exchange(sends, recvs, to_rank: int, from_rank: int, group) -> None:
    ops = [dist.P2POp(dist.isend, t, to_rank, group) for t in sends]
    ops += [dist.P2POp(dist.irecv, t, from_rank, group) for t in recvs]
    if ops:
        for work in dist.batch_isend_irecv(ops):
            work.wait()


def ring_all_gather(blocks, shard: SeqShard, group=None, dim: int = 1) -> list[torch.Tensor]:
    """Every rank's blocks for all ranks, in rank order, by ``cp - 1`` neighbour exchanges.

    ``blocks`` are this rank's tensors (``dim`` is the sequence axis, sizes ``shard.sizes[rank]``);
    the result has one concatenated, full-sequence tensor per input. Pure data movement.
    """

    cp, rank, sizes = shard.cp, shard.rank, shard.sizes
    mine = [b.contiguous() for b in blocks]
    held = [[None] * cp for _ in mine]
    for i, b in enumerate(mine):
        held[i][rank] = b
    nxt, prv = _peer(group, (rank + 1) % cp), _peer(group, (rank - 1) % cp)
    current = mine
    for step in range(cp - 1):
        sending, receiving = (rank - step) % cp, (rank - step - 1) % cp
        recv = [b.new_empty((*b.shape[:dim], sizes[receiving], *b.shape[dim + 1 :])) for b in mine]
        _exchange(
            current if sizes[sending] else [],
            recv if sizes[receiving] else [],
            nxt,
            prv,
            group,
        )
        for i, t in enumerate(recv):
            held[i][receiving] = t
        current = recv
    return [torch.cat(parts, dim=dim) for parts in held]


class _RingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, shard, group, chunk_rows):
        ext = fa._extension()
        k_full, v_full = ring_all_gather([k, v], shard, group)
        k_full = k_full.transpose(1, 2).contiguous()
        v_full = v_full.transpose(1, 2).contiguous()
        qh = q.transpose(1, 2).contiguous()
        out = torch.empty_like(qh)
        for start in range(0, qh.shape[2], chunk_rows):
            end = min(start + chunk_rows, qh.shape[2])
            block_out, _, p = ext.h3_attention_forward(
                qh[:, :, start:end].contiguous(), k_full, v_full
            )
            out[:, :, start:end] = block_out
            del block_out, p
        ctx.save_for_backward(qh, k_full, v_full)
        ctx.shard, ctx.group, ctx.chunk_rows = shard, group, chunk_rows
        return out.transpose(1, 2).contiguous()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        ext = fa._extension()
        qh, k_full, v_full = ctx.saved_tensors
        shard, group = ctx.shard, ctx.group
        cp, rank, sizes = shard.cp, shard.rank, shard.sizes
        gh = grad_out.transpose(1, 2).contiguous()
        batch, heads, _, dim = k_full.shape
        acc_shape = k_full.shape
        if rank == 0:
            dk = torch.zeros(acc_shape, device=qh.device, dtype=torch.float32)
            dv = torch.zeros(acc_shape, device=qh.device, dtype=torch.float32)
        else:  # the accumulators after the rows of ranks 0 .. rank-1
            dk = torch.empty(acc_shape, device=qh.device, dtype=torch.float32)
            dv = torch.empty(acc_shape, device=qh.device, dtype=torch.float32)
            dist.recv(dk, _peer(group, rank - 1), group=group)
            dist.recv(dv, _peer(group, rank - 1), group=group)
        dq = torch.empty_like(qh)
        for start in range(0, qh.shape[2], ctx.chunk_rows):
            end = min(start + ctx.chunk_rows, qh.shape[2])
            qt = qh[:, :, start:end].contiguous()
            p = ext.h3_attention_probabilities(qt, k_full, v_full)
            dqt, _, _ = ext.h3_attention_backward_accumulate(
                gh[:, :, start:end].contiguous(), qt, k_full, v_full, p, dk, dv
            )
            dq[:, :, start:end] = dqt
            del qt, p, dqt
        lo = shard.lo
        if rank < cp - 1:
            dist.send(dk, _peer(group, rank + 1), group=group)
            dist.send(dv, _peer(group, rank + 1), group=group)
            own = [
                torch.empty((batch, heads, sizes[rank], dim), device=qh.device, dtype=qh.dtype)
                for _ in range(2)
            ]
            if sizes[rank]:
                dist.recv(own[0], _peer(group, cp - 1), group=group)
                dist.recv(own[1], _peer(group, cp - 1), group=group)
            dk_own, dv_own = own
        else:  # the chain is complete: scale and round once, then hand every rank its rows
            dk.mul_(_SCALE)
            dk_b, dv_b = dk.to(qh.dtype), dv.to(qh.dtype)
            for peer in range(cp - 1):
                if sizes[peer]:
                    start = sum(sizes[:peer])
                    for t in (dk_b, dv_b):
                        dist.send(
                            t[:, :, start : start + sizes[peer]].contiguous(),
                            _peer(group, peer),
                            group=group,
                        )
            dk_own = dk_b[:, :, lo : lo + sizes[rank]].contiguous()
            dv_own = dv_b[:, :, lo : lo + sizes[rank]].contiguous()
        return (
            dq.transpose(1, 2).contiguous(),
            dk_own.transpose(1, 2).contiguous(),
            dv_own.transpose(1, 2).contiguous(),
            None,
            None,
            None,
        )


def ring_attention(q, k, v, shard: SeqShard, group=None, chunk_rows: int = 128) -> torch.Tensor:
    """Attention of a sequence-sharded ``q, k, v (B, S_local, H, D)``; bitwise the unsharded op.

    Works for any head count and any ``cp`` (no ``heads % cp`` requirement) and any length,
    including ranks that hold no rows.
    """

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if q.shape != k.shape or q.shape != v.shape or q.dim() != 4:
        raise ValueError("ring_attention needs equal (B, S_local, H, D) q, k, v")
    if q.dtype != torch.bfloat16:
        raise ValueError("the H3 attention kernels are BF16")
    if q.shape[2] not in LEGAL_HEADS or q.shape[3] != 128:
        raise ValueError(f"H3 attention takes (B, S, H, 128) with H in {LEGAL_HEADS}")
    if q.shape[1] != shard.sizes[shard.rank]:
        raise ValueError(f"expected {shard.sizes[shard.rank]} local rows, got {q.shape[1]}")
    if shard.cp == 1:
        return local_attention(q, k, v, chunk_rows)
    return _RingAttention.apply(
        q.contiguous(), k.contiguous(), v.contiguous(), shard, group, chunk_rows
    )
