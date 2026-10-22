# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""One H3 block under tensor / sequence / context parallelism (RFC #420 WS2 closeouts).

``run_tp_block`` is the block graph with the attention and FFN projections replaced by the
``h3-parallel-v1`` operators: column-parallel Q/K/V and gate/up, per-head norm, rotary and
attention on the rank's heads, Ulysses head<->sequence all-to-all across the CP group,
row-parallel O and down with the deterministic collective, optional sequence parallelism
(rows of the residual stream sharded across the TP group). With ``tp=cp=1`` it is the
canonical single-GPU block every other layout must equal bitwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
from rl_engine.validation.models import h3_block as hb

HIDDEN, HEAD_DIM = hb.HIDDEN, hb.HEAD_DIM


@dataclass
class Grid:
    tp: int
    cp: int
    tp_rank: int
    cp_rank: int
    tp_collective: Any = None
    cp_group: Any = None

    def seq_shard(self, length: int) -> par.SeqShard:
        return par.SeqShard(length, self.cp, self.cp_rank)


def make_grid(tp: int, cp: int, *, local_size: int | None = None) -> Grid:
    """Rank ``r`` is ``(cp_rank, tp_rank) = divmod(r, tp)``; TP groups are consecutive ranks.

    ``local_size`` (ranks per node) selects the cross-node collective when the TP group spans
    nodes; ``None`` means the TP group is on one node (CUDA IPC fixed tree).
    """

    from rl_engine.distributed.algorithms.collectives import DeterministicCollective
    from rl_engine.distributed.algorithms.hierarchical import HierarchicalCollective

    world, rank = dist.get_world_size(), dist.get_rank()
    if world != tp * cp:
        raise ValueError(f"world {world} != tp {tp} x cp {cp}")
    cp_rank, tp_rank = divmod(rank, tp)
    tp_groups = [dist.new_group(list(range(c * tp, (c + 1) * tp))) for c in range(cp)]
    cp_groups = [dist.new_group(list(range(t, world, tp))) for t in range(tp)]
    collective = None
    if tp > 1:
        mine = tp_groups[cp_rank]
        if local_size is not None and local_size < tp:
            collective = HierarchicalCollective(
                local_size, mine, torch.cuda.current_device(), max_size_bytes=256 << 20
            )
        else:
            collective = DeterministicCollective(
                mine, torch.cuda.current_device(), max_size_bytes=256 << 20
            )
    return Grid(tp, cp, tp_rank, cp_rank, collective, cp_groups[tp_rank] if cp > 1 else None)


def _attention_fn(ops):
    def attend(q, k, v):  # (B, S, H', D) -> (B, S, H', D)
        q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        return ops.attention(q, k, v, causal=False).transpose(1, 2).contiguous()

    return attend


def run_tp_block(
    weights: dict[str, torch.Tensor],
    hidden: torch.Tensor,
    temb: torch.Tensor,
    index: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor],
    ops: hb.CandidateOps,
    grid: Grid,
    *,
    sp: bool = False,
    seq_len: int | None = None,
    leaves: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Block 0 for this rank. ``hidden/index/rope`` are the full packed sequence on every rank.

    The rank slices its CP rows (``cp_metadata_coshard``: hidden, AdaLN row index and both
    rotary tables with the same ``SeqShard``) and, with ``sp``, its TP row share of those.
    Returns this rank's rows of the block output (CP rows, or the SP share of them). With
    ``leaves`` the weight shards are tracked and stored there by name so the caller can read
    their gradients.
    """

    if hidden.shape[0] != 1:
        raise ValueError("the parallel block takes one packed sequence (batch 1)")
    tp, w = grid.tp, (lambda name: weights[f"{hb.BLOCK}.{name}"])
    seq = seq_len or hidden.shape[1]
    shard = grid.seq_shard(seq)
    h = par.coshard(shard, hidden, dim=1)[0]
    idx = par.coshard(shard, index, dim=0)[0]
    cos, sin = par.coshard(shard, rope[0], rope[1], dim=0)
    rows = h.shape[1]
    if sp:
        if rows % tp:
            raise ValueError(f"sequence parallelism needs {rows} local rows divisible by tp={tp}")
        share = rows // tp
        lo, hi = grid.tp_rank * share, (grid.tp_rank + 1) * share
        h_res, idx_res = h[:, lo:hi].contiguous(), idx[lo:hi].contiguous()
    else:
        h_res, idx_res = h, idx
    coll = grid.tp_collective

    def leaf(name, tensor):
        tensor = tensor.detach().clone()
        if leaves is not None:
            tensor.requires_grad_(True)
            leaves[name] = tensor
        return tensor

    table = ops.projection(temb, w("adaln_proj.linear.weight"), w("adaln_proj.linear.bias"))

    def proj_in(x, name_weights, gather):
        flat = x.reshape(-1, HIDDEN)
        return par.column_parallel_linear(flat, name_weights, coll, tp, gather=gather)

    x = ops.norm.forward_modulated(h_res, w("norm1.weight"), table[0], table[1], idx_res)
    q, k, v = proj_in(
        x,
        [
            leaf(f"attn.to_{n}.weight", par.shard_rows(w(f"attn.to_{n}.weight"), tp, grid.tp_rank))
            for n in "qkv"
        ],
        sp,
    )
    local_heads = par.HEADS // tp

    def heads(t):
        return t.view(1, rows, local_heads, HEAD_DIM)

    q = ops.norm(heads(q), w("attn.norm_q.weight"))
    k = ops.norm(heads(k), w("attn.norm_k.weight"))
    from rl_engine.validation.models import h3_provider as prov

    q, k = prov.provider_apply_rotary(q, cos, sin), prov.provider_apply_rotary(k, cos, sin)
    attn = par.ulysses_attention(q, k, heads(v), _attention_fn(ops), shard, grid.cp_group)
    o = par.row_parallel_linear(
        attn.reshape(rows, local_heads * HEAD_DIM),
        leaf("attn.to_out.0.weight", par.shard_cols(w("attn.to_out.0.weight"), tp, grid.tp_rank)),
        coll,
        tp,
        scatter=sp,
    )
    h1 = ops.gate(h_res, o.view(1, -1, HIDDEN), table[2], idx_res)
    x2 = ops.norm.forward_modulated(h1, w("norm2.weight"), table[3], table[4], idx_res)
    fused = w("ff.net.0.proj.weight")
    half = fused.shape[0] // 2
    hid, gate = proj_in(
        x2,
        [
            leaf("ff.hidden", par.shard_rows(fused[:half], tp, grid.tp_rank)),
            leaf("ff.gate", par.shard_rows(fused[half:], tp, grid.tp_rank)),
        ],
        sp,
    )
    act = ops.swiglu(gate.contiguous(), hid.contiguous())
    down = par.row_parallel_linear(
        act,
        leaf("ff.net.2.weight", par.shard_cols(w("ff.net.2.weight"), tp, grid.tp_rank)),
        coll,
        tp,
        scatter=sp,
    )
    return ops.gate(h1, down.view(1, -1, HIDDEN), table[5], idx_res)


def block_target(
    collective,
    dummy,
    *,
    tp: int,
    cp: int,
    sp: bool = False,
    seq_len: int = 0,
    backward: bool = False,
    local_size: int = -1,
    layout: str = "tiny",
    seed: int = 0,
):
    """``run_world`` target: this rank's rows of the parallel block (and gradients) for one case."""

    del collective  # the grid builds its own TP/CP groups
    from rl_engine.validation.models.h3_weights import load_h3_block_weights

    grid = make_grid(tp, cp, local_size=local_size if local_size > 0 else None)
    return block_case(
        load_h3_block_weights("cuda"),
        grid,
        sp=sp,
        seq_len=seq_len,
        backward=backward,
        layout=layout,
        seed=seed,
    )


def block_case(weights, grid: Grid, *, sp=False, seq_len=0, backward=False, layout="tiny", seed=0):
    """Run block 0 on ``grid``; the global rows this rank returns are in ``rows``."""

    ctx = hb.make_block_context(weights, layout=layout, seed=seed)
    n = seq_len or ctx["seq_len"]
    hidden = ctx["hidden"][:, :n].clone().requires_grad_(backward)
    index = ctx["index"][:n]
    rope = (ctx["rope"][0][:n], ctx["rope"][1][:n])
    ops = hb.CandidateOps()
    leaves: dict[str, torch.Tensor] | None = {} if backward else None
    with torch.set_grad_enabled(backward):
        out = run_tp_block(
            weights, hidden, ctx["temb"], index, rope, ops, grid, sp=sp, seq_len=n, leaves=leaves
        )
    shard = grid.seq_shard(n)
    lo, hi = shard.lo, shard.hi
    if sp:
        share = (hi - lo) // grid.tp
        lo, hi = lo + grid.tp_rank * share, lo + (grid.tp_rank + 1) * share
    result = {"out": out.detach(), "rows": torch.tensor([lo, hi])}
    if backward:
        gen = torch.Generator().manual_seed(seed + 99)
        full_grad = torch.randn(1, n, HIDDEN, generator=gen).to(out.dtype).cuda()
        out.backward(full_grad[:, lo:hi])
        result["dx"] = hidden.grad[:, lo:hi].detach()
        for name, tensor in leaves.items():
            result["dw." + name] = tensor.grad.detach()
    return result
