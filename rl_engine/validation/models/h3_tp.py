# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""One H3 block under tensor / sequence / context parallelism (RFC #420 WS2 closeouts).

``run_tp_block`` is the block graph with the attention and FFN projections replaced by the
``h3-parallel-v2`` operators: column-parallel Q/K/V and gate/up, per-head norm, rotary and
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
from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel_lora as pl
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
    # "interim": the block's DeterministicAttentionOp (Ulysses only; its dK/dV is one GEMM, not a
    # chain a ring could continue). "h3_full": the H3 full-attention kernels (#553), which both
    # Ulysses and the ring (``cp_mode="ring"``, any CP size) reproduce bitwise.
    attention: str = "interim"
    cp_mode: str = "ulysses"

    def seq_shard(self, length: int) -> par.SeqShard:
        return par.SeqShard(length, self.cp, self.cp_rank)


def check_ulysses_heads(tp: int, cp: int, *, attention: str, cp_mode: str) -> None:
    """Refuse, before any group exists, a Ulysses split the H3 attention kernels cannot run.

    Ulysses gives each rank ``56 / tp / cp`` heads. The H3 kernels take the head counts of a TP
    rank (``ring_attention.LEGAL_HEADS``), so e.g. CP7 (8 heads) is refused here instead of
    inside the first block; the ring keeps all local heads and covers those CP sizes.
    """

    if cp == 1 or cp_mode != "ulysses":
        return
    local = par.HEADS // tp
    if attention == "h3_full":
        from rl_engine.backends.cuda.model_specific.minimax_h3.ring_attention import LEGAL_HEADS

        if local % cp or local // cp not in LEGAL_HEADS:
            raise ValueError(
                f"Ulysses splits {local} heads over cp={cp}, the H3 attention kernels take "
                f"{LEGAL_HEADS} heads: use cp_mode='ring'"
            )
    elif local % cp:
        raise ValueError(f"Ulysses cannot split {local} heads over cp={cp}")


def make_grid(
    tp: int,
    cp: int,
    *,
    local_size: int | None = None,
    attention: str = "interim",
    cp_mode: str = "ulysses",
) -> Grid:
    """Rank ``r`` is ``(cp_rank, tp_rank) = divmod(r, tp)``; TP groups are consecutive ranks.

    ``local_size`` (ranks per node) selects the cross-node collective when the TP group spans
    nodes; ``None`` means the TP group is on one node (CUDA IPC fixed tree).
    """

    from rl_engine.distributed.algorithms.collectives import DeterministicCollective
    from rl_engine.distributed.algorithms.hierarchical import HierarchicalCollective

    if attention not in ("interim", "h3_full") or cp_mode not in ("ulysses", "ring"):
        raise ValueError(f"unknown attention {attention!r} or cp_mode {cp_mode!r}")
    if cp_mode == "ring" and attention != "h3_full":
        raise ValueError("the ring continues the H3 kernels' dK/dV chain: use attention='h3_full'")
    check_ulysses_heads(tp, cp, attention=attention, cp_mode=cp_mode)
    world, rank = dist.get_world_size(), dist.get_rank()
    if world != tp * cp:
        raise ValueError(f"world {world} != tp {tp} x cp {cp}")
    cp_rank, tp_rank = divmod(rank, tp)
    tp_ranks = [list(range(c * tp, (c + 1) * tp)) for c in range(cp)]
    tp_groups = [dist.new_group(ranks) for ranks in tp_ranks]
    cp_groups = [dist.new_group(list(range(t, world, tp))) for t in range(tp)]
    collective: DeterministicCollective | HierarchicalCollective | None = None
    if tp > 1:
        mine = tp_groups[cp_rank]
        if local_size is not None and local_size < tp:
            collective = HierarchicalCollective(
                local_size,
                mine,
                torch.cuda.current_device(),
                max_size_bytes=256 << 20,
                group_ranks=tp_ranks,  # every rank creates all TP groups' subgroups (CP > 1)
            )
        else:
            collective = DeterministicCollective(
                mine, torch.cuda.current_device(), max_size_bytes=256 << 20
            )
    return Grid(
        tp,
        cp,
        tp_rank,
        cp_rank,
        collective,
        cp_groups[tp_rank] if cp > 1 else None,
        attention,
        cp_mode,
    )


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
    lora: dict[str, dict[str, torch.Tensor]] | None = None,
    lora_scale: float = 2.0,
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
    tp = grid.tp
    seq = seq_len or hidden.shape[1]
    shard = grid.seq_shard(seq)
    par.require_rows_on_every_rank(shard)  # every rank refuses alike, before any collective
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
    return run_tp_block_rows(
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
        lora=lora,
        lora_scale=lora_scale,
    )


def shard_leaf(
    name: str, tensor: torch.Tensor, leaves: dict[str, torch.Tensor] | None
) -> torch.Tensor:
    """A detached view (or contiguous copy) of ``tensor``; a gradient leaf in ``leaves`` if set.

    A contiguous shard (all of the weight for tp=1, a row slice otherwise) stays a view of it: a
    leaf is only read, and a copy per block would keep the weights twice over for as long as
    autograd holds the leaf (all 50 blocks of a checkpointed forward). A tensor that is not
    contiguous is copied.
    """

    tensor = tensor.detach().contiguous()
    if leaves is not None:
        tensor.requires_grad_(True)
        leaves[name] = tensor
    return tensor


def run_tp_block_rows(
    weights: dict[str, torch.Tensor],
    h_res: torch.Tensor,
    idx_res: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    temb: torch.Tensor,
    ops: hb.CandidateOps,
    grid: Grid,
    *,
    sp: bool,
    shard: par.SeqShard,
    leaves: dict[str, torch.Tensor] | None = None,
    shards: dict[str, torch.Tensor] | None = None,
    lora: dict[str, dict[str, torch.Tensor]] | None = None,
    lora_scale: float = 2.0,
) -> torch.Tensor:
    """The block body for rows this rank already holds (what a chained model hands each block).

    ``h_res`` / ``idx_res`` are this rank's residual rows (its CP rows, or its TP share of them
    with ``sp``), ``cos`` / ``sin`` the rotary tables of its CP rows. Returns the residual rows.

    ``shards`` holds this rank's shards of the sharded projections, keyed like
    ``h3_parallel_model.SHARDED_WEIGHTS`` (a weight-sharded model loads only those); without it
    each shard is cut from the full weight in ``weights``. The two give the same bytes.

    ``lora`` holds this rank's LoRA shards of the six targets (``h3_lora.shard_block_lora``);
    the projections then run ``h3-parallel-lora-v2`` with ``lora_scale``, and the shards are
    gradient leaves ``lora.<target>.<part>`` next to the weight shards.
    """

    par.require_rows_on_every_rank(shard)  # the chained model enters here, not through run_tp_block
    tp, w = grid.tp, (lambda name: weights[f"{hb.BLOCK}.{name}"])
    rows = shard.sizes[grid.cp_rank]
    coll = grid.tp_collective

    def leaf(name, tensor):
        """This block's weight shard ``name`` (see ``shard_leaf``)."""
        return shard_leaf(name, tensor, leaves)

    def sharded(name, cut):
        """Shard ``name``: the stored one when the model holds shards, else ``cut()``."""
        return leaf(name, shards[name] if shards is not None else cut())

    table = ops.projection(temb, w("adaln_proj.linear.weight"), w("adaln_proj.linear.bias"))

    cp_shard = shard if grid.cp > 1 else None  # dW sweeps every CP rank's rows
    cp_args = dict(cp_shard=cp_shard, cp_group=grid.cp_group)

    def lo(target, part):
        """This rank's LoRA tensor ``part`` of ``target`` as a gradient leaf."""
        assert lora is not None
        return leaf(f"lora.{target}.{part}", lora[target][part])

    def proj_in(x, name_weights, gather, groups=None, kernel="projection"):
        """Q/K/V on ``h3-projection-gemm-v1``; gate/up on its row's kernel (``fixed_fp32_gemm``)."""
        flat = x.reshape(-1, HIDDEN)
        if groups is None:
            return par.column_parallel_linear(
                flat, name_weights, coll, tp, gather=gather, kernel=kernel, **cp_args
            )
        return pl.column_parallel_lora_linear(
            flat,
            name_weights,
            groups,
            lora_scale,
            coll,
            tp,
            grid.tp_rank,
            gather,
            kernel=kernel,
            **cp_args,
        )

    def proj_out(x, w_shard, target):
        if lora is None:
            return par.row_parallel_linear(x, w_shard, coll, tp, scatter=sp, **cp_args)
        return pl.row_parallel_lora_linear(
            x,
            w_shard,
            lo(target, "A"),
            lo(target, "B"),
            lora_scale,
            coll,
            tp,
            grid.tp_rank,
            sp,
            **cp_args,
        )

    x = ops.norm.forward_modulated(h_res, w("norm1.weight"), table[0], table[1], idx_res)
    q, k, v = proj_in(
        x,
        [
            sharded(
                f"attn.to_{n}.weight",
                lambda n=n: par.shard_rows(w(f"attn.to_{n}.weight"), tp, grid.tp_rank),
            )
            for n in "qkv"
        ],
        sp,
        (
            None
            if lora is None
            else [(lo(f"attn.to_{n}", "A"), [lo(f"attn.to_{n}", "B")]) for n in "qkv"]
        ),
    )
    local_heads = par.HEADS // tp

    def heads(t):
        return t.view(1, rows, local_heads, HEAD_DIM)

    q = ops.qk_norm(heads(q), w("attn.norm_q.weight"))
    k = ops.qk_norm(heads(k), w("attn.norm_k.weight"))
    from rl_engine.validation.models import h3_provider as prov

    q, k = prov.provider_apply_rotary(q, cos, sin), prov.provider_apply_rotary(k, cos, sin)
    if grid.attention == "h3_full":
        from rl_engine.backends.cuda.model_specific.minimax_h3 import ring_attention as ring

        if grid.cp_mode == "ring":
            attn = ring.ring_attention(q, k, heads(v), shard, grid.cp_group)
        else:
            attn = par.ulysses_attention(q, k, heads(v), ring.local_attention, shard, grid.cp_group)
    else:
        attn = par.ulysses_attention(q, k, heads(v), _attention_fn(ops), shard, grid.cp_group)
    o = proj_out(
        attn.reshape(rows, local_heads * HEAD_DIM),
        sharded(
            "attn.to_out.0.weight",
            lambda: par.shard_cols(w("attn.to_out.0.weight"), tp, grid.tp_rank),
        ),
        "attn.to_out.0",
    )
    h1 = ops.gate(h_res, o.view(1, -1, HIDDEN), table[2], idx_res)
    x2 = ops.norm.forward_modulated(h1, w("norm2.weight"), table[3], table[4], idx_res)
    fused = w("ff.net.0.proj.weight")
    half = fused.shape[0] // 2
    hid, gate = proj_in(
        x2,
        [
            sharded("ff.hidden", lambda: par.shard_rows(fused[:half], tp, grid.tp_rank)),
            sharded("ff.gate", lambda: par.shard_rows(fused[half:], tp, grid.tp_rank)),
        ],
        sp,
        (
            None
            if lora is None
            else [
                (
                    lo("ff.net.0.proj", "A"),
                    [lo("ff.net.0.proj", "B.hidden"), lo("ff.net.0.proj", "B.gate")],
                )
            ]
        ),
        kernel="fixed",
    )
    act = ops.swiglu(gate.contiguous(), hid.contiguous())
    down = proj_out(
        act,
        sharded("ff.net.2.weight", lambda: par.shard_cols(w("ff.net.2.weight"), tp, grid.tp_rank)),
        "ff.net.2",
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
    attention: str = "interim",
    cp_mode: str = "ulysses",
    lora_rank: int = 0,
    lora_seed: int = 0,
    lora_b_zero: bool = False,
):
    """``run_world`` target: this rank's rows of the parallel block (and gradients) for one case."""

    del collective  # the grid builds its own TP/CP groups
    from rl_engine.validation.models.h3_weights import load_h3_block_weights

    grid = make_grid(
        tp,
        cp,
        local_size=local_size if local_size > 0 else None,
        attention=attention,
        cp_mode=cp_mode,
    )
    return block_case(
        load_h3_block_weights("cuda"),
        grid,
        sp=sp,
        seq_len=seq_len,
        backward=backward,
        layout=layout,
        seed=seed,
        lora_rank=lora_rank,
        lora_seed=lora_seed,
        lora_b_zero=lora_b_zero,
    )


def block_case(
    weights,
    grid: Grid,
    *,
    sp=False,
    seq_len=0,
    backward=False,
    layout="tiny",
    seed=0,
    lora_rank: int = 0,
    lora_seed: int = 0,
    lora_b_zero: bool = False,
):
    """Run block 0 on ``grid``; the global rows this rank returns are in ``rows``.

    ``lora_rank > 0`` puts a seeded LoRA on the six targets (``h3_lora.make_block_lora``,
    scale ``2``), cut for this rank; its gradients are returned as ``dw.lora.<target>.<part>``.
    """

    lora = None
    if lora_rank:
        from rl_engine.validation.models import h3_lora

        full = h3_lora.make_block_lora(
            0, rank=lora_rank, seed=lora_seed, b_zero=lora_b_zero, device="cuda"
        )
        lora = h3_lora.shard_block_lora(full, grid.tp, grid.tp_rank)

    ctx = hb.make_block_context(weights, layout=layout, seed=seed)
    n = seq_len or ctx["seq_len"]
    hidden = ctx["hidden"][:, :n].clone().requires_grad_(backward)
    index = ctx["index"][:n]
    rope = (ctx["rope"][0][:n], ctx["rope"][1][:n])
    ops = hb.CandidateOps()
    leaves: dict[str, torch.Tensor] | None = {} if backward else None
    with torch.set_grad_enabled(backward):
        out = run_tp_block(
            weights,
            hidden,
            ctx["temb"],
            index,
            rope,
            ops,
            grid,
            sp=sp,
            seq_len=n,
            leaves=leaves,
            lora=lora,
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
        assert leaves is not None
        for name, tensor in leaves.items():
            result["dw." + name] = tensor.grad.detach()
    return result
