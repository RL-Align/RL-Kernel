# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""LoRA on the H3 tensor-, sequence- and context-parallel projections (``h3-parallel-lora-v2``).

``y = round(x W^T + s * (x A^T) B^T)`` with a frozen base ``W`` and trainable ``A (r, K)``,
``B (N, r)``, FP32 accumulate and one rounding to the activation dtype. The base products run the
kernel of the projection's row, as in ``h3-parallel-v2`` (``h3-projection-gemm-v1`` for Q/K/V,
``fixed_fp32_gemm`` for gate/up, O and down); every LoRA product is a ``h3-projection-gemm-v1``
GEMM with FP32 operands (``ieee``). ``A`` and ``B`` may be stored in BF16;
they are read as FP32, and their gradients are rounded once to their dtype.

Every reduction that crosses ranks uses the **8 canonical units** of ``h3-parallel-v2``, so the
result is the same for TP 1/2/4/8, and ``tp=1`` (``Grid(1, 1)``) is the canonical run:

* *column-parallel* (Q/K/V; the hidden and gate halves of ``ff.net.0.proj``, which share one
  ``A``): ``A`` is held whole by every TP rank, ``B`` is cut by output rows like the base.
  ``z = x A^T`` and the forward need no collective. In the backward ``dz = s * sum_N dy B``
  reduces over the sharded output axis: unit ``j`` is output block ``j`` of every weight the
  ``A`` serves, the units are added in balanced-tree order and the ranks' partials in the same
  tree (the base ``dX`` units).
* *row-parallel* (O and FFN-down): ``A`` is cut by input columns (the base's K-units), ``B`` is
  held whole. ``z = sum_K x A^T`` is the base's unit tree, all-reduced (also under sequence
  parallelism: ``z`` is ``T x r``); ``dz = s * dy B``, ``dB`` and the ``A`` shard's gradient
  need no collective.
* *context parallelism*: ``dA`` and ``dB`` sum over tokens, from the rows of every CP rank
  gathered exactly (as the base ``dW``), so every CP rank computes the same gradient.

A Q/K/V column-parallel forward is bitwise ``strict_lora_linear`` (#574), whose base is
``h3-projection-gemm-v1``; everywhere a sum crosses units it is the canonical tree, which ``tp=1``
uses too.

v1 -> v2: the base GEMMs of gate/up, O and down moved to their rows' kernel.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from rl_engine.backends.cuda.model_specific.minimax_h3 import fixed_fp32_gemm as fx
from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
from rl_engine.backends.shared.triton.gemm import h3_projection as g
from rl_engine.distributed.algorithms.hierarchical import tree_add

CONTRACT = "h3-parallel-lora-v2"
F32 = torch.float32


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.float().contiguous()


def _rows_of(t: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """This rank's rows of a sequence-parallel tensor (the split ``reduce_scatter`` makes)."""

    n = t.shape[0] // tp
    return t.narrow(0, rank * n, n)


# --- Column-parallel --------------------------------------------------------------------------


class _ColumnParallelLoRA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, collective, tp, rank, gather, cp_shard, cp_group, scale, layout, kernel, *tensors
    ):
        """``tensors`` = the weight shards, then per group its ``A`` and its ``B`` shards."""
        n_w = layout[0]
        weights, rest = tensors[:n_w], tensors[n_w:]
        groups, i = [], 0
        for count in layout[1:]:
            groups.append((rest[i], rest[i + 1 : i + 1 + count]))
            i += 1 + count
        full_x = collective.all_gather(x) if gather else x
        x32 = _f32(full_x)
        zs = [g.gemm(x32, _f32(a), b_t=True) for a, _ in groups]  # (T, r), one per A
        outs, w_index = [], 0
        for z, (_, bs) in zip(zs, groups):
            for b in bs:
                base = kernel(full_x, weights[w_index], b_t=True, out_dtype=F32)
                low = g.gemm(z, _f32(b), b_t=True)
                outs.append((base + scale * low).to(x.dtype))
                w_index += 1
        ctx.save_for_backward(x, full_x if gather else x, *tensors, *zs)
        ctx.meta = (collective, tp, rank, gather, cp_shard, cp_group, scale, layout)
        ctx.kernel = kernel
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grads):
        collective, tp, rank, gather, cp_shard, cp_group, scale, layout = ctx.meta
        saved = ctx.saved_tensors
        x, full_x = saved[0], saved[1]
        n_w = layout[0]
        weights = saved[2 : 2 + n_w]
        rest = saved[2 + n_w : len(saved) - (len(layout) - 1)]
        zs = saved[len(saved) - (len(layout) - 1) :]
        groups, i = [], 0
        for count in layout[1:]:
            groups.append((rest[i], rest[i + 1 : i + 1 + count]))
            i += 1 + count
        grads = [t.contiguous() for t in grads]
        g32 = [t.float() for t in grads]
        local = par.UNITS // tp

        # base dX: the h3-parallel-v2 units (output block j of every weight, in order)
        units = []
        for j in range(local):
            acc = None
            for grad, w in zip(grads, weights):
                width = grad.shape[1] // local
                piece = ctx.kernel(
                    grad[:, j * width : (j + 1) * width],
                    w[j * width : (j + 1) * width],
                    out_dtype=F32,
                )
                acc = piece if acc is None else acc + piece
            units.append(acc)
        base_partial = tree_add(units)

        # dz per A: the same units over the weights that A serves, then one all-reduce for all
        dz_parts, w_index = [], 0
        for _, bs in groups:
            units = []
            for j in range(local):
                acc = None
                for k, b in enumerate(bs):
                    grad = g32[w_index + k]
                    width = grad.shape[1] // local
                    piece = g.gemm(
                        grad[:, j * width : (j + 1) * width],
                        _f32(b)[j * width : (j + 1) * width],
                    )
                    acc = piece if acc is None else acc + piece
                units.append(acc)
            dz_parts.append(tree_add(units))
            w_index += len(bs)
        dz_all = torch.cat(dz_parts, dim=1)
        if tp > 1:
            base_partial = (
                collective.reduce_scatter(base_partial)
                if gather
                else collective.all_reduce(base_partial)
            )
            dz_all = collective.all_reduce(dz_all)
        dz_all = dz_all * scale
        dzs = dz_all.split([a.shape[0] for a, _ in groups], dim=1)

        # dX = base + sum over A of dz A (in group order), one rounding
        low = None
        for dz, (a, _) in zip(dzs, groups):
            term = g.gemm(dz.contiguous(), _f32(a))
            low = term if low is None else low + term
        if gather:
            low = _rows_of(low, tp, rank)
        dx = (base_partial + low).to(x.dtype)

        rows_x = par._cp_rows(full_x, cp_shard, cp_group)
        rows_x32 = _f32(rows_x)
        dws = [
            ctx.kernel(par._cp_rows(grad, cp_shard, cp_group), rows_x, a_t=True) for grad in grads
        ]
        lora_grads, w_index = [], 0
        for dz, z, (a, bs) in zip(dzs, zs, groups):
            rows_dz = par._cp_rows(dz.contiguous(), cp_shard, cp_group)
            lora_grads.append(g.gemm(rows_dz, rows_x32, a_t=True).to(a.dtype))  # dA (r, K)
            rows_z = par._cp_rows(z, cp_shard, cp_group)
            for b in bs:
                rows_g = par._cp_rows(g32[w_index], cp_shard, cp_group)
                lora_grads.append((g.gemm(rows_g, rows_z, a_t=True) * scale).to(b.dtype))
                w_index += 1
        return (dx, None, None, None, None, None, None, None, None, None, *dws, *lora_grads)


def column_parallel_lora_linear(
    x: torch.Tensor,
    weights: Sequence[torch.Tensor],
    groups: Sequence[tuple[torch.Tensor, Sequence[torch.Tensor]]],
    scale: float,
    collective=None,
    tp: int = 1,
    rank: int = 0,
    gather: bool = False,
    cp_shard=None,
    cp_group=None,
    kernel: str = "projection",
):
    """``column_parallel_linear`` with LoRA: ``groups`` = ``[(A, [B shard per weight]), ...]``.

    ``kernel`` picks the base GEMM as in ``parallel.column_parallel_linear`` (``"fixed"`` for
    gate/up).

    The ``B`` shards of all groups, in order, line up with ``weights``; an ``A`` serves the
    weights of its group (Q, K and V one each; the hidden and gate halves of the FFN-in share
    one). Returns one output per weight.
    """

    if tp > 1 and collective is None:
        raise ValueError("tp > 1 needs a collective")
    n_b = sum(len(bs) for _, bs in groups)
    if n_b != len(weights):
        raise ValueError(f"{len(weights)} weights but {n_b} B shards")
    for w, b in zip(weights, (b for _, bs in groups for b in bs)):
        if b.shape[0] != w.shape[0]:
            raise ValueError(f"B shard rows {b.shape[0]} differ from the weight's {w.shape[0]}")
    par.check_column_widths(weights, tp)
    gather = gather and tp > 1  # one rank holds every row already
    layout = (len(weights),) + tuple(len(bs) for _, bs in groups)
    flat = [t for a, bs in groups for t in (a, *bs)]
    return _ColumnParallelLoRA.apply(
        x.contiguous(),
        collective,
        tp,
        rank,
        gather,
        cp_shard,
        cp_group,
        scale,
        layout,
        par.kernel_of(kernel),
        *weights,
        *flat,
    )


# --- Row-parallel -----------------------------------------------------------------------------


class _RowParallelLoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, a, b, collective, tp, rank, scatter, cp_shard, cp_group, scale):
        base = par._units_partial(x, w, tp)  # the rows' kernel (O / down)
        z = par._units_partial(_f32(x), _f32(a), tp, kernel=g.gemm)  # (T, r), FP32 operands
        if tp > 1:
            base = collective.reduce_scatter(base) if scatter else collective.all_reduce(base)
            z = collective.all_reduce(z)
        z_rows = _rows_of(z, tp, rank) if scatter else z
        y = (base + scale * g.gemm(z_rows.contiguous(), _f32(b), b_t=True)).to(x.dtype)
        ctx.save_for_backward(x, w, a, b, z)
        ctx.meta = (collective, tp, scatter, cp_shard, cp_group, scale)
        return y

    @staticmethod
    def backward(ctx, grad):
        x, w, a, b, z = ctx.saved_tensors
        collective, tp, scatter, cp_shard, cp_group, scale = ctx.meta
        if scatter:
            grad = collective.all_gather(grad.contiguous())
        grad = grad.contiguous()
        g32 = grad.float()
        dz = g.gemm(g32, _f32(b)) * scale  # (T, r): reduction over N, not sharded
        dx = (fx.gemm(grad, w, out_dtype=F32) + g.gemm(dz, _f32(a))).to(x.dtype)
        rows_g = par._cp_rows(grad, cp_shard, cp_group)
        rows_x = par._cp_rows(x, cp_shard, cp_group)
        dw = fx.gemm(rows_g, rows_x, a_t=True)
        rows_g32 = par._cp_rows(g32, cp_shard, cp_group)
        da = g.gemm(par._cp_rows(dz, cp_shard, cp_group), _f32(rows_x), a_t=True).to(a.dtype)
        db = (g.gemm(rows_g32, par._cp_rows(z, cp_shard, cp_group), a_t=True) * scale).to(b.dtype)
        return dx, dw, da, db, None, None, None, None, None, None, None


def row_parallel_lora_linear(
    x: torch.Tensor,
    w_shard: torch.Tensor,
    a_shard: torch.Tensor,
    b: torch.Tensor,
    scale: float,
    collective=None,
    tp: int = 1,
    rank: int = 0,
    scatter: bool = False,
    cp_shard=None,
    cp_group=None,
):
    """``row_parallel_linear`` with LoRA: ``a_shard (r, K/tp)`` cut like ``w_shard``; ``b (N, r)``
    held whole."""

    if x.dim() != 2 or x.shape[1] != w_shard.shape[1] or a_shard.shape[1] != w_shard.shape[1]:
        raise ValueError("row_parallel_lora_linear needs x, w_shard and a_shard with K/tp columns")
    if b.shape[0] != w_shard.shape[0] or b.shape[1] != a_shard.shape[0]:
        raise ValueError("B must be (N, r) for w_shard (N, K/tp) and a_shard (r, K/tp)")
    if tp > 1 and collective is None:
        raise ValueError("tp > 1 needs a collective")
    scatter = scatter and tp > 1  # one rank holds every row already
    return _RowParallelLoRA.apply(
        x.contiguous(),
        w_shard,
        a_shard,
        b,
        collective,
        tp,
        rank,
        scatter,
        cp_shard,
        cp_group,
        scale,
    )


# --- Sharding, counting and the rollout layout ------------------------------------------------

# The six LoRA targets of an H3 block and how each is cut ("col": B by rows, A whole; "row": A by
# columns, B whole). ``ff.net.0.proj`` is one LoRA over the fused [hidden, gate] weight.
TARGETS = {
    "attn.to_q": "col",
    "attn.to_k": "col",
    "attn.to_v": "col",
    "attn.to_out.0": "row",
    "ff.net.0.proj": "col",
    "ff.net.2": "row",
}


def shard_lora(target: str, a: torch.Tensor, b: torch.Tensor, tp: int, rank: int) -> dict:
    """This TP rank's LoRA tensors for ``target``: the cut of ``h3-parallel-lora-v2``.

    ``ff.net.0.proj`` returns its ``B`` as two shards, ``B.hidden`` and ``B.gate``, cut like the
    base's ``ff.hidden`` / ``ff.gate``.
    """

    kind = TARGETS[target]
    if kind == "row":
        return {"A": par.shard_cols(a, tp, rank), "B": b.contiguous()}
    if target == "ff.net.0.proj":
        half = b.shape[0] // 2
        return {
            "A": a.contiguous(),
            "B.hidden": par.shard_rows(b[:half], tp, rank),
            "B.gate": par.shard_rows(b[half:], tp, rank),
        }
    return {"A": a.contiguous(), "B": par.shard_rows(b, tp, rank)}


def lora_counted(target: str, part: str, tp_rank: int, cp_rank: int) -> bool:
    """Whether this rank counts LoRA tensor ``part`` ("A" / "B...") of ``target`` in a norm.

    TP-replicated tensors (A of a column-parallel target, B of a row-parallel one) count on TP
    rank 0, and every CP rank holds the same gradient, so only CP rank 0 counts.
    """

    if cp_rank != 0:
        return False
    replicated = (TARGETS[target] == "col") == (part == "A")
    return tp_rank == 0 if replicated else True


def to_rollout_layout(lora: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """The vLLM-Omni names and layout of one block's full (unsharded) LoRA, as upstream
    ``map_lora_tensors(strict=True)`` maps them: data movement only.

    ``qkv_proj`` keeps one A and one B per slice (q, k, v); ``fc1`` gets the A twice and the
    B rows reordered from ``[up, gate]`` to ``[gate, up]`` and split into ``fc1_0`` (gate) and
    ``fc1_1`` (up); ``out_proj`` and ``fc2`` are renames.
    """

    out = {}
    for name, slot in (("attn.to_q", "q"), ("attn.to_k", "k"), ("attn.to_v", "v")):
        out[f"attn.qkv_proj.{slot}.lora_A"] = lora[name]["A"]
        out[f"attn.qkv_proj.{slot}.lora_B"] = lora[name]["B"]
    out["attn.out_proj.lora_A"] = lora["attn.to_out.0"]["A"]
    out["attn.out_proj.lora_B"] = lora["attn.to_out.0"]["B"]
    a, b = lora["ff.net.0.proj"]["A"], lora["ff.net.0.proj"]["B"]
    half = b.shape[0] // 2  # Diffusers' fused weight is [hidden(up), gate]
    out["mlp.fc1_0.lora_A"], out["mlp.fc1_0.lora_B"] = a, b[half:]  # gate
    out["mlp.fc1_1.lora_A"], out["mlp.fc1_1.lora_B"] = a, b[:half]  # up
    out["mlp.fc2.lora_A"] = lora["ff.net.2"]["A"]
    out["mlp.fc2.lora_B"] = lora["ff.net.2"]["B"]
    return out
