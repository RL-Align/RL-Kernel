# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Validation of ``sglang_runtime_adapter_h3`` on SGLang's own H3 modules (RFC #420).

``projection_target`` is a ``run_world`` target. On each rank it builds block 0's attention and
MLP from ``sglang.multimodal_gen`` at SGLang's TP size, loads the pinned Diffusers block-0 tensors
through SGLang's own loader (``_diffusers_h3_checkpoint`` and each parameter's TP
``weight_loader``), and evaluates the four projections three ways:

* ``native``: SGLang's own forward;
* ``bound``: SGLang's modules after ``install_h3_sglang_rollout``;
* ``canonical``: the training arithmetic of ``h3-parallel-v1`` on the same rank's shards
  (``par.column_parallel_linear`` / ``par.row_parallel_linear``, as in
  ``h3_tp.run_tp_block_rows``).

Column outputs are the rank's shard (``[q; k; v]`` and ``[gate; up]`` of its heads / channels),
row outputs are the full reduced rows. Only CUDA BF16 is meaningful here; ``sglang`` must be
importable in the worker.
"""

from __future__ import annotations

import os

import torch

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par

BLOCK = "transformer_blocks.0"
FFN = 14336
NAMES = ("qkv", "out", "fc1", "fc2")


def _sglang_block(weights: dict[str, torch.Tensor]):
    """SGLang's block-0 attention and MLP under a ``blocks.0`` root, loaded by SGLang's loader."""

    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as sm

    arch = MiniMaxH3DiTArchConfig()
    # The tensors are Diffusers-named: _diffusers_h3_checkpoint fuses q/k/v as [Q; K; V] and
    # swaps the fc1 halves, so SGLang's grouped-row reorder does not apply.
    arch.checkpoint_uses_diffusers_layout = True
    root = torch.nn.Module()
    root.blocks = torch.nn.ModuleList([torch.nn.Module()])
    block = root.blocks[0]
    block.attn = sm.MiniMaxH3Attention(arch, None, prefix="blocks.0.attn")
    block.mlp = sm.MiniMaxH3MLP(arch, None, prefix="blocks.0.mlp")
    root.cuda()
    params = dict(root.named_parameters())
    loaded = []
    for name, tensor in sm._diffusers_h3_checkpoint(iter(weights.items())):
        if name not in params:
            continue
        param = params[name]
        loader = getattr(param, "weight_loader", None)
        if loader is None:
            raise RuntimeError(f"{name}: SGLang parameter without a weight_loader")
        loader(param, tensor)
        loaded.append(name)
    return root, block, sorted(loaded)


def _canonical(weights, x, a, h, collective, tp, rank):
    w = lambda name: weights[f"{BLOCK}.{name}"]  # noqa: E731
    q, k, v = par.column_parallel_linear(
        x, [par.shard_rows(w(f"attn.to_{n}.weight"), tp, rank) for n in "qkv"], collective, tp
    )
    fused = w("ff.net.0.proj.weight")
    hid, gate = par.column_parallel_linear(
        x,
        [par.shard_rows(fused[:FFN], tp, rank), par.shard_rows(fused[FFN:], tp, rank)],
        collective,
        tp,
    )
    a_local = par.shard_cols(a, tp, rank).contiguous()
    h_local = par.shard_cols(h, tp, rank).contiguous()
    out = par.row_parallel_linear(
        a_local, par.shard_cols(w("attn.to_out.0.weight"), tp, rank), collective, tp
    )
    fc2 = par.row_parallel_linear(
        h_local, par.shard_cols(w("ff.net.2.weight"), tp, rank), collective, tp
    )
    return {
        "qkv": torch.cat([q, k, v], 1),
        "out": out,
        "fc1": torch.cat([gate, hid], 1),
        "fc2": fc2,
    }, (a_local, h_local)


def _evaluate(block, x, a_local, h_local):
    return {
        "qkv": block.attn.qkv_proj(x)[0],
        "out": block.attn.out_proj(a_local)[0],
        "fc1": block.mlp.fc1(x)[0],
        "fc2": block.mlp.fc2(h_local)[0],
    }


def projection_target(collective, *, x, a, h, prefix_rows: int = 0, **weights):
    """One rank: native, bound and canonical projections of block 0 (see the module docstring).

    ``x (T, 5376)`` feeds ``qkv_proj`` / ``fc1``; ``a (T, 7168)`` and ``h (T, 14336)`` are the full
    inputs of ``out_proj`` / ``fc2``, of which each rank takes its TP columns. With
    ``prefix_rows`` the bound and native projections are also run on the first ``prefix_rows``
    rows alone (row invariance).
    """

    import torch.distributed as dist
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    from rl_engine.integrations.engines.rollout.sglang import h3 as adapter

    rank, world = dist.get_rank(), dist.get_world_size()
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
    maybe_init_distributed_environment_and_model_parallel(tp_size=world, sp_size=1)
    weights = {name.replace("__", "."): tensor for name, tensor in weights.items()}
    root, block, loaded = _sglang_block(weights)
    tp = world
    coll = collective if tp > 1 else None
    canonical, (a_local, h_local) = _canonical(weights, x, a, h, coll, tp, rank)
    out = {"loaded": loaded, "tp": tp}
    with torch.no_grad():
        native = _evaluate(block, x, a_local, h_local)
        native_prefix = (
            _evaluate(block, x[:prefix_rows], a_local[:prefix_rows], h_local[:prefix_rows])
            if prefix_rows
            else None
        )
        with adapter.install_h3_sglang_rollout(root, collective=coll, tp=tp) as handle:
            bound = _evaluate(block, x, a_local, h_local)
            bound_prefix = (
                _evaluate(block, x[:prefix_rows], a_local[:prefix_rows], h_local[:prefix_rows])
                if prefix_rows
                else None
            )
            out["report"] = adapter.report(root, handle)
        restored = _evaluate(block, x, a_local, h_local)
    for name in NAMES:
        out[f"canonical_{name}"] = canonical[name]
        out[f"native_{name}"] = native[name]
        out[f"bound_{name}"] = bound[name]
        out[f"restored_equals_native_{name}"] = bool(torch.equal(restored[name], native[name]))
        if prefix_rows:
            assert native_prefix is not None and bound_prefix is not None
            out[f"native_prefix_{name}"] = native_prefix[name]
            out[f"bound_prefix_{name}"] = bound_prefix[name]
    return out


def block_weight_inputs(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Block-0 projection tensors as ``run_world`` inputs (keys without dots)."""

    names = [f"{BLOCK}.attn.to_{n}.weight" for n in "qkv"] + [
        f"{BLOCK}.attn.to_out.0.weight",
        f"{BLOCK}.ff.net.0.proj.weight",
        f"{BLOCK}.ff.net.2.weight",
    ]
    return {name.replace(".", "__"): weights[name] for name in names}
