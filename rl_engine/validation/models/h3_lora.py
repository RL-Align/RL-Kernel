# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Seeded H3 LoRA for the parallel harness (``h3-parallel-lora-v1``).

``make_block_lora`` draws one block's full LoRA from a seed: for each of the six targets of the
verl-omni H3 recipe, ``A (r, K) ~ N(0, 1/r)`` (peft's ``gaussian`` init) and ``B (N, r)`` zero
(the recipe's init) or ``N(0, b_std)`` (to exercise the B path), stored BF16 like the recipe's
adapters. ``shard_block_lora`` cuts this TP rank's tensors with ``parallel_lora.shard_lora``.
The draw depends only on the seed, the block index and the target, never on the grid.
"""

from __future__ import annotations

import torch

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel_lora as pl

# (K, N) of each target's base weight in an H3 block
SHAPES = {
    "attn.to_q": (5376, 7168),
    "attn.to_k": (5376, 7168),
    "attn.to_v": (5376, 7168),
    "attn.to_out.0": (7168, 5376),
    "ff.net.0.proj": (5376, 28672),
    "ff.net.2": (14336, 5376),
}
RANK, ALPHA = 64, 128  # the recipe's r and alpha: scale = alpha / r = 2.0


def make_block_lora(
    block: int,
    *,
    rank: int = RANK,
    seed: int = 0,
    b_zero: bool = False,
    b_std: float = 0.02,
    shapes: dict[str, tuple[int, int]] | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, dict[str, torch.Tensor]]:
    """``{target: {"A": (r, K), "B": (N, r)}}`` for block ``block``, BF16."""

    out = {}
    for i, (target, (k, n)) in enumerate((shapes or SHAPES).items()):
        gen = torch.Generator().manual_seed((seed * 1009 + block) * 31 + i)
        a = torch.randn(rank, k, generator=gen) / rank
        b = torch.zeros(n, rank) if b_zero else torch.randn(n, rank, generator=gen) * b_std
        out[target] = {"A": a.bfloat16().to(device), "B": b.bfloat16().to(device)}
    return out


def shard_block_lora(lora: dict, tp: int, tp_rank: int) -> dict[str, dict[str, torch.Tensor]]:
    """This TP rank's tensors of a full block LoRA (``A`` / ``B`` or ``B.hidden`` / ``B.gate``)."""

    return {t: pl.shard_lora(t, p["A"], p["B"], tp, tp_rank) for t, p in lora.items()}


def leaf_names(target: str) -> list[str]:
    """The gradient-leaf names of one target's shards, as ``run_tp_block_rows`` records them."""

    parts = ["A", "B.hidden", "B.gate"] if target == "ff.net.0.proj" else ["A", "B"]
    return [f"lora.{target}.{p}" for p in parts]


def attach_lora(
    model,
    tp: int,
    tp_rank: int,
    *,
    rank: int = RANK,
    alpha: float = ALPHA,
    seed: int = 0,
    b_zero: bool = False,
    device: torch.device | str = "cuda",
) -> int:
    """Put this TP rank's shards of a seeded LoRA on every transformer block of ``model``
    (replacing any), for ``h3_parallel_model.parallel_model_execution``. Returns the bytes held."""

    from rl_engine.validation.models import h3_parallel_model as pm

    held = 0
    for index, block in enumerate(model.transformer_blocks):
        full = make_block_lora(index, rank=rank, seed=seed, b_zero=b_zero)
        tensors = {
            t: {p: v.to(device) for p, v in parts.items()}
            for t, parts in shard_block_lora(full, tp, tp_rank).items()
        }
        held += sum(
            v.numel() * v.element_size() for parts in tensors.values() for v in parts.values()
        )
        setattr(
            block,
            pm.LORA_ATTR,
            {"tp": tp, "tp_rank": tp_rank, "scale": alpha / rank, "tensors": tensors},
        )
    return held


LEAVES_PER_BLOCK = sum(len(leaf_names(t)) for t in SHAPES)  # 13
