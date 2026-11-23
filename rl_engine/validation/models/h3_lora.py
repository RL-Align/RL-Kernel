# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Seeded H3 LoRA for the parallel harness (``h3-parallel-lora-v1``).

``make_block_lora`` draws one block's full LoRA from a seed: for each of the six targets of the
verl-omni H3 recipe, ``A (r, K)`` normal with mean 0 and standard deviation ``1 / r`` (peft's
``gaussian`` init: ``nn.init.normal_(lora_A.weight, std=1 / r)``) and ``B (N, r)`` zero (the
recipe's init) or normal with standard deviation ``b_std`` (to exercise the B path), stored BF16
like the recipe's adapters. ``shard_block_lora`` cuts this TP rank's tensors with
``parallel_lora.shard_lora``.
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
        a = torch.randn(rank, k, generator=gen) / rank  # std 1 / r, as peft's gaussian init
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


def lora_digests(model) -> dict[str, str]:
    """``{"<block>.<target>.<part>": SHA-256 of dtype, shape and bytes}`` of every LoRA shard the
    model holds (``attach_lora`` / ``install_lora_shards``)."""

    from rl_engine.integrations.engines.rollout.vllm.h3_omni_model import tensor_digest
    from rl_engine.validation.models import h3_parallel_model as pm

    out = {}
    for index, block in enumerate(model.transformer_blocks):
        lora = getattr(block, pm.LORA_ATTR, None)
        if lora is None:
            continue
        for target, parts in lora["tensors"].items():
            for part, t in parts.items():
                out[f"{index}.{target}.{part}"] = tensor_digest(t)
    return out


def install_lora_shards(
    model,
    shards: list[dict[str, dict[str, torch.Tensor]]],
    tp: int,
    tp_rank: int,
    *,
    scale: float,
    expected: dict[str, str] | None = None,
    device: torch.device | str = "cuda",
) -> dict[str, str]:
    """Put LoRA shards received from the actor's TP rank ``tp_rank`` on every block.

    ``shards[i]`` is block ``i``'s ``{target: {part: tensor}}`` in ``shard_block_lora``'s layout.
    Actor and rollout both run TP ``tp``, so the actor's rank ``r`` sends the rollout's rank ``r``
    its own shards, without a gather. Every tensor is checked against ``shard_lora``'s shapes for
    ``(tp, tp_rank)``; ``expected`` (the sender's ``lora_digests``) must equal the installed
    digests exactly, else the previous LoRA is put back and the call raises. Returns the
    installed digests.
    """

    from rl_engine.validation.models import h3_parallel_model as pm

    blocks = list(model.transformer_blocks)
    if len(shards) != len(blocks):
        raise ValueError(f"{len(shards)} LoRA blocks for {len(blocks)} transformer blocks")
    ranks = {p["A"].shape[0] for by_target in shards for p in by_target.values() if "A" in p}
    if len(ranks) != 1 or min(ranks) <= 0:
        raise ValueError(f"LoRA ranks {sorted(ranks)}: one rank r > 0 for every target is needed")
    for index, parts_by_target in enumerate(shards):
        if set(parts_by_target) != set(SHAPES):
            raise ValueError(
                f"block {index}: targets {sorted(parts_by_target)} != {sorted(SHAPES)}"
            )
        for target, parts in parts_by_target.items():
            r = parts["A"].shape[0]
            k, n = SHAPES[target]
            want = pl.shard_lora(
                target,
                torch.empty(r, k, device="meta"),
                torch.empty(n, r, device="meta"),
                tp,
                tp_rank,
            )
            if set(parts) != set(want) or any(parts[p].shape != want[p].shape for p in want):
                raise ValueError(f"block {index} {target}: the shards are not ({tp}, {tp_rank})'s")
    previous = [block.__dict__.get(pm.LORA_ATTR) for block in blocks]
    for block, parts_by_target in zip(blocks, shards):
        setattr(
            block,
            pm.LORA_ATTR,
            {
                "tp": tp,
                "tp_rank": tp_rank,
                "scale": scale,
                "tensors": {
                    t: {p: v.to(device) for p, v in parts.items()}
                    for t, parts in parts_by_target.items()
                },
            },
        )
    got = lora_digests(model)
    if expected is not None and got != expected:
        bad = sorted(k for k in set(got) | set(expected) if got.get(k) != expected.get(k))
        for block, old in zip(blocks, previous):
            if old is None:
                block.__dict__.pop(pm.LORA_ATTR, None)
            else:
                setattr(block, pm.LORA_ATTR, old)
        raise RuntimeError(f"LoRA sync digest mismatch on {len(bad)} shards, e.g. {bad[:3]}")
    return got


LEAVES_PER_BLOCK = sum(len(leaf_names(t)) for t in SHAPES)  # 13
