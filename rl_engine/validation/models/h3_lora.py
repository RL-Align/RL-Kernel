# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Seeded H3 LoRA for the parallel harness (``h3-parallel-lora-v2``).

``make_block_lora`` draws one block's full LoRA from a seed: for each of the six targets of the
verl-omni H3 recipe, ``A (r, K)`` normal with mean 0 and standard deviation ``1 / r`` (peft's
``gaussian`` init: ``nn.init.normal_(lora_A.weight, std=1 / r)``) and ``B (N, r)`` zero (the
recipe's init) or normal with standard deviation ``b_std`` (to exercise the B path), stored BF16
like the recipe's adapters. ``shard_block_lora`` cuts this TP rank's tensors with
``parallel_lora.shard_lora``.
The draw depends only on the seed, the block index and the target, never on the grid.

The two token-refiner blocks carry the same six targets (the recipe's ``target_modules`` match
them too). The refiner runs replicated on every rank, so its LoRA is held whole: refiner block
``i`` draws as block ``REFINER_SEED_BLOCK + i``, ``attach_refiner_lora`` puts it on
``model.token_refiner``, and each target linear then runs ``h3-parallel-lora-v2`` at ``tp=1``
(base GEMM included, so the refiner's projections leave cuBLAS once a LoRA is attached). That
needs the refiner on the provider path (``strict_boundaries=False``): the strict refiner row
reads the weights directly and would not see the LoRA, so ``require_refiner_lora_supported``
refuses it. While a refiner block holds a LoRA its attention runs on PyTorch's math SDPA
backend: the default backends' backward does not repeat bit for bit at the recipe's 1024 text
tokens (cuDNN on B200), and the LoRA gradients pass through it. Actor and rollout install the
LoRA through the same patch, so both run the same refiner forward.
"""

from __future__ import annotations

from types import MethodType

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
# A token-refiner block's LoRA: {"scale": float, "tensors": {target: {"A": (r, K), "B": (N, r)}}}
REFINER_ATTR = "_rlk_refiner_lora"
REFINER_SEED_BLOCK = 1000  # refiner block i draws its LoRA as block 1000 + i
_REFINER_FORWARD = "_rlk_refiner_forward"  # a patched target linear's original forward
_REFINER_BLOCK_FORWARD = "_rlk_refiner_block_forward"  # a patched refiner block's original forward


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
    known, extra = list(SHAPES), 0
    for target, (k, n) in (shapes or SHAPES).items():
        # a recipe target draws with its index in the recipe's map, whatever subset or order of
        # targets ``shapes`` lists; other names follow, in their order
        if target in SHAPES:
            i = known.index(target)
        else:
            i, extra = len(known) + extra, extra + 1
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


def refiner_blocks(model) -> list:
    """The token-refiner blocks of ``model`` (two in the released checkpoint; none for a model
    without a token refiner)."""

    return list(getattr(getattr(model, "token_refiner", None), "refiner_blocks", ()))


def make_refiner_lora(index: int, **kwargs) -> dict[str, dict[str, torch.Tensor]]:
    """Refiner block ``index``'s full LoRA, BF16, held whole (``make_block_lora`` arguments)."""

    return make_block_lora(REFINER_SEED_BLOCK + index, **kwargs)


def _refiner_forward(block, target: str):
    """A target linear's forward: ``h3-parallel-lora-v2`` at tp=1 while ``block`` holds a LoRA.

    As in the blocks, the gate/up base (``ff.net.0.proj``) runs the gate/up row's fixed FP32-tile
    GEMM, and Q/K/V keep ``h3-projection-gemm-v1``; the row targets run the O / down rows' kernel.
    """

    kind = pl.TARGETS[target]
    kernel = "fixed" if target == "ff.net.0.proj" else "projection"

    def forward(linear, x):
        lora = block.__dict__.get(REFINER_ATTR)
        if lora is None:
            return linear.__dict__[_REFINER_FORWARD](x)
        parts, scale = lora["tensors"][target], lora["scale"]
        x2 = x.reshape(-1, x.shape[-1])
        if kind == "col":
            y = pl.column_parallel_lora_linear(
                x2, [linear.weight], [(parts["A"], [parts["B"]])], scale, kernel=kernel
            )[0]
        else:
            y = pl.row_parallel_lora_linear(x2, linear.weight, parts["A"], parts["B"], scale)
        return y.reshape(*x.shape[:-1], y.shape[-1])

    return forward


def _refiner_block_forward(block):
    """The block's forward, on the math SDPA backend while it holds a LoRA (see module doc)."""

    def forward(self, *args, **kwargs):
        original = self.__dict__[_REFINER_BLOCK_FORWARD]
        if REFINER_ATTR not in self.__dict__:
            return original(*args, **kwargs)
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.MATH]):
            return original(*args, **kwargs)

    return forward


def _patch_refiner(block) -> None:
    """Route the block's six target linears through ``_refiner_forward`` and its attention onto
    the math SDPA backend (once)."""

    if _REFINER_BLOCK_FORWARD not in block.__dict__:
        block.__dict__[_REFINER_BLOCK_FORWARD] = block.forward
        block.forward = MethodType(_refiner_block_forward(block), block)
    for target in SHAPES:
        linear = block.get_submodule(target)
        if linear.bias is not None:
            raise ValueError(f"refiner {target} has a bias; h3-parallel-lora-v2 has none")
        if _REFINER_FORWARD not in linear.__dict__:
            linear.__dict__[_REFINER_FORWARD] = linear.forward
            linear.forward = MethodType(_refiner_forward(block, target), linear)


def _check_refiner_parts(index: int, parts_by_target: dict) -> None:
    if set(parts_by_target) != set(SHAPES):
        raise ValueError(f"refiner {index}: targets {sorted(parts_by_target)} != {sorted(SHAPES)}")
    for target, parts in parts_by_target.items():
        k, n = SHAPES[target]
        r = parts["A"].shape[0] if "A" in parts else 0
        if set(parts) != {"A", "B"} or parts["A"].shape != (r, k) or parts["B"].shape != (n, r):
            raise ValueError(f"refiner {index} {target}: not a whole ({r}, {k}) / ({n}, {r}) LoRA")


def _set_refiner(model, loras: list[dict] | None, scale: float, device) -> None:
    """Put whole refiner LoRAs on ``model.token_refiner`` (``None`` removes them)."""

    blocks = refiner_blocks(model)
    if loras is not None and len(loras) != len(blocks):
        raise ValueError(f"{len(loras)} refiner LoRAs for {len(blocks)} refiner blocks")
    for i, block in enumerate(blocks):
        if loras is None:
            block.__dict__.pop(REFINER_ATTR, None)
            continue
        _check_refiner_parts(i, loras[i])
        _patch_refiner(block)
        tensors = {t: {p: v.to(device) for p, v in parts.items()} for t, parts in loras[i].items()}
        setattr(block, REFINER_ATTR, {"scale": scale, "tensors": tensors})


def attach_refiner_lora(
    model,
    *,
    rank: int = RANK,
    alpha: float = ALPHA,
    seed: int = 0,
    b_zero: bool = False,
    device: torch.device | str = "cuda",
) -> int:
    """Put a seeded whole LoRA on every token-refiner block. Returns the bytes held."""

    loras = [
        make_refiner_lora(i, rank=rank, seed=seed, b_zero=b_zero)
        for i in range(len(refiner_blocks(model)))
    ]
    _set_refiner(model, loras, alpha / rank, device)
    return sum(
        v.numel() * v.element_size() for lo in loras for p in lo.values() for v in p.values()
    )


def has_refiner_lora(model) -> bool:
    return any(REFINER_ATTR in b.__dict__ for b in refiner_blocks(model))


def require_refiner_lora_supported(model, *, strict_boundaries: bool) -> None:
    """The strict refiner row does not apply a LoRA: refuse that combination."""

    if has_refiner_lora(model):
        require_provider_refiner(strict_boundaries=strict_boundaries)


def require_provider_refiner(*, strict_boundaries: bool) -> None:
    """Refuse a refiner LoRA under strict boundaries (before one is installed, too)."""

    if strict_boundaries:
        raise ValueError(
            "refiner LoRA runs on the provider refiner (strict_boundaries=False); the strict "
            "refiner row would ignore it"
        )


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
    refiner: bool = False,
) -> int:
    """Put this TP rank's shards of a seeded LoRA on every transformer block of ``model``
    (replacing any), for ``h3_parallel_model.parallel_model_execution``; with ``refiner`` also a
    whole one on each token-refiner block, without it any refiner LoRA is removed. Returns the
    bytes held."""

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
    if refiner:
        held += attach_refiner_lora(
            model, rank=rank, alpha=alpha, seed=seed, b_zero=b_zero, device=device
        )
    else:  # as ``install_lora_shards``: the model holds exactly what was asked for
        _set_refiner(model, None, alpha / rank, device)
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
    for index, block in enumerate(refiner_blocks(model)):
        lora = block.__dict__.get(REFINER_ATTR)
        if lora is None:
            continue
        for target, parts in lora["tensors"].items():
            for part, t in parts.items():
                out[f"refiner.{index}.{target}.{part}"] = tensor_digest(t)
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
    refiner: list[dict[str, dict[str, torch.Tensor]]] | None = None,
) -> dict[str, str]:
    """Put LoRA shards received from the actor's TP rank ``tp_rank`` on every block.

    ``refiner`` (one whole ``{target: {"A", "B"}}`` per token-refiner block) is installed too;
    without it any refiner LoRA is removed, so the model holds exactly what was sent.

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
    for index, parts_by_target in enumerate(shards):
        for target, parts in parts_by_target.items():
            if not isinstance(parts, dict) or "A" not in parts:
                raise ValueError(f"block {index} {target}: the shards have no A")
            bad = sorted(k for k, v in parts.items() if not isinstance(v, torch.Tensor))
            if bad:
                raise ValueError(f"block {index} {target}: {bad} are not tensors")
    ranks = {p["A"].shape[0] for by_target in shards for p in by_target.values()}
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
    if refiner is not None:
        if len(refiner) != len(refiner_blocks(model)):
            raise ValueError(
                f"{len(refiner)} refiner LoRAs for {len(refiner_blocks(model))} refiner blocks"
            )
        for index, parts_by_target in enumerate(refiner):
            _check_refiner_parts(index, parts_by_target)
        refiner_ranks = {p["A"].shape[0] for by_target in refiner for p in by_target.values()}
        if refiner and refiner_ranks != ranks:  # [] fits a model without refiner blocks
            raise ValueError(
                f"refiner LoRA ranks {sorted(refiner_ranks)} != the blocks' rank {sorted(ranks)}"
            )
    # every transfer first (the refiner's too): a failed one (out of memory, say) changes no block
    staged = [
        {
            "tp": tp,
            "tp_rank": tp_rank,
            "scale": scale,
            "tensors": {
                t: {p: v.to(device) for p, v in parts.items()}
                for t, parts in parts_by_target.items()
            },
        }
        for parts_by_target in shards
    ]
    staged_refiner = (
        None
        if refiner is None
        else [
            {t: {p: v.to(device) for p, v in parts.items()} for t, parts in by_target.items()}
            for by_target in refiner
        ]
    )
    previous = [block.__dict__.get(pm.LORA_ATTR) for block in blocks]
    previous_refiner = [b.__dict__.get(REFINER_ATTR) for b in refiner_blocks(model)]

    def restore() -> None:
        for block, old in zip(blocks, previous):
            if old is None:
                block.__dict__.pop(pm.LORA_ATTR, None)
            else:
                setattr(block, pm.LORA_ATTR, old)
        for block, old in zip(refiner_blocks(model), previous_refiner):
            if old is None:
                block.__dict__.pop(REFINER_ATTR, None)
            else:
                setattr(block, REFINER_ATTR, old)

    try:
        for block, lora in zip(blocks, staged):
            setattr(block, pm.LORA_ATTR, lora)
        _set_refiner(model, staged_refiner, scale, device)
        got = lora_digests(model)
    except BaseException:
        restore()
        raise
    if expected is not None and got != expected:
        bad = sorted(k for k in set(got) | set(expected) if got.get(k) != expected.get(k))
        restore()
        raise RuntimeError(f"LoRA sync digest mismatch on {len(bad)} shards, e.g. {bad[:3]}")
    return got


LEAVES_PER_BLOCK = sum(len(leaf_names(t)) for t in SHAPES)  # 13
