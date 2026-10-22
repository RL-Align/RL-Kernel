# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``h3_checkpoint_mapper``: Diffusers <-> vLLM-Omni native H3 names and layouts.

The training side (diffusers, FSDP2) and the rollout side (vLLM-Omni) hold the same 638
tensors under different names and fused layouts. This module is the single, tested mapping, so
the strict operators on both sides read bit-identical weights:

* ``transformer_blocks.N`` -> ``blocks.N``;
  ``token_refiner.refiner_blocks.N`` -> ``token_refiner.blocks.N``;
* ``attn.to_q/to_k/to_v`` -> one fused ``attn.qkv_proj`` laid out ``[Q; K; V]`` (rows), the
  layout ``h3_qkv_gemm.forward_fused`` is bitwise-equal to;
* ``attn.to_out.0`` -> ``attn.out_proj``, ``attn.norm_q/k`` -> ``attn.q_norm/k_norm``;
* ``ff.net.0.proj`` (value-first ``[hidden; gate]``) -> ``mlp.fc1`` (gate-first, halves swapped),
  ``ff.net.2`` -> ``mlp.fc2``;
* model-level names per vLLM-Omni's ``fasth3`` table (``proj_in`` -> ``video_patch_proj`` ...).

The tables mirror ``vllm_omni.diffusion.models.minimax_h3.fasth3`` (checked by a test when that
package is importable). Mapping is pure data movement:
``native_to_diffusers(diffusers_to_native(x))`` is bitwise ``x``.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

PLAIN, SWAP_HALVES = "plain", "swap_halves"
QKV_SLOTS = ("q", "k", "v")

MODEL_LEVEL = {
    "proj_in": "video_patch_proj",
    "proj_out": "final_layer.video_out",
    "audio_proj_in": "audio_patch_proj",
    "audio_proj_out": "final_layer.audio_out",
    "context_embedder": "condition_proj",
    "time_embedder.linear_1": "time_embedder.proj_in",
    "time_embedder.linear_2": "time_embedder.proj_out",
    "norm_out.linear": "final_layer.adaln_proj.linear",
    "norm_out.norm": "final_layer.norm",
    "token_refiner.final_norm": "token_refiner.final_norm",
}
BLOCK_PREFIXES = (
    ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
    ("transformer_blocks.", "blocks."),
)
BLOCK_TARGETS = {
    "attn.to_q": ("attn.qkv_proj", "q"),
    "attn.to_k": ("attn.qkv_proj", "k"),
    "attn.to_v": ("attn.qkv_proj", "v"),
    "attn.to_out.0": ("attn.out_proj", PLAIN),
    "attn.norm_q": ("attn.q_norm", PLAIN),
    "attn.norm_k": ("attn.k_norm", PLAIN),
    "ff.net.0.proj": ("mlp.fc1", SWAP_HALVES),
    "ff.net.2": ("mlp.fc2", PLAIN),
    "adaln_proj.linear": ("adaln_proj.linear", PLAIN),
    "norm1": ("norm1", PLAIN),
    "norm2": ("norm2", PLAIN),
}


def resolve(module: str) -> tuple[str, str] | None:
    """Diffusers module path -> ``(native module path, layout)`` or ``None``."""

    if module in MODEL_LEVEL:
        return MODEL_LEVEL[module], PLAIN
    for source, target in BLOCK_PREFIXES:
        if module.startswith(source):
            index, _, suffix = module[len(source) :].partition(".")
            mapped = BLOCK_TARGETS.get(suffix)
            if index.isdigit() and mapped is not None:
                return f"{target}{index}.{mapped[0]}", mapped[1]
    return None


def swap_halves(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=0)
    return torch.cat((second, first), dim=0)


def fuse_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``[Q; K; V]`` rows (the serving layout)."""

    return torch.cat([q, k, v], dim=0)


def split_qkv(fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if fused.shape[0] % 3:
        raise ValueError(f"fused QKV rows {fused.shape[0]} are not a multiple of 3")
    q, k, v = fused.chunk(3, dim=0)
    return q, k, v


def grouped_qkv_to_qkv(weight: torch.Tensor, *, heads: int, head_dim: int) -> torch.Tensor:
    """Native checkpoint's per-head ``[q,k,v]`` groups -> ``[Q; K; V]`` (vLLM-Omni's loader)."""

    per_group = 3 * head_dim
    if weight.shape[0] != heads * per_group:
        raise ValueError(f"grouped QKV rows {weight.shape[0]} != {heads * per_group}")
    grouped = weight.reshape(heads, per_group, *weight.shape[1:])
    q, k, v = grouped.split(head_dim, dim=1)
    return torch.cat([t.reshape(heads * head_dim, *weight.shape[1:]) for t in (q, k, v)])


def qkv_to_grouped(fused: torch.Tensor, *, heads: int, head_dim: int) -> torch.Tensor:
    q, k, v = split_qkv(fused)
    parts = [t.reshape(heads, head_dim, *fused.shape[1:]) for t in (q, k, v)]
    return torch.cat(parts, dim=1).reshape(heads * 3 * head_dim, *fused.shape[1:])


def diffusers_to_native(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Diffusers state dict -> native names, fused QKV, gate-first ``fc1``; unmapped keys fail."""

    out: dict[str, torch.Tensor] = {}
    qkv: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in state.items():
        module, _, kind = name.rpartition(".")
        target = resolve(module)
        if target is None or kind not in ("weight", "bias"):
            raise ValueError(f"unmapped Diffusers H3 tensor: {name}")
        native, layout = target
        key = f"{native}.{kind}"
        if layout in QKV_SLOTS:
            qkv.setdefault(key, {})[layout] = tensor
        elif layout == SWAP_HALVES:
            out[key] = swap_halves(tensor)
        else:
            out[key] = tensor
    for key, parts in qkv.items():
        if set(parts) != set(QKV_SLOTS):
            raise ValueError(f"{key} needs q, k and v, got {sorted(parts)}")
        out[key] = fuse_qkv(parts["q"], parts["k"], parts["v"])
    return out


def _native_block(module: str) -> tuple[str, str, str] | None:
    for source, target in BLOCK_PREFIXES:
        if module.startswith(target):
            index, _, suffix = module[len(target) :].partition(".")
            if index.isdigit():
                return source, index, suffix
    return None


def native_to_diffusers(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Inverse of :func:`diffusers_to_native` (splits fused QKV, swaps ``fc1`` halves back)."""

    model = {native: diff for diff, native in MODEL_LEVEL.items()}
    block = {
        native: (diff, layout)
        for diff, (native, layout) in BLOCK_TARGETS.items()
        if layout not in QKV_SLOTS
    }
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        module, _, kind = name.rpartition(".")
        if module in model:
            out[f"{model[module]}.{kind}"] = tensor
            continue
        parsed = _native_block(module)
        if parsed is None:
            raise ValueError(f"unmapped native H3 tensor: {name}")
        source, index, suffix = parsed
        if suffix == "attn.qkv_proj":
            for slot, part in zip(QKV_SLOTS, split_qkv(tensor)):
                out[f"{source}{index}.attn.to_{slot}.{kind}"] = part
        elif suffix in block:
            diff, layout = block[suffix]
            out[f"{source}{index}.{diff}.{kind}"] = (
                swap_halves(tensor) if layout == SWAP_HALVES else tensor
            )
        else:
            raise ValueError(f"unmapped native H3 tensor: {name}")
    return out
