# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Load the H3 transformer with only this tensor-parallel rank's projection shards.

The full-model grid harness (``h3_parallel_model``) keeps every weight on every rank and cuts the
shards of the sharded projections (``SHARDED_WEIGHTS``: Q/K/V, the attention output, the fused
FFN input as its hidden and gate halves, and the FFN output) out of them on each call. For
training, a rank should hold only its shards. :func:`load_sharded_model` builds the model with its
parameters on the ``meta`` device, loads every other parameter in full from the checkpoint, and
reads each sharded projection's shard straight from the checkpoint file (a safetensors slice, so
the rest of the weight is never read). The shards are the same bytes the harness cuts
(``parallel.shard_rows`` / ``shard_cols``), so a grid run gives the same results; the full
projection weights stay on ``meta`` and are never touched by the parallel blocks; calling one of
those projections outside the grid raises.

The checkpoint already stores the dtypes ``from_pretrained(torch_dtype=torch.bfloat16)`` gives
(FP32 for the modules Diffusers keeps in FP32, BF16 otherwise), and tensors are loaded as stored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
from rl_engine.validation.models import h3_parallel_model as pm

INDEX = "diffusion_pytorch_model.safetensors.index.json"
BLOCKS = "transformer_blocks"
# Sharded projection -> (checkpoint weight under the block, how to cut it).
SHARD_SOURCES: dict[str, tuple[str, str]] = {
    "attn.to_q.weight": ("attn.to_q.weight", "rows"),
    "attn.to_k.weight": ("attn.to_k.weight", "rows"),
    "attn.to_v.weight": ("attn.to_v.weight", "rows"),
    "attn.to_out.0.weight": ("attn.to_out.0.weight", "cols"),
    "ff.hidden": ("ff.net.0.proj.weight", "rows_first_half"),
    "ff.gate": ("ff.net.0.proj.weight", "rows_second_half"),
    "ff.net.2.weight": ("ff.net.2.weight", "cols"),
}
# Block weights that are only ever used through their shards: left on ``meta``.
UNLOADED = frozenset(src for src, _ in SHARD_SOURCES.values())

assert set(SHARD_SOURCES) == set(pm.SHARDED_WEIGHTS), "keep in step with SHARDED_WEIGHTS"


def _refuse_outside_grid(module, args):
    """Forward pre-hook of a projection whose full weight was left on ``meta``.

    ``F.linear`` with a ``meta`` weight does not raise: it returns a ``meta`` tensor, so a sharded
    model called outside ``parallel_model_execution`` would run on with no values. The parallel
    blocks read the stored shards and never call these modules.
    """

    raise RuntimeError(
        "this projection holds only its tensor-parallel shards (h3_sharded_load); run the model "
        "inside h3_parallel_model.parallel_model_execution on the grid it was loaded for"
    )


def cut_shard(view, shape: tuple[int, ...], how: str, tp: int, tp_rank: int) -> torch.Tensor:
    """This rank's shard of a weight of ``shape`` from ``view`` (a tensor or safetensors slice).

    The bounds are those of ``parallel.shard_rows`` / ``shard_cols`` (whose checks run on a
    ``meta`` stand-in first), so the result is the same bytes they give on the full weight.
    """

    if how == "cols":
        par.shard_cols(torch.empty(shape, device="meta"), tp, tp_rank)
        k = shape[1]
        return view[:, tp_rank * k // tp : (tp_rank + 1) * k // tp]
    if how == "rows":
        offset, n = 0, shape[0]
    elif how in ("rows_first_half", "rows_second_half"):
        if shape[0] % 2:  # a fused [hidden; gate] weight; the runtime refuses an odd one too
            raise ValueError(f"a fused weight of {shape[0]} rows has no equal halves to cut")
        half = shape[0] // 2
        offset, n = (0 if how == "rows_first_half" else half), half
    else:
        raise ValueError(f"unknown shard kind {how!r}")
    par.shard_rows(torch.empty((n, shape[1]), device="meta"), tp, tp_rank)
    return view[offset + tp_rank * n // tp : offset + (tp_rank + 1) * n // tp]


def _set_parameter(model: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    owner, _, leaf = name.rpartition(".")
    module = model.get_submodule(owner) if owner else model
    old = module._parameters[leaf]
    if old is None or tuple(old.shape) != tuple(tensor.shape):
        raise ValueError(f"{name}: checkpoint shape {tuple(tensor.shape)} != model {old}")
    module._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)


def load_sharded_model(
    transformer_dir: str | Path,
    tp: int,
    tp_rank: int,
    device: torch.device | str,
    *,
    config_overrides: dict[str, Any] | None = None,
):
    """The H3 transformer with this rank's projection shards (see the module docstring).

    ``config_overrides`` (tests only, e.g. ``num_layers``) changes the model built; checkpoint
    tensors of blocks the model does not have are not read.
    """

    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel
    from safetensors import safe_open

    par.check_tp(tp, tp_rank)
    root = Path(transformer_dir)
    config = MiniMaxH3Transformer3DModel.load_config(str(root))
    config = {**config, **(config_overrides or {})}
    with init_empty_weights(include_buffers=False):  # parameters on meta, buffers made for real
        model = MiniMaxH3Transformer3DModel.from_config(config)
    weight_map: dict[str, str] = json.loads((root / INDEX).read_text())["weight_map"]
    params = dict(model.named_parameters())
    missing = sorted(set(params) - set(weight_map))
    if missing:
        raise ValueError(f"checkpoint lacks {len(missing)} model parameters, e.g. {missing[:3]}")

    shards: dict[int, dict[str, torch.Tensor]] = {}
    by_file: dict[str, list[str]] = {}
    for name in params:
        by_file.setdefault(weight_map[name], []).append(name)
    for file, names in sorted(by_file.items()):
        with safe_open(str(root / file), framework="pt") as f:
            for name in names:
                parts = name.split(".")
                in_block = parts[0] == BLOCKS
                local = ".".join(parts[2:]) if in_block else ""
                if in_block and local in UNLOADED:
                    view = f.get_slice(name)
                    shape = tuple(view.get_shape())
                    if shape != tuple(params[name].shape):  # the cut follows the stored shape
                        raise ValueError(
                            f"{name}: checkpoint shape {shape} != model {tuple(params[name].shape)}"
                        )
                    for shard_name, (src, how) in SHARD_SOURCES.items():
                        if src == local:
                            shard = cut_shard(view, shape, how, tp, tp_rank)
                            shards.setdefault(int(parts[1]), {})[shard_name] = shard.to(device)
                    continue
                _set_parameter(model, name, f.get_tensor(name).to(device))
    for buffer_owner in model.modules():  # buffers were made for real; place them
        for key, buf in list(buffer_owner._buffers.items()):
            if buf is not None:
                buffer_owner._buffers[key] = buf.to(device)
    for index, block in enumerate(model.transformer_blocks):
        tensors = shards.get(index, {})
        if set(tensors) != set(SHARD_SOURCES):
            raise ValueError(
                f"block {index} lacks shards {sorted(set(SHARD_SOURCES) - set(tensors))}"
            )
        setattr(block, pm.SHARDS_ATTR, {"tp": tp, "tp_rank": tp_rank, "tensors": tensors})
        for src in UNLOADED:
            block.get_submodule(src.rpartition(".")[0]).register_forward_pre_hook(
                _refuse_outside_grid
            )
    return model
