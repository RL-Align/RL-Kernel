# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Seeded attention-exit tensors at the QKV suite's image/text/CFG layouts."""

import torch

from rl_engine.reference.model_specific.qwenimage.adaln_mod_projection import HIDDEN
from rl_engine.validation.fixtures.qwenimage_qkv import QKV_LAYOUTS as ATTN_OUT_LAYOUTS
from rl_engine.validation.fixtures.qwenimage_qkv import QKV_WORKLOADS as ATTN_OUT_WORKLOADS
from rl_engine.validation.fixtures.qwenimage_qkv import (
    canonical_qkv_indices as canonical_attn_out_indices,
)
from rl_engine.validation.fixtures.qwenimage_qkv import qkv_workload_rows as attn_out_workload_rows


def make_attn_out_workload(name, dtype, device, seed=386):
    """Construct both output-projection inputs; no attention or model weights run.

    Reuse the QKV suite's geometry and row-layout helpers. Values represent the
    flattened [24,128] head axis after joint attention, with independent image
    and text output-projection parameters and a complete upstream gradient.
    """
    spec = ATTN_OUT_WORKLOADS[name]
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def rand(shape, scale=1):
        return (torch.randn(shape, generator=generator) * scale).to(device=device, dtype=dtype)

    branches = 2 if spec.negative_lengths else 1
    length_sets = [spec.positive_lengths]
    if spec.negative_lengths:
        length_sets.append(spec.negative_lengths)
    lengths = torch.tensor(length_sets, device=device).repeat_interleave(spec.images_per_prompt, 1)
    text_width = int(lengths.max().item())
    text_valid = torch.arange(text_width, device=device)[None, None, :] < lengths[:, :, None]
    image_valid = torch.ones(
        (branches, spec.batch, spec.image_tokens), device=device, dtype=torch.bool
    )
    streams = {}
    for stream, valid in (("image", image_valid), ("text", text_valid)):
        heads = rand((*valid.shape, 24, 128))
        x = heads.flatten(-2) * valid[..., None]
        grad = rand(x.shape, HIDDEN**-0.5) * valid[..., None]
        sample = torch.arange(spec.batch, device=device)[None, :, None].expand_as(valid)
        branch = torch.arange(branches, device=device)[:, None, None].expand_as(valid)
        position = torch.arange(valid.shape[-1], device=device)[None, None, :].expand_as(valid)
        keys = torch.stack((sample, branch, position), -1).to(torch.int64)
        keys[..., 0] = torch.where(valid, keys[..., 0], -1)
        streams[stream] = {
            "x": x,
            "grad": grad,
            "keys": keys,
            "weight": rand((HIDDEN, HIDDEN), HIDDEN**-0.5),
            "bias": rand((HIDDEN,), 0.1),
        }
    return {
        "streams": streams,
        "metadata": {
            "name": name,
            "height": spec.height,
            "width": spec.width,
            "prompt_batch": len(spec.positive_lengths),
            "images_per_prompt": spec.images_per_prompt,
            "effective_batch": spec.batch,
            "cfg_branches": branches,
            "positive_lengths": list(spec.positive_lengths),
            "negative_lengths": list(spec.negative_lengths),
            "image_attn_out_shape": list(streams["image"]["x"].shape),
            "text_attn_out_shape": list(streams["text"]["x"].shape),
            "input_scope": "seeded attention-exit values at model-derived token layouts",
        },
    }


__all__ = [
    "ATTN_OUT_LAYOUTS",
    "ATTN_OUT_WORKLOADS",
    "attn_out_workload_rows",
    "canonical_attn_out_indices",
    "make_attn_out_workload",
]
