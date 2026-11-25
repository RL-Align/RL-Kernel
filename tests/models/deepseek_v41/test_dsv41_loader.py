# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4.1 checkpoint layout: checkpoint_spec equals the release's tensors (names, dtypes,
shapes from its safetensors headers, release_tensors.json), MXFP8 32 x 32 dequantization
is exact, and random_checkpoint fills the toy config."""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

from rl_engine.models.deepseek_v41 import Dsv41Config, release_config, toy_config
from rl_engine.models.deepseek_v41.loader import (
    E4M3,
    E8M0,
    checkpoint_spec,
    mxfp8_dequant,
    mxfp8_quantize,
    random_checkpoint,
    skipped,
)

RELEASE = json.loads(pathlib.Path(__file__).with_name("release_tensors.json").read_text())
DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": E4M3,
    "F8_E8M0": E8M0,
    "I8": torch.int8,
}


def _release(entries: dict) -> dict:
    return {n: (tuple(shape), DTYPES[dt]) for n, (dt, shape) in entries.items()}


def test_spec_equals_the_release_layout():
    cfg = Dsv41Config.from_dict(release_config())
    assert checkpoint_spec(cfg, experts=0) == _release(RELEASE["tensors"])
    per_layer = {int(k): v for k, v in RELEASE["expert_tensors_per_layer"].items()}
    assert per_layer == {i: 6 * cfg.n_routed_experts for i in range(cfg.num_hidden_layers)}
    full = checkpoint_spec(cfg, layers=1)
    sample = _release(RELEASE["expert_samples"])
    assert {n: full[n] for n in sample if n.startswith("layers.0.")} == {
        n: v for n, v in sample.items() if n.startswith("layers.0.")
    }


def test_skipped_names():
    assert skipped("mtp.0.attn.wq_a.weight") and skipped("vision.blocks.0.norm.weight")
    assert skipped("layers.3.ffn.gate.bias_vl") and skipped("image_newline")
    assert not skipped("layers.3.ffn.gate.bias") and not skipped("embed.weight")


def test_mxfp8_dequant_is_exact():
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(96, 200, generator=gen) * torch.logspace(-4, 4, 96)[:, None]
    q, s = mxfp8_quantize(w)
    assert q.dtype == E4M3 and s.dtype == E8M0 and s.shape == (3, 7)
    d = mxfp8_dequant(q, s)
    exact = q.float() * s.float().repeat_interleave(32, 0).repeat_interleave(32, 1)[:, :200]
    assert d.dtype == torch.bfloat16 and torch.equal(d.float(), exact)
    rel = ((d.float() - w).abs() / w.abs().amax(1, keepdim=True)).max()
    assert float(rel) < 2**-4  # one E4M3 step at the block max


def test_random_toy_checkpoint_matches_its_spec():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Dsv41Config.from_dict(toy_config())
    try:
        ckpt = random_checkpoint(cfg, experts=2, device=device)
    except (RuntimeError, ImportError) as exc:  # MXFP4 quantization may need CUDA
        pytest.skip(str(exc))
    spec = checkpoint_spec(cfg, experts=2)
    assert set(ckpt) == set(spec)
    for name, (shape, dtype) in spec.items():
        assert tuple(ckpt[name].shape) == shape and ckpt[name].dtype == dtype, name
    w = mxfp8_dequant(ckpt["layers.0.attn.wq_a.weight"], ckpt["layers.0.attn.wq_a.scale"])
    assert torch.isfinite(w.float()).all() and 0.5 < float(w.float().std() * 5120**0.5) < 2
