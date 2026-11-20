# SPDX-License-Identifier: Apache-2.0
"""Regression from the first actual inverse-RoPE divergence."""

import json
import os
from pathlib import Path

import pytest
import torch

from rl_engine.backends.rocm.attention.inverse_rope import inverse_rope_tail
from rl_engine.backends.rocm.attention.qkv_preprocess import DSV4QKVPreprocess

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_real_attention_output_tail_bytes_and_vjp():
    from transformers import AutoConfig

    if not os.getenv("RLK_DSV4_TRACE_ROOT") or not os.getenv("RLK_DSV4_CHECKPOINT"):
        pytest.skip("set original checkpoint and actual sparse-forward trace")
    root = Path(os.environ["RLK_DSV4_TRACE_ROOT"])
    ap, rp = root / "actor-trace-v0-A", root / "generation-trace"
    am = json.loads((ap / "manifest.json").read_text())
    rm = [json.loads(s) for s in (rp / "manifest.jsonl").read_text().splitlines()]

    def load(events, name, path):
        e = next(
            e for e in events if e["name"] == name and e["layer"] == 0 and e.get("forward", 0) == 0
        )
        return torch.load(path / e["file"], weights_only=True, map_location="cpu")

    actual_input = load(am, "layer.0.attn.sparse.output", ap)[:129].reshape(129, 1, 64, 512)
    expected = load(rm, "layer.0.attn.method._o_proj.input", rp)[0][..., 448:]
    previous = load(am, "layer.0.attn.group_output_projection.input", ap)[0][:129].reshape(
        129, 64, 512
    )[..., 448:]
    assert not torch.equal(previous.view(torch.uint8), expected.view(torch.uint8))
    tail = actual_input[..., 448:].contiguous().cuda().requires_grad_()
    cache = DSV4QKVPreprocess(
        AutoConfig.from_pretrained(os.environ["RLK_DSV4_CHECKPOINT"]), 1
    ).cos_sin_cache
    positions = torch.arange(129, device="cuda")[:, None]
    out = inverse_rope_tail(tail, positions, cache)
    assert torch.equal(out[:, 0].detach().cpu().view(torch.uint8), expected.view(torch.uint8))
    out.float().square().mean().backward()
    assert (
        tail.grad is not None
        and bool(torch.isfinite(tail.grad).all())
        and bool(tail.grad.ne(0).any())
    )
