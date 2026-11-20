# SPDX-License-Identifier: Apache-2.0
"""Regression from actual worker operands, not a reference-model forward."""

import json
import os
from pathlib import Path

import pytest
import torch

from rl_engine.backends.rocm.attention.qkv_preprocess import DSV4QKVPreprocess

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_original_query_matches_real_rollout_and_has_input_vjp():
    from transformers import AutoConfig

    if not os.getenv("RLK_DSV4_TRACE_ROOT") or not os.getenv("RLK_DSV4_CHECKPOINT"):
        pytest.skip("set original checkpoint and real-worker trace paths")
    trace = Path(os.environ["RLK_DSV4_TRACE_ROOT"])
    ap, rp = trace / "actor-trace-v0-A", trace / "generation-trace"
    a = json.loads((ap / "manifest.json").read_text())
    r = [json.loads(s) for s in (rp / "manifest.jsonl").read_text().splitlines()]

    def load(events, name, root):
        e = next(
            e
            for e in events
            if e["layer"] == 0 and e["name"] == name and ("forward" not in e or e["forward"] == 0)
        )
        return torch.load(root / e["file"], weights_only=True, map_location="cpu")

    raw = (
        load(a, "layer.0.attn.linear_q_up_proj.output", ap)[0][:129]
        .reshape(129, 1, 64, 512)
        .cuda()
        .requires_grad_(True)
    )
    kv = load(a, "layer.0.attn.kv_layernorm.output", ap)[:129].cuda().requires_grad_(True)
    expected, _, _ = load(r, "layer.0.attn.method.forward_mqa.input", rp)
    config = AutoConfig.from_pretrained(os.environ["RLK_DSV4_CHECKPOINT"])
    op = DSV4QKVPreprocess(config, 1).bfloat16()
    assert op.cos_sin_cache.dtype == torch.float32
    q, cached_kv = op(raw, kv, torch.arange(129, device="cuda")[:, None])
    actual = q[:, 0].detach().cpu()
    bad = actual.view(torch.int16) != expected.view(torch.int16)
    indices = bad.nonzero()
    print(
        "QUERY_BYTE_COMPARISON",
        {"elements": actual.numel(), "mismatches": int(bad.sum()), "first": indices[:4].tolist()},
        flush=True,
    )
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert bool(torch.isfinite(cached_kv).all())
    (q.float().square().mean() + cached_kv.float().square().mean()).backward()
    for grad in (raw.grad, kv.grad):
        assert grad is not None and bool(torch.isfinite(grad).all()) and bool(grad.ne(0).any())
