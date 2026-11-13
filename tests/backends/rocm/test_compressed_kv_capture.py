# SPDX-License-Identifier: Apache-2.0
"""Actual C4 fused compression/cache bytes and nonzero training gradients."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from rl_engine.backends.rocm.attention.native_compressor import compressed_kv
from rl_engine.backends.rocm.attention.qkv_preprocess import DSV4QKVPreprocess

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


@pytest.mark.parametrize("layer,ratio,width,n", ((2, 4, 1024, 32), (3, 128, 512, 1)))
def test_actual_c4_compressed_kv_bytes_and_backward(layer, ratio, width, n):
    from transformers import AutoConfig

    if not os.getenv("RLK_DSV4_TRACE_ROOT") or not os.getenv("RLK_DSV4_CHECKPOINT"):
        pytest.skip("set original checkpoint and real C4 capture")
    root = Path(os.environ["RLK_DSV4_TRACE_ROOT"]) / "generation-trace"
    manifest = [json.loads(s) for s in (root / "manifest.jsonl").read_text().splitlines()]

    def load(name):
        e = next(
            e for e in manifest if e["name"] == name and e["layer"] == layer and e["forward"] == 0
        )
        return torch.load(root / e["file"], weights_only=True, map_location="cpu")

    projection = load(f"layer.{layer}.attn.compressor.input")[0]
    expected = load(f"layer.{layer}.attn.sparse_prefill.input")["kv"][:n, 0]
    kv = projection[:, :width].cuda()[:, None].contiguous().requires_grad_()
    score = projection[:, width:].cuda()[:, None].contiguous().requires_grad_()
    with safe_open(
        str(next(Path(os.environ["RLK_DSV4_CHECKPOINT"]).glob("*.safetensors"))), framework="pt"
    ) as source:
        ape = source.get_tensor(f"layers.{layer}.attn.compressor.ape").cuda().requires_grad_()
        weight = (
            source.get_tensor(f"layers.{layer}.attn.compressor.norm.weight").cuda().requires_grad_()
        )
    config = AutoConfig.from_pretrained(os.environ["RLK_DSV4_CHECKPOINT"])
    cache = DSV4QKVPreprocess(config, ratio).cos_sin_cache
    actual = compressed_kv(kv, score, ape, weight, cache, ratio)
    bad = actual[:, 0].detach().cpu().view(torch.int16) != expected.view(torch.int16)
    print(
        "C4_COMPRESSED_KV_BYTES",
        dict(elements=expected.numel(), mismatch_count=int(bad.sum())),
        flush=True,
    )
    assert not bool(bad.any())
    actual.float().square().mean().backward()
    for x in (kv, score, ape, weight):
        assert (
            x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(x.grad.ne(0).any())
        )
