# SPDX-License-Identifier: Apache-2.0
"""Actual C4 FP32 projected operands, original BF16 checkpoint weights."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from rl_engine.integrations.engines.train.megatron.native_compressor_projection import (
    CompressorProjectionFP32,
)

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


@pytest.mark.parametrize("name,offset", (("wkv", 0), ("wgate", 1024)))
def test_original_c4_projection_fp32_and_backward(name, offset):
    if not os.getenv("RLK_DSV4_TRACE_ROOT") or not os.getenv("RLK_DSV4_CHECKPOINT"):
        pytest.skip("set original checkpoint and real C4 trace")
    root = Path(os.environ["RLK_DSV4_TRACE_ROOT"])
    ap, rp = root / "actor-trace-v0-A", root / "generation-trace"
    am = json.loads((ap / "manifest.json").read_text())
    rm = [json.loads(s) for s in (rp / "manifest.jsonl").read_text().splitlines()]

    def load(entries, label, path):
        e = next(
            e
            for e in entries
            if e["name"] == label and e["layer"] == 2 and e.get("forward", 0) == 0
        )
        return torch.load(path / e["file"], weights_only=True, map_location="cpu")

    x = (
        load(am, "layer.2.attn.core_attention.compressor.input", ap)[0][:129]
        .cuda()
        .requires_grad_()
    )
    expected = load(rm, "layer.2.attn.compressor.input", rp)[0][:, offset : offset + 1024]
    old = load(am, "layer.2.attn.core_attention.compressor.linear_" + name + ".output", ap)[0]
    assert old.dtype == torch.bfloat16 and expected.dtype == torch.float32
    with safe_open(
        str(next(Path(os.environ["RLK_DSV4_CHECKPOINT"]).glob("*.safetensors"))), framework="pt"
    ) as source:
        weight = source.get_tensor("layers.2.attn.compressor." + name + ".weight").cuda()
    ordinary = torch.nn.Linear(
        weight.shape[1], weight.shape[0], bias=False, device="cuda", dtype=weight.dtype
    )
    with torch.no_grad():
        ordinary.weight.copy_(weight)
    op = CompressorProjectionFP32(ordinary)
    assert op.weight is ordinary.weight
    actual = op(x)[0]
    assert actual.dtype == torch.float32
    assert torch.equal(
        actual.detach().cpu().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )
    actual.square().mean().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(x.grad.ne(0).any())
    assert op.weight.grad is not None and bool(op.weight.grad.ne(0).any())
