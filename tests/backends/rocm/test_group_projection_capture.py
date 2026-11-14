# SPDX-License-Identifier: Apache-2.0
"""Original grouped FP8 projection on the first actual mismatching operands."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from rl_engine.backends.rocm.gemm.checkpoint_fp8 import CheckpointFP8Group

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_original_group_projection_matches_actor_capture_not_bf16_fallback():
    if not os.getenv("RLK_DSV4_TRACE_ROOT") or not os.getenv("RLK_DSV4_CHECKPOINT"):
        pytest.skip("set original checkpoint and real wo_a trace")
    root = Path(os.environ["RLK_DSV4_TRACE_ROOT"])
    ap, rp = root / "actor-trace-v0-A", root / "generation-trace"
    am = json.loads((ap / "manifest.json").read_text())
    rm = [json.loads(s) for s in (rp / "manifest.jsonl").read_text().splitlines()]

    def load(events, name, path):
        e = next(
            e for e in events if e["name"] == name and e["layer"] == 0 and e.get("forward", 0) == 0
        )
        return torch.load(path / e["file"], weights_only=True, map_location="cpu")

    x = (
        load(am, "layer.0.attn.group_output_projection.input", ap)[0][:129, 0]
        .cuda()
        .requires_grad_()
    )
    expected = load(am, "layer.0.attn.group_output_projection.output", ap)[:129, 0]
    old = load(rm, "layer.0.attn.wo_a.bf16_einsum.output", rp)
    assert not torch.equal(old.view(torch.uint8), expected.view(torch.uint8))
    with safe_open(
        str(next(Path(os.environ["RLK_DSV4_CHECKPOINT"]).glob("*.safetensors"))), framework="pt"
    ) as source:
        weight = source.get_tensor("layers.0.attn.wo_a.weight").cuda()
        scale = source.get_tensor("layers.0.attn.wo_a.scale").cuda()
    op = CheckpointFP8Group(weight, scale, 8)
    y = op(x)
    assert torch.equal(y.detach().cpu().view(torch.uint8), expected.view(torch.uint8))
    y.float().square().mean().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(x.grad.ne(0).any())
