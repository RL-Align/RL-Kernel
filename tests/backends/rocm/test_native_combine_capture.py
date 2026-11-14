# SPDX-License-Identifier: Apache-2.0
"""Existing P6 on identical real expert outputs; legacy native combine fails."""

import json
import os
from pathlib import Path

import pytest
import torch

from rl_engine.integrations.engines.train.megatron.native_combine import combine_rows

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_actual_original_weighted_expert_combine_and_vjp():
    if not os.getenv("RLK_DSV4_TRACE_ROOT"):
        pytest.skip("set real W4A16 worker trace")
    from megatron.core.transformer.moe.moe_utils import unpermute

    root = Path(os.environ["RLK_DSV4_TRACE_ROOT"])
    ap, rp = root / "actor-trace-v0-A", root / "generation-trace"
    am = json.loads((ap / "manifest.json").read_text())
    rm = [json.loads(s) for s in (rp / "manifest.jsonl").read_text().splitlines()]

    def load(entries, name, path):
        e = next(
            e for e in entries if e["name"] == name and e["layer"] == 0 and e.get("forward", 0) == 0
        )
        return torch.load(path / e["file"], weights_only=True, map_location="cpu")

    _, routing = load(am, "layer.0.ffn.router.output", ap)
    experts, tokens = routing.T.nonzero(as_tuple=True)
    y = load(am, "layer.0.ffn.experts.down.output", ap).cuda().requires_grad_()
    expected = load(rm, "layer.0.ffn.routed_modular.output", rp)
    tokens, experts = tokens.cuda(), experts.cuda()
    result = combine_rows(y, tokens, experts, routing.shape[0], "actual-capture-v0")
    assert torch.equal(result[:129].detach().cpu().view(torch.uint8), expected.view(torch.uint8))
    old = unpermute(y, tokens, restore_shape=result.shape, routing_map=routing.cuda(), fused=False)
    assert not torch.equal(old[:129].detach().cpu().view(torch.uint8), expected.view(torch.uint8))
    for _ in range(5):
        repeat = combine_rows(y, tokens, experts, routing.shape[0], "actual-capture-v0-repeat")
        assert torch.equal(result.view(torch.uint8), repeat.view(torch.uint8))
    result.float().square().mean().backward()
    assert y.grad is not None and bool(torch.isfinite(y.grad).all()) and bool(y.grad.ne(0).any())
