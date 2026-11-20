# SPDX-License-Identifier: Apache-2.0
"""Actual layer-3 learned probabilities and independently checked VJP."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from rl_engine.backends.rocm.moe.native_learned_route import native_learned_route

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_original_learned_probabilities_and_vjp():
    if not os.getenv("RLK_DSV4_TRACE_ROOT") or not os.getenv("RLK_DSV4_CHECKPOINT"):
        pytest.skip("set original checkpoint and real learned route capture")
    root = Path(os.environ["RLK_DSV4_TRACE_ROOT"]) / "generation-trace"
    entries = [json.loads(s) for s in (root / "manifest.jsonl").read_text().splitlines()]

    def load(name):
        e = next(e for e in entries if e["name"] == name and e["layer"] == 3 and e["forward"] == 0)
        return torch.load(root / e["file"], weights_only=True, map_location="cpu")

    logits = load("layer.3.ffn.router.select.input")["router_logits"].cuda().requires_grad_()
    expected, expected_ids = load("layer.3.ffn.router.select.output")
    with safe_open(
        str(next(Path(os.environ["RLK_DSV4_CHECKPOINT"]).glob("*.safetensors"))), framework="pt"
    ) as source:
        bias = source.get_tensor("layers.3.ffn.gate.bias").cuda()
    weights, ids = native_learned_route(logits, bias, 1.5)
    assert torch.equal(ids.cpu(), expected_ids)
    assert torch.equal(weights.detach().cpu().view(torch.uint8), expected.view(torch.uint8))
    torch.manual_seed(42)
    grad = torch.randn_like(weights)
    weights.backward(grad)
    oracle = logits.detach().double().requires_grad_()
    score = F.softplus(oracle.gather(1, ids.long())).sqrt()
    ref = 1.5 * score / score.sum(-1, keepdim=True)
    ref.backward(grad.double())
    torch.testing.assert_close(logits.grad.double(), oracle.grad, rtol=0.0001, atol=0.000001)
    assert bool(torch.isfinite(logits.grad).all()) and bool(logits.grad.ne(0).any())
