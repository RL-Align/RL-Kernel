# SPDX-License-Identifier: Apache-2.0
"""First actual hash probability mismatch, unchanged checkpoint IDs and logits."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from rl_engine.backends.rocm.moe.native_hash_route import native_hash_route

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_original_hash_probabilities_match_rollout_and_have_vjp():
    if not os.getenv("RLK_DSV4_TRACE_ROOT") or not os.getenv("RLK_DSV4_CHECKPOINT"):
        pytest.skip("set original checkpoint and actual router trace")
    root = Path(os.environ["RLK_DSV4_TRACE_ROOT"]) / "generation-trace"
    manifest = [json.loads(s) for s in (root / "manifest.jsonl").read_text().splitlines()]

    def load(name):
        e = next(e for e in manifest if e["name"] == name and e["layer"] == 0 and e["forward"] == 0)
        return torch.load(root / e["file"], weights_only=True, map_location="cpu")

    payload = load("layer.0.ffn.router.select.input")
    expected, expected_ids = load("layer.0.ffn.router.select.output")
    logits = payload["router_logits"].cuda().requires_grad_()
    tokens = payload["input_ids"].cuda()
    with safe_open(
        str(next(Path(os.environ["RLK_DSV4_CHECKPOINT"]).glob("*.safetensors"))), framework="pt"
    ) as source:
        table = source.get_tensor("layers.0.ffn.gate.tid2eid").cuda()
    actual, ids = native_hash_route(logits, tokens, table, 1.5)
    assert torch.equal(ids.cpu(), expected_ids)
    assert torch.equal(actual.detach().cpu().view(torch.uint8), expected.view(torch.uint8))
    torch.manual_seed(42)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    assert bool(torch.isfinite(logits.grad).all()) and bool(logits.grad.ne(0).any())
    oracle = logits.detach().double().requires_grad_()
    scores = F.softplus(oracle.gather(1, ids.long())).sqrt()
    weights = scores / scores.sum(-1, keepdim=True) * 1.5
    weights.backward(grad.double())
    torch.testing.assert_close(logits.grad.double(), oracle.grad, rtol=0.0001, atol=0.000001)
