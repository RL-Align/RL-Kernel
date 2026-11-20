# SPDX-License-Identifier: Apache-2.0
"""A stale pre-placement bias reproduces the real updated four-layer routing failure."""

import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from rl_engine.backends.rocm.moe.native_learned_route import native_learned_route
from rl_engine.integrations.engines.train.megatron.checkpoint_state import checkpoint_export_value

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_original_updated_router_buffer_is_exported_after_placement():
    if not os.getenv("RLK_DSV4_TRACE_ROOT") or not os.getenv("RLK_DSV4_CHECKPOINT"):
        pytest.skip("set original updated routing capture and checkpoint")
    fixture = torch.load(
        Path(os.environ["RLK_DSV4_TRACE_ROOT"]) / "updated-router-fixture.pt",
        weights_only=True,
        map_location="cpu",
    )
    with safe_open(
        str(Path(os.environ["RLK_DSV4_CHECKPOINT"]) / "model.safetensors"), framework="pt"
    ) as checkpoint:
        original = checkpoint.get_tensor("layers.3.ffn.gate.bias")
    assert torch.equal(original.view(torch.int32), fixture["bias"].view(torch.int32))
    root = torch.nn.Module()
    root.router = torch.nn.Module()
    root.router.register_buffer("expert_bias", original.clone())
    stale = root.router.expert_bias
    export = checkpoint_export_value(root, stale)
    root.to(device="cuda:0")
    assert root.router.expert_bias is not stale
    # The pinned MCore updates this FP32 state by the recorded expert-count direction.
    root.router.expert_bias.add_(fixture["direction"].to("cuda:0") * 0.001)
    assert export() is root.router.expert_bias
    assert export().dtype == torch.float32
    logits = fixture["logits"].to("cuda:0")

    def route(bias):
        weights, ids = native_learned_route(logits, bias, 1.5)
        dense = torch.zeros_like(logits).scatter(1, ids.long(), weights)
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter(1, ids.long(), True)
        return dense.cpu(), mask.cpu()

    old, _ = route(stale.to("cuda:0"))
    actual, mask = route(export())
    assert torch.equal(old.view(torch.int32), fixture["rollout_weights"].view(torch.int32))
    assert bool((old.view(torch.int32) != fixture["actor_weights"].view(torch.int32)).any())
    assert torch.equal(actual.view(torch.int32), fixture["actor_weights"].view(torch.int32))
    assert torch.equal(mask, fixture["actor_map"])
    assert bool(torch.isfinite(actual).all())
