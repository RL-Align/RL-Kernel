# SPDX-License-Identifier: Apache-2.0
"""Original router input/weight regression for the observed batch-dependent GEMM."""

import json
import os
from pathlib import Path
import pytest
import torch
from safetensors import safe_open
from rl_engine.backends.rocm.moe.router_projection import router_projection

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_original_router_batch_rows_repeat_and_leaf_vjp():
    if not os.getenv("RLK_DSV4_TRACE_ROOT") or not os.getenv("RLK_DSV4_CHECKPOINT"):
        pytest.skip("set original same-version batch captures and checkpoint")
    root = Path(os.environ["RLK_DSV4_TRACE_ROOT"])

    def load(case, name, occurrence=0):
        directory = root / f"actor-trace-{case}"
        entries = json.loads((directory / "manifest.json").read_text())
        e = [e for e in entries if e["layer"] == 0 and e["name"] == name][occurrence]
        return torch.load(directory / e["file"], weights_only=True, map_location="cpu")

    x = load("v0-A-matrix", "rlk.normalized", 1)
    batch = load("v0-AB-matrix", "rlk.normalized", 1)
    assert torch.equal(x.view(torch.int16), batch[::2].view(torch.int16))
    old_single = load("v0-A-matrix", "layer.0.ffn.router.gating.output")
    old_batch = load("v0-AB-matrix", "layer.0.ffn.router.gating.output")
    assert bool((old_single[:, 0].view(torch.int32) != old_batch[:, 0].view(torch.int32)).any())
    with safe_open(
        str(next(Path(os.environ["RLK_DSV4_CHECKPOINT"]).glob("*.safetensors"))), framework="pt"
    ) as checkpoint:
        weight = checkpoint.get_tensor("layers.0.ffn.gate.weight").cuda().requires_grad_()
    before = weight.detach().clone()
    x = x.cuda().requires_grad_()
    y = router_projection(x, weight)
    repeated = router_projection(x, weight)
    moved = router_projection(batch.cuda(), weight)[::2]
    assert torch.equal(y.view(torch.int32), repeated.view(torch.int32))
    assert torch.equal(y.view(torch.int32), moved.view(torch.int32))
    order = torch.tensor([128, 0, 64], device=x.device)
    selected = router_projection(x.index_select(0, order), weight)
    assert torch.equal(selected.view(torch.int32), y.index_select(0, order).view(torch.int32))
    oracle_x = x.detach().double().requires_grad_()
    oracle_w = weight.detach().double().requires_grad_()
    oracle = oracle_x @ oracle_w.T
    torch.testing.assert_close(y.double(), oracle, rtol=0.0001, atol=0.00001)
    torch.manual_seed(43)
    grad = torch.randn_like(y)
    y.backward(grad)
    oracle.backward(grad.double())
    for actual, expected in ((x.grad, oracle_x.grad), (weight.grad, oracle_w.grad)):
        assert bool(torch.isfinite(actual).all()) and bool(actual.ne(0).any())
        torch.testing.assert_close(
            actual.double(), expected.to(actual.dtype).double(), rtol=0.01, atol=0.0001
        )
    assert torch.equal(before.view(torch.int16), weight.detach().view(torch.int16))
