# SPDX-License-Identifier: Apache-2.0
"""Original updated checkpoint/capture regression; separate from model acceptance."""

import json
import os
from pathlib import Path
import pytest
import torch
from rl_engine.backends.rocm.norm.qnorm_fp8 import fused_kv_norm

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_actual_updated_kv_norm_bytes_repeat_rows_and_vjp():
    if not os.getenv("RLK_DSV4_TRACE_ROOT"):
        pytest.skip("set the real AMD updated-cycle capture")
    root = Path(os.environ["RLK_DSV4_TRACE_ROOT"])
    actor = root / "actor-trace-v1-A-fixed"
    rollout = root / "generation-trace"
    ae = json.loads((actor / "manifest.json").read_text())
    re = [json.loads(s) for s in (rollout / "manifest.jsonl").read_text().splitlines()]
    forward = json.loads((root / "rollout-batches-v1-A-fixed.json").read_text())[0]["batches"][0][
        "forward"
    ]

    def a(name):
        e = next(e for e in ae if e["name"] == name and e["layer"] == 2)
        x = torch.load(actor / e["file"], weights_only=True, map_location="cpu")
        return x[0] if isinstance(x, (tuple, list)) else x

    e = next(
        e
        for e in re
        if e["name"] == "layer.2.attn.method._split_qkv_and_norm"
        and e["layer"] == 2
        and e["forward"] == forward
    )
    _, _, expected = torch.load(rollout / e["file"], weights_only=True, map_location="cpu")
    old = a("layer.2.attn.kv_layernorm.output")[:, 0]
    assert bool((old.view(torch.int16) != expected.view(torch.int16)).any())
    raw = a("layer.2.attn.kv_layernorm.input").cuda().requires_grad_()
    q = a("layer.2.attn.q_layernorm.input").cuda()
    payload = torch.load(root / "actor-update.pt", weights_only=True, map_location="cpu", mmap=True)
    weight = payload["weights"]["layers.2.attn.kv_norm.weight"].cuda().requires_grad_()
    qweight = payload["weights"]["layers.2.attn.q_norm.weight"].cuda()

    def run(x, q):
        return fused_kv_norm(x, weight, q, qweight, 1e-6, 1e-6)

    out = run(raw, q)
    assert torch.equal(out[:, 0].detach().cpu().view(torch.int16), expected.view(torch.int16))
    assert torch.equal(run(raw, q).view(torch.int16), out.view(torch.int16))
    order = torch.tensor([128, 0, 94], device=raw.device)
    assert torch.equal(run(raw[order], q[order]).view(torch.int16), out[order].view(torch.int16))
    torch.manual_seed(47)
    grad = torch.randn_like(out)
    out.backward(grad)
    x = raw.detach().double().requires_grad_()
    w = weight.detach().double().requires_grad_()
    ref = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w
    ref.backward(grad.double())
    for actual, expected_grad in ((raw.grad, x.grad), (weight.grad, w.grad)):
        assert bool(torch.isfinite(actual).all()) and bool(actual.ne(0).any())
        torch.testing.assert_close(
            actual.double(), expected_grad.to(actual.dtype).double(), rtol=0.01, atol=0.00001
        )
    print("UPDATED_KVNORM_ORIGINAL_BYTES", out.numel(), "mismatch_count=0", flush=True)
