# SPDX-License-Identifier: Apache-2.0
"""Actual routed activation conversion boundary, distinct from shared experts."""

import json
import os
from pathlib import Path

import pytest
import torch

from rl_engine.integrations.engines.train.megatron.native_w4a16 import _SwiGLU

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_original_routed_activation_bytes_and_backward():
    if not os.getenv("RLK_DSV4_TRACE_ROOT"):
        pytest.skip("set actual routed matmul capture")
    root = Path(os.environ["RLK_DSV4_TRACE_ROOT"]) / "generation-trace"
    entries = [json.loads(s) for s in (root / "manifest.jsonl").read_text().splitlines()]

    def load(name, n=0):
        e = [e for e in entries if e["name"] == name and e["layer"] == 0 and e["forward"] == 0][n]
        return torch.load(root / e["file"], weights_only=True, map_location="cpu")

    x = load("layer.0.ffn.packed.matmul.output").cuda().requires_grad_()
    expected = load("layer.0.ffn.packed.matmul.input", 1)["x"]
    old = torch.empty_like(expected, device="cuda")
    import vllm._custom_ops  # noqa: F401

    torch.ops._C.silu_and_mul_with_clamp(old, x, 10.0, 1.0, 0.0)
    assert not torch.equal(old.cpu().view(torch.uint8), expected.view(torch.uint8))
    actual = _SwiGLU()(x)
    assert torch.equal(actual.detach().cpu().view(torch.uint8), expected.view(torch.uint8))
    actual.float().square().mean().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(x.grad.ne(0).any())
