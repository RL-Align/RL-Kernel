# SPDX-License-Identifier: Apache-2.0
"""Fixed sparse production math on original captured four-layer model operands."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from rl_engine.backends.rocm.attention.native_sparse import NativeSparseAttention

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_original_sparse_operands_forward_repeat_and_input_vjp():
    if not os.getenv("RLK_DSV4_TRACE_ROOT"):
        pytest.skip("set real worker trace fixture")
    root = Path(os.environ["RLK_DSV4_TRACE_ROOT"]) / "actor-trace-v0-A"
    manifest = json.loads((root / "manifest.json").read_text())
    e = next(e for e in manifest if e["name"] == "layer.0.attn.sparse.input")
    q, kv, sink, indices, scale = torch.load(
        root / e["file"], weights_only=True, map_location="cpu"
    )
    q = q[:, 0].contiguous().cuda().requires_grad_()
    kv = kv[:, 0].contiguous().cuda().requires_grad_()
    sink = sink.cuda().requires_grad_()
    indices = F.pad(indices[0].cuda(), (0, 640 - indices.shape[-1]), value=-1)
    op = NativeSparseAttention()
    y = op(q, kv, sink, indices, scale)
    repeat = op(q, kv, sink, indices, scale)
    assert torch.equal(y.view(torch.uint8), repeat.view(torch.uint8))
    row_order = torch.tensor([128, 0, 64], device=q.device)
    moved = op(q.index_select(0, row_order), kv, sink, indices.index_select(0, row_order), scale)
    assert torch.equal(moved.view(torch.uint8), y.index_select(0, row_order).view(torch.uint8))
    torch.manual_seed(42)
    grad = torch.randn_like(y)
    y.backward(grad)
    for x in (q, kv, sink):
        assert (
            x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(x.grad.ne(0).any())
        )
    print("ORIGINAL_SPARSE_FORWARD_VJP", list(y.shape), flush=True)
