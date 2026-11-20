# SPDX-License-Identifier: Apache-2.0
"""Replay actual four-layer decode operands; never substitute model outputs."""

import json
import os
from pathlib import Path

import pytest
import torch

from rl_engine.backends.rocm.attention.native_sparse import NativeSparseAttention

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_real_paged_decode_fixed_math_repeat_and_batch_rows():
    if not os.getenv("RLK_DSV4_TRACE_ROOT"):
        pytest.skip("set real decode worker trace fixture")
    root = Path(os.environ["RLK_DSV4_TRACE_ROOT"]) / "generation-trace"
    entries = [json.loads(s) for s in (root / "manifest.jsonl").read_text().splitlines()]
    operator = NativeSparseAttention()
    for layer in range(4):

        def load(suffix):
            entry = next(e for e in entries if e["name"] == f"layer.{layer}.attn.{suffix}")
            return torch.load(root / entry["file"], weights_only=True, map_location="cpu")

        values = load("sparse_decode.operands")
        args = [values[k].cuda() for k in ("q", "kv", "sink", "indices")]
        actual = operator(*args, values["scale"])
        expected = load("sparse_decode.unrotated_output").cuda()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        repeat = operator(*args, values["scale"])
        assert torch.equal(actual.view(torch.uint8), repeat.view(torch.uint8))
        q, kv, sink, indices = args
        moved = operator(q.repeat(2, 1, 1), kv, sink, indices.repeat(2, 1), values["scale"])
        assert torch.equal(moved.view(torch.uint8), actual.repeat(2, 1, 1).view(torch.uint8))
        print("ORIGINAL_DECODE_REPLAY", layer, actual.numel(), "mismatch_count=0", flush=True)
