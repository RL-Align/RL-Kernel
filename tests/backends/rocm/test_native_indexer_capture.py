# SPDX-License-Identifier: Apache-2.0
"""Original 3899-row real DAPO case beyond the Top512 shortcut."""

import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="ROCm GPU required"
)


def test_original_long_indexer_projections_reproduce_native_selected_sets():
    trace_root = os.getenv("RLK_DSV4_LONG_Q_TRACES")
    checkpoint = os.getenv("RLK_DSV4_CHECKPOINT")
    if not trace_root or not checkpoint:
        pytest.skip("set original trace and checkpoint paths")
    from safetensors import safe_open
    from vllm.transformers_utils.config import get_config
    from examples.operators.dsv4_4l.trace_store import load_capture
    from rl_engine.backends.rocm.attention.native_indexer import native_fp8_indexer
    from rl_engine.backends.rocm.attention.qkv_preprocess import DSV4QKVPreprocess

    def capture(role, name):
        path = Path(trace_root) / f"{role}-rank-0-failure-math-R1-b1"
        entry = next(
            e
            for e in json.loads((path / "manifest.json").read_text())
            if e["name"] == name
        )
        return load_capture(path / entry["file"])

    def projection(name):
        return capture(
            "actor", "layer.2.attn.core_attention.indexer." + name + ".output"
        )[0].cuda()

    q = projection("linear_wq_b")
    weights = projection("linear_weights_proj")
    kv = projection("compressor.linear_wkv")
    score = projection("compressor.linear_wgate")
    with safe_open(
        str(next(Path(checkpoint).glob("*.safetensors"))), framework="pt", device="cuda"
    ) as source:
        ape = source.get_tensor("layers.2.attn.indexer.compressor.ape")
        norm = source.get_tensor("layers.2.attn.indexer.compressor.norm.weight")
    config = get_config(checkpoint, trust_remote_code=False)
    cache = DSV4QKVPreprocess(config, 4).cos_sin_cache
    expected = capture("rollout", "layer.2.attn.indexer.indexer_op.output")[
        :3899
    ].long()
    # B1 and actual padded B2 geometry must select the same real keys.
    for batch in [1, 2]:
        _, indices = native_fp8_indexer(
            q.expand(-1, batch, -1),
            weights.expand(-1, batch, -1),
            kv.expand(-1, batch, -1),
            score.expand(-1, batch, -1),
            ape,
            norm,
            cache,
            1e-6,
        )
        ends = ((torch.arange(3899) + 1) // 4)[:, None]
        expected_valid = (
            expected.masked_fill((expected < 0) | (expected >= ends), -1)
            .sort(-1)
            .values
        )
        for b in range(batch):
            actual = indices[b].cpu()
            actual_valid = (
                actual.masked_fill((actual < 0) | (actual >= ends), -1).sort(-1).values
            )
            assert torch.equal(actual_valid, expected_valid)
