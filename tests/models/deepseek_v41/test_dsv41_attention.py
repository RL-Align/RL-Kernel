# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4.1 attention over the toy config's 7 layers (window, Full r2, Reuse r2, Full r1 with
candidates, Reuse r1, Reindex r1, Reuse r1) with KV / index sharing: a token's output bytes
do not depend on the batch, on chunked prefill or on decode."""

from __future__ import annotations

import pytest
import torch


def _ready() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import rl_engine._C as ext
    except ImportError:
        return False
    return all(
        hasattr(ext, n)
        for n in ("dsv41_q_rope_kv_insert", "dsv41_compress_main", "dsv41_store_rows")
    )


pytestmark = pytest.mark.skipif(not _ready(), reason="CUDA rl_engine._C with DSv4.1 required")

MAX_LEN = 1024


class Stack:
    """The toy config's attention layers with their caches for ``requests`` requests."""

    def __init__(self, requests: int, cache_seed: int, index_topk=16, candidate_blocks=4):
        from rl_engine.models.deepseek_v4.rope import rope_cos_sin
        from rl_engine.models.deepseek_v41 import Dsv41Config, toy_config
        from rl_engine.models.deepseek_v41.attention import (
            Dsv41Attention,
            attention_configs,
            new_group_caches,
            random_layer,
        )

        cfg = Dsv41Config.from_dict(toy_config())
        self.configs = attention_configs(
            cfg, MAX_LEN, index_topk=index_topk, candidate_topk_blocks=candidate_blocks
        )
        self.layers, self.caches = [], []
        kv_owner = {}
        for i, c in enumerate(self.configs):
            cs = rope_cos_sin(cfg, 2 if c.ratio else 1, MAX_LEN)
            self.layers.append(Dsv41Attention(c, random_layer(c, seed=100 + i), cs))
            role = cfg.layer_role(i)
            group = self.caches[role.kv_source] if c.ratio and not c.is_kv_source else None
            self.caches.append(
                new_group_caches(c, requests, MAX_LEN, seed=cache_seed + i, group=group)
            )
            if c.is_kv_source:
                kv_owner[i] = True

    def step(self, ids, seq_lens, query_lens):
        """Per-layer outputs for one step; request r's hidden at position p is a fixed
        function of (ids[r], p, layer)."""
        from rl_engine.models.deepseek_v41.attention import Dsv41Batch

        batch = Dsv41Batch.from_lens(seq_lens, query_lens, 128)
        shared: dict = {}
        outs = []
        for i, (layer, caches) in enumerate(zip(self.layers, self.caches)):
            rows = []
            for r, (s, q) in enumerate(zip(seq_lens, query_lens)):
                g = torch.Generator(device="cuda").manual_seed(1000 * ids[r] + 7 * i)
                full = torch.randn(MAX_LEN, 5120, generator=g, device="cuda").bfloat16()
                rows.append(full[s - q : s])
            outs.append(layer.forward_batch(torch.cat(rows), batch, caches, shared))
        return outs


def _same(a, b, what):
    assert a.shape == b.shape, what
    eq = a.view(torch.int16) == b.view(torch.int16)
    assert bool(eq.all()), f"{what}: {(~eq).sum().item()} of {eq.numel()} differ"


def test_batch_invariant_prefill():
    lens = [300, 77, 513]
    batched = Stack(3, cache_seed=1).step([0, 1, 2], lens, lens)
    offset = 0
    for r, n in enumerate(lens):
        alone = Stack(1, cache_seed=50 + r).step([r], [n], [n])
        for i, (a, b) in enumerate(zip(batched, alone)):
            _same(a[offset : offset + n], b, f"request {r} layer {i}")
        offset += n


def test_chunked_prefill_and_decode_equal_one_prefill():
    n, first = 300, 263
    full = Stack(1, cache_seed=3).step([5], [n], [n])
    chunked = Stack(1, cache_seed=9)
    a = chunked.step([5], [150], [150])
    b = chunked.step([5], [first], [first - 150])
    for i in range(len(full)):
        _same(torch.cat([a[i], b[i]]), full[i][:first], f"chunked layer {i}")
    for p in range(first, n):
        d = chunked.step([5], [p + 1], [1])
        for i in range(len(full)):
            _same(d[i], full[i][p : p + 1], f"decode position {p} layer {i}")


def test_decode_batch_with_other_requests():
    full = Stack(1, cache_seed=4).step([7], [40], [40])
    s = Stack(2, cache_seed=11)
    s.step([7, 8], [39, 90], [39, 90])
    d = s.step([7, 8], [40, 91], [1, 1])
    for i in range(len(full)):
        _same(d[i][:1], full[i][39:40], f"layer {i}")
