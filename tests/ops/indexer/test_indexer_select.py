# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 indexer KV selection (DeepGEMM MQA logits + stable top-k) on SM90.

The selected ids equal the golden selector applied to the DeepGEMM logits,
are batch-invariant, and decode selects exactly what prefill selects for the
same query and KV bytes.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.cuda.deepgemm import deepgemm_available
from rl_engine.kernels.ops.pytorch.indexer import indexer_q_rope_fp8
from rl_engine.kernels.ops.pytorch.topk import stable_topk_per_row


def _ready() -> bool:
    if not torch.cuda.is_available() or not deepgemm_available():
        return False
    try:
        import rl_engine._C as ext
    except ImportError:
        return False
    return hasattr(ext, "stable_topk_per_row_out")


pytestmark = pytest.mark.skipif(
    not _ready(), reason="SM90 + pinned DeepGEMM + CUDA 12.8 JIT + rl_engine._C required"
)

H, D, TOPK, BLOCK = 64, 128, 512, 64


def _kv(n, seed):
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(n, D, generator=gen).cuda()
    amax = k.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (k / scale).to(torch.float8_e4m3fn), scale.squeeze(-1).contiguous()


def _queries(t, seed):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(t, H, D, generator=gen).to(torch.bfloat16).cuda()
    w = torch.randn(t, H, generator=gen).to(torch.bfloat16).cuda()
    pos = torch.randint(0, 4096, (t,), generator=gen).cuda()
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2).double() / 64))
    ang = torch.arange(4096).double()[:, None] * inv[None]
    cache = torch.cat([ang.cos(), ang.sin()], -1).float().cuda()
    q8, wo = indexer_q_rope_fp8(q, pos, cache, w, D**-0.5, H**-0.5)
    return q8.contiguous(), wo.contiguous()


def test_prefill_selection_matches_golden_and_is_batch_invariant():
    from rl_engine.kernels.ops.cuda.indexer.select import (
        indexer_logits_prefill,
        indexer_select_prefill,
    )

    n, t = 3000, 400
    k8, ksc = _kv(n, 0)
    q8, w = _queries(t, 1)
    ke = torch.clamp((torch.arange(t) + 1) * n // t, min=1).int().cuda()
    ks = torch.zeros_like(ke)
    ids = indexer_select_prefill(q8, w, k8, ksc, ks, ke, TOPK)
    logits = indexer_logits_prefill(q8, w, k8, ksc, ks, ke)
    assert torch.equal(ids, stable_topk_per_row(logits, ks, ke, TOPK))
    first = int(ke[0])  # query 0 sees 7 candidates: all selected, then -1 padding
    assert sorted(ids[0, :first].tolist()) == list(range(first))
    assert bool((ids[0, first:] == -1).all())
    for lo, hi in ((0, 1), (37, 38), (100, 229), (399, 400)):
        sub = indexer_select_prefill(q8[lo:hi], w[lo:hi], k8, ksc, ks[lo:hi], ke[lo:hi], TOPK)
        assert torch.equal(sub, ids[lo:hi]), (lo, hi)


def _paged(reqs, seed):
    blocks = [-(-r[0].shape[0] // BLOCK) for r in reqs]
    total = sum(blocks) + 2
    order = torch.randperm(total, generator=torch.Generator().manual_seed(seed))
    cache = torch.zeros(total, BLOCK * (D + 4), dtype=torch.uint8, device="cuda")
    table = torch.zeros(len(reqs), max(blocks), dtype=torch.int32, device="cuda")
    used = 0
    for i, (k8, sc) in enumerate(reqs):
        for b in range(blocks[i]):
            phys = int(order[used])
            used += 1
            table[i, b] = phys
            lo, hi = b * BLOCK, min(k8.shape[0], (b + 1) * BLOCK)
            cache[phys, : (hi - lo) * D] = k8[lo:hi].reshape(-1).view(torch.uint8)
            cache[phys, BLOCK * D : BLOCK * D + (hi - lo) * 4] = sc[lo:hi].view(torch.uint8)
    lens = torch.tensor([[r[0].shape[0]] for r in reqs], dtype=torch.int32, device="cuda")
    return cache.view(total, BLOCK, 1, D + 4), table, lens


def test_decode_selects_what_prefill_selects():
    from rl_engine.kernels.ops.cuda.indexer.select import (
        indexer_select_decode,
        indexer_select_prefill,
    )

    lengths = [1, 300, 512, 513, 2000, 4097]
    reqs = [_kv(n, 10 + i) for i, n in enumerate(lengths)]
    q8, w = _queries(len(lengths), 2)
    cache, table, lens = _paged(reqs, 3)
    max_len = 4160
    ids = indexer_select_decode(q8.unsqueeze(1), w, cache, lens, table, max_len, TOPK)
    for i, (k8, sc) in enumerate(reqs):
        zero = torch.zeros(1, dtype=torch.int32, device="cuda")
        end = torch.tensor([lengths[i]], dtype=torch.int32, device="cuda")
        pre = indexer_select_prefill(q8[i : i + 1], w[i : i + 1], k8, sc, zero, end, TOPK)
        assert torch.equal(ids[i : i + 1], pre), lengths[i]
        alone = indexer_select_decode(
            q8[i : i + 1].unsqueeze(1),
            w[i : i + 1],
            cache,
            lens[i : i + 1],
            table[i : i + 1],
            max_len,
            TOPK,
        )
        assert torch.equal(alone, ids[i : i + 1]), lengths[i]
