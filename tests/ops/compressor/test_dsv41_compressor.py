# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4.1 compressor entries: the ratio-2 main KV (window 2, no APE), the unpooled ratio-1
main KV and the indexer K (``k_norm(wk(latent))``) are byte-equal to the golden, with the
pre-RoPE latent; batch-invariant; the backward at ratios 1 and 2 matches autograd."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.compressor import (
    dsv41_compress_main,
    dsv41_store_rows,
    save_states,
)

from .test_compress_main import EPS, K_BS, NOPE, Pool, _cos_sin


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        import rl_engine._C as ext
    except ImportError:
        return False
    return hasattr(ext, "dsv41_compress_main") and hasattr(ext, "dsv41_store_rows")


pytestmark = pytest.mark.skipif(not _ready(), reason="CUDA rl_engine._C with DSv4.1 required")

D = 512


def _batch(lengths):
    req = torch.cat([torch.full((n,), r, dtype=torch.int32) for r, n in enumerate(lengths)]).cuda()
    pos = torch.cat([torch.arange(n) for n in lengths]).cuda()
    return req, pos


def _same(a, b):
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


# --------------------------------------------------------------------------- ratio 2


def _run_r2(lengths, kv, score, w, req, pos, cos_sin, seed):
    from rl_engine.kernels.ops.cuda.compressor import dsv41_compress_main_cuda

    pool = Pool(lengths, 2, seed=seed)
    s_slots, k_slots = pool.slots(req.long(), pos)
    save_states(kv, score, torch.zeros(2, D, device="cuda"), pos, pool.state, s_slots)
    latent = torch.zeros(len(pos), D, dtype=torch.bfloat16, device="cuda")
    dsv41_compress_main_cuda(
        pool.state, pool.s_table, req, pos, w, EPS, cos_sin, pool.cache, K_BS, k_slots, latent
    )
    return pool, k_slots, latent


@pytest.mark.parametrize("scales", [(1.0, 2.0), (1e-3, 0.1), (40.0, 30.0)])
def test_ratio2_matches_golden(scales):
    lengths = [1, 2, 3, 128, 301, 1024]
    gen = torch.Generator().manual_seed(1)
    total = sum(lengths)
    kv = (torch.randn(total, D, generator=gen) * scales[0]).cuda()
    score = (torch.randn(total, D, generator=gen) * scales[1]).cuda()
    w = (1 + 0.1 * torch.randn(D, generator=gen)).to(torch.bfloat16).cuda()
    req, pos = _batch(lengths)
    cos_sin = _cos_sin()
    pool, k_slots, latent = _run_r2(lengths, kv, score, w, req, pos, cos_sin, seed=0)

    gold = Pool(lengths, 2, seed=5)
    s_slots, _ = gold.slots(req.long(), pos)
    save_states(kv, score, torch.zeros(2, D, device="cuda"), pos, gold.state, s_slots)
    q, sc, rope, tokens, lat = dsv41_compress_main(
        gold.state, gold.s_table, req, pos, w, EPS, cos_sin
    )
    assert len(tokens) == int((k_slots >= 0).sum()) > 0
    for i, slot in enumerate(k_slots[tokens].tolist()):
        got_q, got_sc, got_rope = pool.entry(slot)
        _same(got_q, q[i])
        _same(got_sc, sc[i])
        _same(got_rope, rope[i])
    _same(latent[tokens], lat)


def test_ratio2_batch_invariant():
    lengths = [513, 1000, 2048]
    gen = torch.Generator().manual_seed(3)
    total = sum(lengths)
    kv, score = (
        torch.randn(total, D, generator=gen).cuda(),
        torch.randn(total, D, generator=gen).cuda(),
    )
    w = (1 + 0.1 * torch.randn(D, generator=gen)).to(torch.bfloat16).cuda()
    req, pos = _batch(lengths)
    cos_sin = _cos_sin(torch.bfloat16)
    full, k_full, lat_full = _run_r2(lengths, kv, score, w, req, pos, cos_sin, seed=4)
    sel = req == 1
    alone, k_alone, lat_alone = _run_r2(
        lengths, kv[sel], score[sel], w, req[sel], pos[sel], cos_sin, seed=8
    )
    a, b = k_full[sel], k_alone
    for x, y in zip(a[a >= 0].tolist(), b[b >= 0].tolist()):
        for u, v in zip(full.entry(x), alone.entry(y)):
            _same(u, v)
    keep = (pos[sel] + 1) % 2 == 0
    _same(lat_full[sel][keep], lat_alone[keep])


# --------------------------------------------------------------------------- unpooled rows


class RowCache:
    """A paged cache of entries ``pos // ratio`` per request, blocks permuted."""

    def __init__(self, lengths, ratio, row_bytes, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.ratio, self.row_bytes = ratio, row_bytes
        blocks = [max(1, -(-(n // ratio) // K_BS)) for n in lengths]
        self.cache = torch.zeros(
            sum(blocks) + 2, K_BS * row_bytes, dtype=torch.uint8, device="cuda"
        )
        perm = torch.randperm(self.cache.shape[0], generator=gen).tolist()
        self.table = torch.zeros(len(lengths), max(blocks), dtype=torch.long)
        for r, nb in enumerate(blocks):
            for b in range(nb):
                self.table[r, b] = perm.pop()
        self.table = self.table.cuda()

    def slots(self, req, pos):
        entry = pos // self.ratio
        k = self.table[req, entry // K_BS] * K_BS + entry % K_BS
        return torch.where((pos + 1) % self.ratio == 0, k, -1)

    def ds_mla(self, slot):
        block, i = self.cache[slot // K_BS], slot % K_BS
        row = block[i * 576 : (i + 1) * 576]
        return row[:NOPE], block[K_BS * 576 + i * 8 : K_BS * 576 + (i + 1) * 8], row[NOPE:]

    def indexer(self, slot):
        block, i = self.cache[slot // K_BS], slot % K_BS
        return block[i * 128 : (i + 1) * 128], block[K_BS * 128 + 4 * i : K_BS * 128 + 4 * i + 4]


def _run_rows(lengths, rows, w, req, pos, cos_sin, ratio, seed):
    from rl_engine.kernels.ops.cuda.compressor import dsv41_store_rows_cuda

    d = rows.shape[1]
    rc = RowCache(lengths, ratio, 584 if d == 512 else 132, seed=seed)
    k_slots = rc.slots(req.long(), pos)
    latent = torch.zeros(len(pos), d, dtype=torch.bfloat16, device="cuda")
    dsv41_store_rows_cuda(rows, pos, w, EPS, cos_sin, rc.cache, K_BS, k_slots, ratio, latent)
    return rc, k_slots, latent


@pytest.mark.parametrize("d,ratio", [(512, 1), (128, 1), (128, 2)])
@pytest.mark.parametrize("mag", [1.0, 1e-3, 50.0])
def test_rows_match_golden(d, ratio, mag):
    lengths = [1, 2, 7, 130, 1000]
    gen = torch.Generator().manual_seed(d + ratio)
    rows = (torch.randn(sum(lengths), d, generator=gen) * mag).cuda()
    w = (1 + 0.1 * torch.randn(d, generator=gen)).to(torch.bfloat16).cuda()
    req, pos = _batch(lengths)
    cos_sin = _cos_sin()
    rc, k_slots, latent = _run_rows(lengths, rows, w, req, pos, cos_sin, ratio, seed=1)
    entry, tokens, lat = dsv41_store_rows(rows, pos, ratio, w, EPS, cos_sin)
    assert len(tokens) == int((k_slots >= 0).sum())
    for i, slot in enumerate(k_slots[tokens].tolist()):
        got = rc.ds_mla(slot) if d == 512 else rc.indexer(slot)
        for g, want in zip(got, entry):
            _same(g, want[i : i + 1].reshape(-1))
    _same(latent[tokens], lat)


def test_rows_skip_missing_slots_and_batch_invariant():
    from rl_engine.kernels.ops.cuda.compressor import dsv41_store_rows_cuda

    lengths = [300, 700]
    rows = torch.randn(sum(lengths), 128, device="cuda")
    w = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    req, pos = _batch(lengths)
    cos_sin = _cos_sin()
    rc, k_full, _ = _run_rows(lengths, rows, w, req, pos, cos_sin, 2, seed=2)
    sel = req == 1
    alone, k_alone, _ = _run_rows(lengths, rows[sel], w, req[sel], pos[sel], cos_sin, 2, seed=6)
    a, b = k_full[sel], k_alone
    for x, y in zip(a[a >= 0].tolist(), b[b >= 0].tolist()):
        for u, v in zip(rc.indexer(x), alone.indexer(y)):
            _same(u, v)
    # A token without a slot writes nothing.
    rc2 = RowCache(lengths, 1, 132, seed=3)
    before = rc2.cache.clone()
    none = torch.full((len(pos),), -1, dtype=torch.long, device="cuda")
    dsv41_store_rows_cuda(rows, pos, w, EPS, cos_sin, rc2.cache, K_BS, none, 1)
    assert torch.equal(rc2.cache, before)


# --------------------------------------------------------------------------- backward


def _reference_entries(score, ratio, length, w, cos_sin):
    """FP32 autograd reference of one request's entries from score [T, (2 if gated) * 512]."""
    n = length // ratio
    kv = score[:, :D].reshape(n, ratio, D) if n else score[:0, :D].reshape(0, ratio, D)
    if ratio > 1:
        gate = score[:, D:].reshape(n, ratio, D)
        x = (torch.softmax(gate, dim=1) * kv).sum(1)
    else:
        x = kv[:, 0]
    r = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    nn = x * r * w[None]
    cs = cos_sin[torch.arange(n, device=score.device) * ratio]
    cos, sin = cs[:, :32], cs[:, 32:]
    even, odd = nn[:, NOPE::2], nn[:, NOPE + 1 :: 2]
    rot = torch.stack([even * cos - odd * sin, odd * cos + even * sin], -1).reshape(n, 64)
    return torch.cat([nn[:, :NOPE], rot], -1)


@pytest.mark.parametrize("ratio", [1, 2])
def test_backward_matches_autograd(ratio):
    import rl_engine._C as ext

    lengths = [6, 9, 64]
    gen = torch.Generator().manual_seed(ratio)
    cols = (2 if ratio > 1 else 1) * D
    score = torch.randn(sum(lengths), cols, generator=gen).cuda()
    w = (1 + 0.1 * torch.randn(D, generator=gen)).cuda()
    cos_sin = _cos_sin()
    d_entries, meta, refs, start, base = [], [], [], 0, 0
    for length in lengths:
        n = length // ratio
        d = torch.randn(n, D, generator=gen).cuda()
        d_entries.append(d)
        meta += [(base + e, start, e) for e in range(n)]
        s = score[start : start + n * ratio].clone().requires_grad_(True)
        (_reference_entries(s, ratio, n * ratio, w, cos_sin) * d).sum().backward()
        g = torch.zeros(length, cols, device="cuda")
        g[: n * ratio] = s.grad
        refs.append(g)
        start, base = start + length, base + n
    d_score = torch.zeros(score.shape, dtype=torch.bfloat16, device="cuda")
    ext.dsv4_compressor_backward(
        score,
        torch.zeros(ratio, D, device="cuda"),
        torch.cat(d_entries),
        torch.tensor(meta, dtype=torch.int64, device="cuda"),
        w.contiguous(),
        cos_sin,
        ratio,
        EPS,
        d_score,
    )
    ref = torch.cat(refs)
    err = (d_score.float() - ref).abs().max() / ref.abs().max()
    assert float(err) < 1e-2, float(err)
