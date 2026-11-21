# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 indexer compressor: golden contract, CUDA bytes, batch invariance, chunking.

The golden is defined on CUDA (libdevice expf), so these tests need a GPU.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.compressor import compress_indexer, save_states


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        import rl_engine._C as ext
    except ImportError:
        return False
    return hasattr(ext, "compress_indexer_fp8")


pytestmark = pytest.mark.skipif(
    not _ready(), reason="CUDA rl_engine._C with the compressor required"
)

D, ROPE, RATIO = 128, 64, 4
WIDTH = 2 * D  # coff * D for C4
STATE_BS, K_BS = 16, 64
EPS = 1e-6


def _cache(max_pos=1 << 15, dtype=torch.float32):
    inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2).double() / ROPE))
    ang = torch.arange(max_pos).double()[:, None] * inv[None]
    return torch.cat([ang.cos(), ang.sin()], -1).to(dtype).cuda()


class Pool:
    """Paged state and K caches shared by several requests, with shuffled blocks."""

    def __init__(self, lengths, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.lengths = lengths
        s_blocks = [-(-n // STATE_BS) for n in lengths]
        k_blocks = [-(-(n // RATIO) // K_BS) or 1 for n in lengths]
        self.state = torch.full(
            (sum(s_blocks) + 3, STATE_BS, 2 * WIDTH), float("nan"), device="cuda"
        )
        self.k_cache = torch.zeros(
            sum(k_blocks) + 2, K_BS * (D + 4), dtype=torch.uint8, device="cuda"
        )
        sp = torch.randperm(self.state.shape[0], generator=gen)
        kp = torch.randperm(self.k_cache.shape[0], generator=gen)
        self.state_table = torch.zeros(len(lengths), max(s_blocks), dtype=torch.int32)
        self.k_table = torch.zeros(len(lengths), max(k_blocks), dtype=torch.int32)
        si = ki = 0
        for r in range(len(lengths)):
            for b in range(s_blocks[r]):
                self.state_table[r, b] = int(sp[si])
                si += 1
            for b in range(k_blocks[r]):
                self.k_table[r, b] = int(kp[ki])
                ki += 1
        self.state_table = self.state_table.cuda()
        self.k_table = self.k_table.cuda()

    def slots(self, req, pos):
        """State slots and K-cache slots for the given (request, position) tokens."""
        s = self.state_table[req, pos // STATE_BS].long() * STATE_BS + pos % STATE_BS
        boundary = (pos + 1) % RATIO == 0
        entry = pos // RATIO
        k = self.k_table[req, entry // K_BS].long() * K_BS + entry % K_BS
        return s, torch.where(boundary, k, -1)

    def entry(self, slot):
        block = self.k_cache[slot // K_BS]
        i = slot % K_BS
        q = block[i * D : (i + 1) * D]
        s = block[K_BS * D + i * 4 : K_BS * D + (i + 1) * 4].view(torch.float32)
        return q, s


def _inputs(lengths, seed=1, kv_scale=1.0, score_scale=2.0):
    gen = torch.Generator().manual_seed(seed)
    total = sum(lengths)
    kv = (torch.randn(total, WIDTH, generator=gen) * kv_scale).cuda()
    score = (torch.randn(total, WIDTH, generator=gen) * score_scale).cuda()
    ape = torch.randn(RATIO, WIDTH, generator=gen).cuda()
    w = (1 + 0.1 * torch.randn(D, generator=gen)).cuda()
    req = torch.cat([torch.full((n,), r, dtype=torch.int32) for r, n in enumerate(lengths)]).cuda()
    pos = torch.cat([torch.arange(n) for n in lengths]).cuda()
    return kv, score, ape, w, req, pos


def _run(pool, kv, score, ape, w, req, pos, cos_sin, golden):
    s_slots, k_slots = pool.slots(req.long(), pos)
    save_states(kv, score, ape, pos, pool.state, s_slots)
    if golden:
        q, sc, tokens = compress_indexer(pool.state, pool.state_table, req, pos, w, EPS, cos_sin)
        return q, sc, k_slots[tokens]
    from rl_engine.kernels.ops.cuda.compressor import compress_indexer_cuda

    compress_indexer_cuda(
        pool.state, pool.state_table, req, pos, w, EPS, cos_sin, pool.k_cache, K_BS, k_slots
    )
    return None, None, k_slots[k_slots >= 0]


@pytest.mark.parametrize(
    "kv_scale,score_scale", [(1.0, 2.0), (1e-3, 0.1), (50.0, 20.0), (1.0, 60.0), (3e-5, 5.0)]
)
@pytest.mark.parametrize("cs_dtype", [torch.float32, torch.bfloat16])
def test_cuda_matches_golden(kv_scale, score_scale, cs_dtype):
    lengths = [1, 3, 4, 5, 64, 999, 2048]
    data = _inputs(lengths, kv_scale=kv_scale, score_scale=score_scale)
    cos_sin = _cache(dtype=cs_dtype)
    gold_pool, cuda_pool = Pool(lengths), Pool(lengths)
    q, sc, slots = _run(gold_pool, *data, cos_sin, golden=True)
    _, _, cuda_slots = _run(cuda_pool, *data, cos_sin, golden=False)
    assert torch.equal(slots, cuda_slots)
    for i, slot in enumerate(slots.tolist()):
        got_q, got_s = cuda_pool.entry(slot)
        assert torch.equal(got_q, q[i].view(torch.uint8)), i
        assert torch.equal(got_s, sc[i : i + 1]), i


def test_batch_invariance_and_chunked_prefill():
    from rl_engine.kernels.ops.cuda.compressor import compress_indexer_cuda

    lengths = [37, 512, 1000]
    kv, score, ape, w, req, pos = _inputs(lengths, seed=5)
    cos_sin = _cache()
    full = Pool(lengths, seed=2)
    _run(full, kv, score, ape, w, req, pos, cos_sin, golden=False)

    # Request 2 alone, in two chunks (state of chunk 1 feeds chunk 2's windows).
    alone = Pool(lengths, seed=9)
    sel = req == 2
    kv2, sc2, r2, p2 = kv[sel], score[sel], req[sel], pos[sel]
    for lo, hi in ((0, 333), (333, 1000)):
        s_slots, k_slots = alone.slots(r2[lo:hi].long(), p2[lo:hi])
        save_states(kv2[lo:hi], sc2[lo:hi], ape, p2[lo:hi], alone.state, s_slots)
        compress_indexer_cuda(
            alone.state,
            alone.state_table,
            r2[lo:hi],
            p2[lo:hi],
            w,
            EPS,
            cos_sin,
            alone.k_cache,
            K_BS,
            k_slots,
        )
    _, all_slots = full.slots(r2.long(), p2)
    _, alone_slots = alone.slots(r2.long(), p2)
    for a, b in zip(all_slots[all_slots >= 0].tolist(), alone_slots[alone_slots >= 0].tolist()):
        assert torch.equal(full.entry(a)[0], alone.entry(b)[0])
        assert torch.equal(full.entry(a)[1], alone.entry(b)[1])


def test_window_semantics_on_a_single_entry():
    """Uniform scores average the window; the first entry only sees its own tokens."""
    lengths = [8]
    kv, score, ape, w, req, pos = _inputs(lengths, seed=7)
    score.zero_()
    ape.zero_()
    pool = Pool(lengths)
    s_slots, _ = pool.slots(req.long(), pos)
    save_states(kv, score, ape, pos, pool.state, s_slots)
    from rl_engine.kernels.ops.pytorch.compressor import compress_window

    x, tokens = compress_window(pool.state, pool.state_table, req, pos, RATIO, True, D)
    assert tokens.tolist() == [3, 7]
    # entry at pos 3: window positions 0..3 are the newer half (channel half 1)
    torch.testing.assert_close(x[0], kv[0:4, D:].mean(0), rtol=1e-6, atol=1e-6)
    # entry at pos 7: positions 0..3 from half 0, 4..7 from half 1
    torch.testing.assert_close(
        x[1], torch.cat([kv[0:4, :D], kv[4:8, D:]]).mean(0), rtol=1e-6, atol=1e-6
    )
