# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 main-KV compressor (head 512, C4 / C128) into the fp8_ds_mla cache."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.compressor import (
    compress_main,
    dequant_ds_mla,
    quant_ds_mla,
    save_states,
)


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        import rl_engine._C as ext
    except ImportError:
        return False
    return hasattr(ext, "compress_main_ds_mla")


pytestmark = pytest.mark.skipif(
    not _ready(), reason="CUDA rl_engine._C with the compressor required"
)

D, ROPE, NOPE = 512, 64, 448
ROW, SCALES = 576, 8
STATE_BS, K_BS = 16, 64
EPS = 1e-6


def _cos_sin(dtype=torch.float32, max_pos=1 << 15):
    inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2).double() / ROPE))
    ang = torch.arange(max_pos).double()[:, None] * inv[None]
    return torch.cat([ang.cos(), ang.sin()], -1).to(dtype).cuda()


class Pool:
    def __init__(self, lengths, ratio, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.ratio = ratio
        width = (2 if ratio == 4 else 1) * D
        s_blocks = [-(-n // STATE_BS) for n in lengths]
        k_blocks = [max(1, -(-(n // ratio) // K_BS)) for n in lengths]
        self.state = torch.full(
            (sum(s_blocks) + 2, STATE_BS, 2 * width), float("nan"), device="cuda"
        )
        self.cache = torch.zeros(
            sum(k_blocks) + 2, K_BS * (ROW + SCALES), dtype=torch.uint8, device="cuda"
        )
        sp = torch.randperm(self.state.shape[0], generator=gen).tolist()
        kp = torch.randperm(self.cache.shape[0], generator=gen).tolist()
        self.s_table = torch.zeros(len(lengths), max(s_blocks), dtype=torch.int32)
        self.k_table = torch.zeros(len(lengths), max(k_blocks), dtype=torch.int32)
        for r in range(len(lengths)):
            for b in range(s_blocks[r]):
                self.s_table[r, b] = sp.pop()
            for b in range(k_blocks[r]):
                self.k_table[r, b] = kp.pop()
        self.s_table, self.k_table = self.s_table.cuda(), self.k_table.cuda()

    def slots(self, req, pos):
        s = self.s_table[req, pos // STATE_BS].long() * STATE_BS + pos % STATE_BS
        entry = pos // self.ratio
        k = self.k_table[req, entry // K_BS].long() * K_BS + entry % K_BS
        return s, torch.where((pos + 1) % self.ratio == 0, k, -1)

    def entry(self, slot):
        block = self.cache[slot // K_BS]
        i = slot % K_BS
        row = block[i * ROW : (i + 1) * ROW]
        sc = block[K_BS * ROW + i * SCALES : K_BS * ROW + (i + 1) * SCALES]
        return row[:NOPE], sc, row[NOPE:].view(torch.bfloat16)


def _inputs(lengths, ratio, seed=1, kv_scale=1.0, score_scale=2.0):
    gen = torch.Generator().manual_seed(seed)
    width = (2 if ratio == 4 else 1) * D
    total = sum(lengths)
    kv = (torch.randn(total, width, generator=gen) * kv_scale).cuda()
    score = (torch.randn(total, width, generator=gen) * score_scale).cuda()
    ape = torch.randn(ratio, width, generator=gen).cuda()
    w = (1 + 0.1 * torch.randn(D, generator=gen)).to(torch.bfloat16).cuda()
    req = torch.cat([torch.full((n,), r, dtype=torch.int32) for r, n in enumerate(lengths)]).cuda()
    pos = torch.cat([torch.arange(n) for n in lengths]).cuda()
    return kv, score, ape, w, req, pos


def _cuda_run(pool, kv, score, ape, w, req, pos, cos_sin):
    from rl_engine.kernels.ops.cuda.compressor import compress_main_cuda

    s_slots, k_slots = pool.slots(req.long(), pos)
    save_states(kv, score, ape, pos, pool.state, s_slots)
    compress_main_cuda(
        pool.state, pool.s_table, req, pos, w, EPS, cos_sin, pool.cache, K_BS, k_slots, pool.ratio
    )
    return k_slots


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("kv_scale,score_scale", [(1.0, 2.0), (1e-3, 0.1), (40.0, 30.0)])
def test_cuda_matches_golden(ratio, kv_scale, score_scale):
    lengths = [3, 128, 300, 1024] if ratio == 4 else [127, 128, 300, 1024, 2600]
    data = _inputs(lengths, ratio, kv_scale=kv_scale, score_scale=score_scale)
    cos_sin = _cos_sin()
    pool = Pool(lengths, ratio)
    k_slots = _cuda_run(pool, *data, cos_sin)
    gold_pool = Pool(lengths, ratio)
    kv, score, ape, w, req, pos = data
    s_slots, _ = gold_pool.slots(req.long(), pos)
    save_states(kv, score, ape, pos, gold_pool.state, s_slots)
    q, sc, rope, tokens = compress_main(
        gold_pool.state, gold_pool.s_table, req, pos, w, EPS, cos_sin, ratio
    )
    assert len(tokens) == int((k_slots >= 0).sum())
    for i, slot in enumerate(k_slots[tokens].tolist()):
        got_q, got_sc, got_rope = pool.entry(slot)
        assert torch.equal(got_q, q[i].view(torch.uint8)), i
        assert torch.equal(got_sc, sc[i]), i
        assert torch.equal(got_rope.view(torch.int16), rope[i].view(torch.int16)), i


@pytest.mark.parametrize("ratio", [4, 128])
def test_batch_invariance(ratio):
    lengths = [513, 1000, 2048]
    kv, score, ape, w, req, pos = _inputs(lengths, ratio, seed=3)
    cos_sin = _cos_sin(torch.bfloat16)
    full = Pool(lengths, ratio, seed=4)
    k_full = _cuda_run(full, kv, score, ape, w, req, pos, cos_sin)
    sel = req == 1
    alone = Pool(lengths, ratio, seed=8)
    k_alone = _cuda_run(alone, kv[sel], score[sel], ape, w, req[sel], pos[sel], cos_sin)
    a, b = k_full[sel], k_alone
    for x, y in zip(a[a >= 0].tolist(), b[b >= 0].tolist()):
        for u, v in zip(full.entry(x), alone.entry(y)):
            assert torch.equal(u.view(torch.uint8), v.view(torch.uint8))


def test_ds_mla_roundtrip_and_scale_bytes():
    gen = torch.Generator().manual_seed(0)
    values = (torch.randn(50, D, generator=gen) * torch.logspace(-3, 3, 50)[:, None]).to(
        torch.bfloat16
    )
    q, sc, rope = quant_ds_mla(values)
    assert q.shape == (50, NOPE) and sc.shape == (50, 8) and rope.shape == (50, ROPE)
    assert bool((sc[:, 7] == 0).all())
    back = dequant_ds_mla(q, sc, rope)
    assert torch.equal(back[:, NOPE:], values[:, NOPE:])
    rel = (back[:, :NOPE].float() - values[:, :NOPE].float()).abs() / values[
        :, :NOPE
    ].float().abs().amax(-1, keepdim=True)
    assert float(rel.max()) < 2**-4  # e4m3 step at the block max
