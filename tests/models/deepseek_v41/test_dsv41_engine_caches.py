# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The toy DSv4.1 model on a serving engine's cache layout (``integrations.dsv41.engine_caches``)
gives the bytes of its own caches: vLLM's block sizes (32 window rows, 64 / ratio compressed
entries), every layer's blocks a slice of one shared page, shuffled block tables, and the
ratio-2 compressor state kept in an 8-row ring between steps. The schedule is the gate's
continuous batching (staggered arrivals, chunked prefill, decode beside prefill); Engram
windows come from vLLM-style look-back rows."""

from __future__ import annotations

import pytest
import torch

from tests.models.deepseek_v41.test_dsv41_model import VOCAB, _ready, _same, model  # noqa: F401

pytestmark = pytest.mark.skipif(not _ready(), reason="CUDA rl_engine._C with DSv4.1 required")

PAGE = 69696  # vLLM's block stride on the toy model (bytes)
RING = 8


class _Engine:
    """Caches laid out as vLLM's hybrid allocation: one uint8 page per block, every layer's
    caches at their own offset inside it; each cache kind with its own shuffled table."""

    def __init__(self, m, reqs: int, max_len: int, seed: int):
        from rl_engine.integrations.dsv41.engine_caches import EngineLayer

        cfg = m.config
        gen = torch.Generator().manual_seed(seed)
        parts = []  # (layer, kind, rows per block, bytes per row, dtype)
        for i in range(len(m.layers)):
            parts.append((i, "swa", 32, 584))
            role = cfg.layer_role(i)
            if role.ratio and role.kv_source == i:
                per = 64 // role.ratio
                parts += [(i, "kv", per, 584), (i, "idx_k", per, 132)]
                if role.ratio > 1:
                    parts.append((i, "ring", RING, 4 * 1024))
        offsets, off = {}, 0
        for i, kind, rows, nbytes in parts:
            offsets[(i, kind)] = off
            off += -(-rows * nbytes // 576) * 576
        page = -(-max(off, PAGE) // 64) * 64
        tokens_per_block = 64
        blocks_per_req = -(-max_len // 32) + 1
        self.n = reqs * blocks_per_req + 2
        self.buf = torch.zeros(self.n, page, dtype=torch.uint8, device="cuda")
        self.buf.random_(0, 256)  # stale bytes everywhere

        def view(i, kind, rows, nbytes, dtype=torch.uint8):
            o = offsets[(i, kind)]
            v = self.buf[:, o : o + rows * nbytes]
            if dtype != torch.uint8:
                v = v.view(dtype)
            return v.view(self.n, rows, -1)

        def table(block_tokens):
            width = -(-max_len // block_tokens)
            perm = torch.randperm(self.n - 1, generator=gen)[: reqs * width] + 1
            return perm.view(reqs, width).int().cuda()

        self.layers = []
        for i in range(len(m.layers)):
            role = cfg.layer_role(i)
            el = EngineLayer(swa=view(i, "swa", 32, 584), swa_table=table(32))
            if role.ratio and role.kv_source == i:
                per = 64 // role.ratio
                t = table(tokens_per_block)
                el.kv, el.kv_table = view(i, "kv", per, 584), t
                el.idx_k, el.idx_k_table = view(i, "idx_k", per, 132), t
                if role.ratio > 1:
                    el.ring = view(i, "ring", RING, 4 * 1024, torch.float32)
                    el.ring_blocks = torch.randperm(self.n, generator=gen)[:reqs].cuda()
            self.layers.append(el)


def _rows(el, rows):
    from dataclasses import replace

    keys = ("swa_table", "kv_table", "idx_k_table", "ring_blocks")
    pick = {k: getattr(el, k)[rows] for k in keys if getattr(el, k) is not None}
    return replace(el, **pick)


def _rollout(m, ids, steps, eng):
    from rl_engine.integrations.dsv41.engine_caches import StepCaches, engram_windows
    from rl_engine.models.deepseek_v41.attention import Dsv41Batch

    depth = m.config.engram_max_ngram_size - 1
    outs, layers = [], [[] for _ in m.layers]
    for parts in steps:
        rows = torch.tensor([r for r, _, _ in parts], device="cuda")
        seq = [hi for _, _, hi in parts]
        qlen = [hi - lo for _, lo, hi in parts]
        step = StepCaches(m.config, [_rows(el, rows) for el in eng.layers], seq, qlen)
        tok = torch.cat([ids[r][lo:hi] for r, lo, hi in parts])
        qsl = torch.tensor([0] + qlen).cumsum(0)
        look = torch.full((len(parts) + 3, depth + 2), -7, dtype=torch.int32, device="cuda")
        for j, (r, lo, _) in enumerate(parts):
            for k in range(depth + 2):
                look[j, k] = ids[r][lo - 1 - k] if lo - 1 - k >= 0 else -1
        win = engram_windows(tok, qsl, look, m.config.engram_max_ngram_size)
        batch = Dsv41Batch.from_lens(seq, qlen, m.config.sliding_window)
        h, per = m.forward_batch(tok, batch, step.caches, engram_windows=win, return_streams=True)
        step.finish()
        outs.append(h)
        for acc, s in zip(layers, per):
            acc.append(s)
    return outs, layers


def test_engine_layout_matches_own_caches(model):  # noqa: F811
    from rl_engine.models.deepseek_v4.layerwise_gate import rollout_schedule
    from rl_engine.models.deepseek_v41.layerwise_gate import run_rollout

    g = torch.Generator().manual_seed(11)
    prompts, gens = [190, 37, 301, 5], [23, 40, 9, 31]
    ids = [torch.randint(0, VOCAB, (p + n,), generator=g).cuda() for p, n in zip(prompts, gens)]
    steps = rollout_schedule(prompts, gens, 97)
    eng = _Engine(model, len(ids), max(len(x) for x in ids), seed=3)
    outs, layers = _rollout(model, ids, steps, eng)

    ref = run_rollout(model, ids, prompts, token_budget=97)
    from rl_engine.models.deepseek_v4.layerwise_gate import _canonical

    where = _canonical([len(x) for x in ids], steps).cuda()
    _same(torch.cat(outs)[where], ref.hidden, "final hidden")
    for i, (acc, r) in enumerate(zip(layers, ref.streams)):
        _same(torch.cat(acc)[where], r, f"layer {i}")


def test_engram_windows_match_reference():
    from rl_engine.integrations.dsv41.engine_caches import engram_windows
    from rl_engine.models.deepseek_v41.engram import ngram_windows

    g = torch.Generator().manual_seed(5)
    seqs = [torch.randint(0, 50, (n,), generator=g).cuda() for n in (9, 1, 30)]
    chunks = [(0, 9), (0, 1), (2, 30)]
    pos = [torch.arange(lo, hi, device="cuda") for lo, hi in chunks]
    ref = ngram_windows(seqs, pos, 3)
    tok = torch.cat([s[lo:hi] for s, (lo, hi) in zip(seqs, chunks)])
    qsl = torch.tensor([0, 9, 10, 38])
    look = torch.full((3, 4), -1, dtype=torch.int32, device="cuda")
    look[2, 0], look[2, 1] = seqs[2][1], seqs[2][0]
    assert torch.equal(engram_windows(tok, qsl, look, 3), ref)
