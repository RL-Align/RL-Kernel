# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 sparse attention core (workspace + canonical indices + FlashMLA).

Checks: values against an eager attention over the same entries; a request's
outputs do not depend on the rest of the batch; decode equals prefill for the
same token; a permuted top-k set gives the same bytes; C4, C128 and
sliding-window-only layers.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.cuda.flashmla import flashmla_available
from rl_engine.kernels.ops.pytorch.compressor import quant_ds_mla

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and flashmla_available()),
    reason="SM90 + pinned FlashMLA required",
)

H, D, WINDOW, BS = 64, 512, 128, 64
SCALE = D**-0.5
ROW = 576 + 8


def _core():
    from rl_engine.kernels.ops.cuda import dsv4_attention

    return dsv4_attention


def _paged_ds_mla(values_per_req, seed):
    """fp8_ds_mla paged cache holding each request's rows; returns cache, table."""
    gen = torch.Generator().manual_seed(seed)
    blocks = [max(1, -(-v.shape[0] // BS)) for v in values_per_req]
    cache = torch.zeros(sum(blocks) + 2, BS * ROW, dtype=torch.uint8, device="cuda")
    perm = torch.randperm(cache.shape[0], generator=gen).tolist()
    table = torch.zeros(len(values_per_req), max(blocks), dtype=torch.int32)
    for r, vals in enumerate(values_per_req):
        for b in range(blocks[r]):
            table[r, b] = perm.pop()
        if vals.shape[0] == 0:
            continue
        q, sc, rope = quant_ds_mla(vals)
        for i in range(vals.shape[0]):
            blk = int(table[r, i // BS])
            j = i % BS
            row = torch.cat([q[i].view(torch.uint8), rope[i].view(torch.uint8)])
            cache[blk, j * 576 : (j + 1) * 576] = row
            cache[blk, BS * 576 + j * 8 : BS * 576 + (j + 1) * 8] = sc[i]
    return cache, table.cuda()


class Setup:
    """Requests with token KV (sliding window) and compressed KV for one layer type."""

    def __init__(self, seq_lens, ratio, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.ratio = ratio
        self.seq_lens = seq_lens
        tok = [(torch.randn(n, D, generator=gen)).to(torch.bfloat16).cuda() for n in seq_lens]
        self.swa, self.swa_table = _paged_ds_mla(tok, seed + 1)
        if ratio > 1:
            comp = [
                (torch.randn(n // ratio, D, generator=gen)).to(torch.bfloat16).cuda()
                for n in seq_lens
            ]
            self.comp, self.comp_table = _paged_ds_mla(comp, seed + 2)
        else:
            self.comp = self.comp_table = None
        self.max_len = 4096
        self.top_k = 512 if ratio == 4 else (self.max_len // ratio if ratio > 1 else 0)
        self.sink = torch.randn(H, generator=gen).cuda()
        self.q = {
            r: (torch.randn(n, H, D, generator=gen)).to(torch.bfloat16).cuda()
            for r, n in enumerate(seq_lens)
        }
        # C4 indexer output: a random valid subset per token, unordered
        self.topk = {}
        if ratio == 4:
            for r, n in enumerate(seq_lens):
                ids = torch.full((n, 512), -1, dtype=torch.int32)
                for p in range(n):
                    avail = (p + 1) // 4
                    k = min(avail, 512)
                    ids[p, :k] = torch.randperm(avail, generator=gen)[:k].int()
                self.topk[r] = ids.cuda()

    def run(self, reqs, query_ranges, permute=False):
        """Attention for tokens ``query_ranges[i] = (lo, hi)`` of request ``reqs[i]``."""
        core = _core()
        dev = "cuda"
        seq = torch.tensor([hi for lo, hi in query_ranges], dtype=torch.int32, device=dev)
        qlen = torch.tensor([hi - lo for lo, hi in query_ranges], dtype=torch.int32, device=dev)
        gather = qlen + torch.clamp(seq - qlen, max=WINDOW - 1)
        layout = core.SlabLayout(
            self.max_len // max(self.ratio, 1) if self.ratio > 1 else 0, int(gather.max())
        )
        sel = torch.tensor(reqs, device=dev)
        ws = core.build_workspace(
            layout,
            seq,
            gather,
            self.swa,
            self.swa_table[sel],
            BS,
            self.comp,
            None if self.comp_table is None else self.comp_table[sel],
            BS,
            self.ratio,
        )
        token_req = torch.cat(
            [torch.full((hi - lo,), i) for i, (lo, hi) in enumerate(query_ranges)]
        ).to(dev)
        token_pos = torch.cat([torch.arange(lo, hi) for lo, hi in query_ranges]).to(dev)
        q = torch.cat([self.q[r][lo:hi] for r, (lo, hi) in zip(reqs, query_ranges)])
        topk = None
        if self.ratio == 4:
            topk = torch.cat([self.topk[r][lo:hi] for r, (lo, hi) in zip(reqs, query_ranges)])
            if permute:
                topk = topk[:, torch.randperm(512, device=dev)]
        idx, lens = core.build_indices(
            layout, token_req, token_pos, seq - gather, WINDOW, max(self.ratio, 1), self.top_k, topk
        )
        out, lse = core.sparse_attention(q, ws, idx, lens, self.sink, SCALE)
        return out, lse, ws, idx, lens

    def reference(self, r, p, out_row):
        """Eager attention for request r, position p over its selected entries."""
        from rl_engine.kernels.ops.cuda.dsv4_attention import read_ds_mla

        rows = []
        lo = max(0, p - WINDOW + 1)
        t = torch.arange(lo, p + 1, device="cuda")
        slots = self.swa_table[r, t // BS].long() * BS + t % BS
        rows.append(read_ds_mla(self.swa, slots, BS))
        if self.ratio > 1:
            if self.ratio == 4:
                e = self.topk[r][p]
                e = e[e >= 0].long()
            else:
                e = torch.arange((p + 1) // self.ratio, device="cuda")
            if len(e):
                slots = self.comp_table[r, e // BS].long() * BS + e % BS
                rows.append(read_ds_mla(self.comp, slots, BS))
        k = torch.cat(rows).float()
        s = (self.q[r][p].float() @ k.t()) * SCALE
        lse = torch.logsumexp(s, -1)
        o = torch.softmax(s, -1) @ k
        return (o * torch.sigmoid(lse - self.sink)[:, None]).to(torch.bfloat16), lse


def _same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("ratio", [4, 128, 1])
def test_values_batch_invariance_and_decode_equals_prefill(ratio):
    seq_lens = [5, 300, 700] if ratio != 128 else [130, 300, 700]
    s = Setup(seq_lens, ratio)
    # Prefill: every token of every request in one batch.
    full_out, full_lse, *_ = s.run([0, 1, 2], [(0, n) for n in seq_lens])
    offsets = [0, seq_lens[0], seq_lens[0] + seq_lens[1]]
    # Values against eager attention.
    for r in range(3):
        for p in (0, seq_lens[r] // 2, seq_lens[r] - 1):
            ref, ref_lse = s.reference(r, p, None)
            got = full_out[offsets[r] + p]
            assert float((got.float() - ref.float()).norm() / ref.float().norm()) < 2e-2
            # The returned LSE (natural log, sink excluded) is what the backward needs.
            lse = full_lse[offsets[r] + p].float().view(-1)[: ref_lse.numel()]
            assert torch.allclose(lse, ref_lse, atol=2e-2, rtol=0), (r, p)
    # A request alone, and in a different batch order.
    alone, _, *_ = s.run([1], [(0, seq_lens[1])])
    assert _same(alone, full_out[offsets[1] : offsets[2]])
    swapped, _, *_ = s.run([2, 1], [(0, seq_lens[2]), (0, seq_lens[1])])
    assert _same(swapped[seq_lens[2] :], full_out[offsets[1] : offsets[2]])
    # Decode: one token per request at its last position, through a decode-shaped workspace.
    dec, _, *_ = s.run([0, 1, 2], [(n - 1, n) for n in seq_lens])
    for r in range(3):
        assert _same(dec[r], full_out[offsets[r] + seq_lens[r] - 1]), r
    # Chunked prefill: the second half of request 2 alone.
    half = seq_lens[2] // 2
    chunk, _, *_ = s.run([2], [(half, seq_lens[2])])
    assert _same(chunk, full_out[offsets[2] + half : offsets[2] + seq_lens[2]])


def test_topk_order_does_not_matter():
    s = Setup([300, 900], 4, seed=5)
    a, la, *_ = s.run([0, 1], [(0, 300), (0, 900)])
    b, lb, *_ = s.run([0, 1], [(0, 300), (0, 900)], permute=True)
    assert _same(a, b) and torch.equal(la, lb)


def test_index_layout():
    s = Setup([10, 700], 4, seed=7)
    _, _, ws, idx, lens = s.run([0, 1], [(0, 10), (0, 700)])
    assert idx.shape[-1] % 128 == 0
    # token 699 of request 1: 175 compressed ids (descending) then 128 window rows ascending
    t = 10 + 699
    row = idx[t, 0]
    n = int(lens[t])
    assert n == 175 + 128
    comp = row[:175]
    assert bool((comp[:-1] > comp[1:]).all())
    win = row[175:n]
    assert bool((win[1:] - win[:-1] == 1).all())
    assert bool((row[n:] == -1).all())


def test_gather_and_save_states_match_goldens():
    from rl_engine.kernels.ops.cuda.compressor import save_states_cuda
    from rl_engine.kernels.ops.cuda.dsv4_attention import ds_mla_gather
    from rl_engine.kernels.ops.pytorch.compressor import dequant_ds_mla, save_states

    gen = torch.Generator().manual_seed(9)
    vals = (
        (torch.randn(300, D, generator=gen) * torch.logspace(-6, 4, 300)[:, None])
        .to(torch.bfloat16)
        .cuda()
    )
    cache, table = _paged_ds_mla([vals], 3)
    slots = (
        table[0, torch.arange(300, device="cuda") // BS].long() * BS
        + torch.arange(300, device="cuda") % BS
    )
    slots[7] = -1
    out = ds_mla_gather(cache, BS, slots)
    q, sc, rope = quant_ds_mla(vals)
    want = dequant_ds_mla(q, sc, rope)
    want[7] = 0
    assert torch.equal(out.view(torch.int16), want.view(torch.int16))

    width = 256
    kv = torch.randn(50, 2 * width, generator=gen).cuda()
    ape = torch.randn(4, width, generator=gen).cuda()
    pos = torch.randint(0, 1000, (50,), generator=gen).cuda()
    slots = torch.randperm(64, generator=gen)[:50].cuda()
    slots[3] = -1
    a = torch.zeros(4, 16, 2 * width, device="cuda")
    b = torch.zeros_like(a)
    save_states(kv[:, :width], kv[:, width:], ape, pos, a, slots)
    save_states_cuda(kv[:, :width], kv[:, width:], ape, pos, b, slots)
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.parametrize("ratio", [4, 128, 1])
def test_compact_workspace_equals_slab(ratio):
    from rl_engine.kernels.ops.cuda import dsv4_attention as core

    seq_lens = [9, 300, 700] if ratio != 128 else [130, 300, 700]
    s = Setup(seq_lens, ratio, seed=13)
    reqs = [0, 1, 2]
    ranges = [(n - 1, n) for n in seq_lens]
    slab_out, slab_lse, *_ = s.run(reqs, ranges)
    token_req = torch.arange(3, device="cuda")
    token_pos = torch.tensor([n - 1 for n in seq_lens], device="cuda")
    q = torch.cat([s.q[r][lo:hi] for r, (lo, hi) in zip(reqs, ranges)])
    topk = None
    if ratio == 4:
        topk = torch.cat([s.topk[r][lo:hi] for r, (lo, hi) in zip(reqs, ranges)])
    ws, idx, lens = core.build_compact(
        token_req,
        token_pos,
        WINDOW,
        max(ratio, 1),
        s.top_k,
        s.swa,
        s.swa_table,
        BS,
        s.comp,
        s.comp_table,
        BS,
        topk,
    )
    out, lse = core.sparse_attention(q, ws, idx, lens, s.sink, SCALE)
    assert _same(out, slab_out) and torch.equal(lse, slab_lse)
