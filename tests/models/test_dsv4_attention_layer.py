# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 attention layer (T07): the full forward is batch-invariant.

For C4, C128 and sliding-window-only layers, a request's output bytes are the
same whether it runs in a batch, alone, as chunked prefill, or token by token
in decode steps next to other requests.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.cuda.deepgemm import deepgemm_available
from rl_engine.kernels.ops.cuda.flashmla import flashmla_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and deepgemm_available() and flashmla_available()),
    reason="SM90 + pinned DeepGEMM and FlashMLA required",
)

MAX_LEN = 4096


def _cos_sin():
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2).double() / 64))
    ang = torch.arange(MAX_LEN).double()[:, None] * inv[None]
    return torch.cat([ang.cos(), ang.sin()], -1).float().cuda()


def _layer(ratio):
    from rl_engine.models.deepseek_v4.attention import (
        Dsv4Attention,
        Dsv4AttentionConfig,
        random_layer,
    )

    config = Dsv4AttentionConfig(compress_ratio=ratio, max_model_len=MAX_LEN)
    return config, Dsv4Attention(config, random_layer(config, seed=ratio), _cos_sin())


class Runner:
    """Feeds steps (lists of (request, lo, hi)) through a layer with its own caches."""

    def __init__(self, config, layer, hidden, requests, seed):
        from rl_engine.models.deepseek_v4.attention import new_caches

        self.layer, self.hidden = layer, hidden
        self.ids = requests  # global request ids, in cache-row order
        self.caches = new_caches(config, len(requests), MAX_LEN, seed=seed)

    def step(self, parts):
        rows = [self.ids.index(r) for r, _, _ in parts]
        h = torch.cat([self.hidden[r][lo:hi] for r, lo, hi in parts])
        pos = torch.cat([torch.arange(lo, hi) for _, lo, hi in parts]).cuda()
        req = torch.cat([torch.full((hi - lo,), i) for i, (_, lo, hi) in enumerate(parts)]).cuda()
        # Requests in this step use cache rows `rows`: permute the tables accordingly.
        c = self.caches
        sel = torch.tensor(rows, device="cuda")
        view = type(c)(
            **{
                k: (v[sel] if k.endswith("table") and v is not None else v)
                for k, v in c.__dict__.items()
            }
        )
        seq = torch.tensor([hi for _, lo, hi in parts], dtype=torch.int32, device="cuda")
        qlen = torch.tensor([hi - lo for _, lo, hi in parts], dtype=torch.int32, device="cuda")
        return self.layer.forward(h, pos, req, seq, qlen, view)


def _same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("ratio", [4, 128, 1])
def test_layer_is_batch_invariant(ratio):
    config, layer = _layer(ratio)
    lengths = {0: 37, 1: 300, 2: 1000} if ratio != 128 else {0: 200, 1: 300, 2: 1000}
    gen = torch.Generator().manual_seed(11)
    hidden = {
        r: torch.randn(n, config.hidden, generator=gen).to(torch.bfloat16).cuda()
        for r, n in lengths.items()
    }

    full = Runner(config, layer, hidden, [0, 1, 2], seed=1).step(
        [(r, 0, n) for r, n in lengths.items()]
    )
    off = {0: 0, 1: lengths[0], 2: lengths[0] + lengths[1]}
    rows = {r: full[off[r] : off[r] + n] for r, n in lengths.items()}
    assert torch.isfinite(full.float()).all() and float(full.float().abs().mean()) > 0

    alone = Runner(config, layer, hidden, [1], seed=2).step([(1, 0, lengths[1])])
    assert _same(alone, rows[1])

    chunked = Runner(config, layer, hidden, [2], seed=3)
    a = chunked.step([(2, 0, 613)])
    b = chunked.step([(2, 613, lengths[2])])
    assert _same(torch.cat([a, b]), rows[2])

    dec = Runner(config, layer, hidden, [0, 2], seed=4)
    n0, n2 = lengths[0], lengths[2]
    p0 = dec.step([(0, 0, n0 - 3), (2, 0, n2 - 3)])
    assert _same(p0[: n0 - 3], rows[0][: n0 - 3])
    for k in range(3, 0, -1):
        out = dec.step([(0, n0 - k, n0 - k + 1), (2, n2 - k, n2 - k + 1)])
        assert _same(out[0:1], rows[0][n0 - k : n0 - k + 1]), k
        assert _same(out[1:2], rows[2][n2 - k : n2 - k + 1]), k
