# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 LoRA (dsv4-lora-v1): the forward delta is batch-invariant and accurate, and the
attention layer with adapters stays batch-invariant (alone == batch == decode)."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.cuda.deepgemm import deepgemm_available
from rl_engine.kernels.ops.cuda.flashmla import flashmla_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and deepgemm_available()), reason="SM90 + DeepGEMM"
)


def _same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("rank", [8, 16, 64, 96])
@pytest.mark.parametrize("io", [(4096, 1024), (1024, 32768), (8192, 4096)])
def test_lora_apply_batch_invariant_and_accurate(rank, io):
    from rl_engine.models.deepseek_v4.lora import LoraAdapter, lora_apply

    i, o = io
    ad = LoraAdapter.init(i, o, rank, alpha=2 * rank, seed=rank, zero_b=False)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(300, i, generator=g, device="cuda").bfloat16()
    base = torch.randn(300, o, generator=g, device="cuda").bfloat16()
    y = lora_apply(base, x, ad)
    for lo, hi in ((0, 1), (5, 6), (17, 145), (299, 300)):
        assert _same(lora_apply(base[lo:hi].clone(), x[lo:hi].clone(), ad), y[lo:hi])
    h = (x.float() @ ad.a.float().t()).bfloat16().float()
    ref = base.float() + ad.scale * (h @ ad.b.float().t())
    assert float((y.float() - ref).norm() / ref.norm()) < 1e-2


@pytest.mark.skipif(not flashmla_available(), reason="pinned FlashMLA")
@pytest.mark.parametrize("ratio", [4, 1])
def test_attention_with_lora_is_batch_invariant(ratio):
    from rl_engine.models.deepseek_v4.attention import (
        Dsv4Attention,
        Dsv4AttentionConfig,
        Dsv4Batch,
        new_caches,
        random_layer,
    )
    from rl_engine.models.deepseek_v4.lora import LoraAdapter

    max_len = 4096
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2).double() / 64))
    ang = torch.arange(max_len).double()[:, None] * inv[None]
    cs = torch.cat([ang.cos(), ang.sin()], -1).float().cuda()
    config = Dsv4AttentionConfig(compress_ratio=ratio, max_model_len=max_len)
    w = random_layer(config, seed=7)
    c = config
    w.lora = {
        "wq_a": LoraAdapter.init(c.hidden, c.q_lora_rank, 16, 32, seed=1, zero_b=False),
        "wkv": LoraAdapter.init(c.hidden, c.head_dim, 16, 32, seed=2, zero_b=False),
        "wq_b": LoraAdapter.init(c.q_lora_rank, c.heads * c.head_dim, 16, 32, seed=3, zero_b=False),
        "wo_b": LoraAdapter.init(
            c.o_groups * c.o_lora_rank, c.hidden, 16, 32, seed=4, zero_b=False
        ),
    }
    attn = Dsv4Attention(config, w, cs)
    lens = [37, 300]
    g = torch.Generator(device="cuda").manual_seed(3)
    hidden = {
        r: torch.randn(n, c.hidden, generator=g, device="cuda").bfloat16()
        for r, n in enumerate(lens)
    }
    with torch.no_grad():
        full = attn.forward_batch(
            torch.cat([hidden[0], hidden[1]]),
            Dsv4Batch.from_lens(lens, lens, c.window),
            new_caches(config, 2, max_len, seed=1),
        )
        alone = attn.forward_batch(
            hidden[1],
            Dsv4Batch.from_lens([300], [300], c.window),
            new_caches(config, 1, max_len, seed=2),
        )
        assert _same(alone, full[37:])
        caches = new_caches(config, 1, max_len, seed=3)
        attn.forward_batch(hidden[1][:299], Dsv4Batch.from_lens([299], [299], c.window), caches)
        last = attn.forward_batch(
            hidden[1][299:], Dsv4Batch.from_lens([300], [1], c.window), caches
        )
        assert _same(last, full[-1:])
        w.lora = {}
        plain = attn.forward_batch(
            hidden[1],
            Dsv4Batch.from_lens([300], [300], c.window),
            new_caches(config, 1, max_len, seed=4),
        )
        assert not torch.equal(plain, alone)  # the adapters change the output
