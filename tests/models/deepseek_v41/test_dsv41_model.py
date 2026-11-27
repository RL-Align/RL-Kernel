# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4.1 model on the toy config (7 layers, every role, an Engram layer; real attention /
mHC dimensions, F = 256 experts, a 1024-token vocabulary) built from a random checkpoint in
the release's formats: the final hidden of a token does not depend on the batch or on
prefill / decode."""

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
    return hasattr(ext, "dsv41_mhc_pre") and hasattr(ext, "dsv41_engram_gate")


pytestmark = pytest.mark.skipif(not _ready(), reason="CUDA rl_engine._C with DSv4.1 required")

MAX_LEN = 1024
VOCAB = 1024


@pytest.fixture(scope="module")
def model():
    from rl_engine.models.deepseek_v41 import Dsv41Config, toy_config
    from rl_engine.models.deepseek_v41.loader import load_model, random_checkpoint

    cfg = Dsv41Config.from_dict({**toy_config(vocab_size=VOCAB), "moe_intermediate_size": 256})
    ckpt = random_checkpoint(cfg, seed=0)
    return load_model(ckpt, cfg, max_model_len=MAX_LEN, token_map=list(range(VOCAB)))


def _seqs(n, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, VOCAB, (MAX_LEN,), generator=g).cuda() for _ in range(n)]


def _run(model, seqs, seq_lens, query_lens, caches):
    from rl_engine.models.deepseek_v41.attention import Dsv41Batch
    from rl_engine.models.deepseek_v41.engram import ngram_windows

    batch = Dsv41Batch.from_lens(seq_lens, query_lens, model.config.sliding_window)
    pos = [torch.arange(s - q, s, device="cuda") for s, q in zip(seq_lens, query_lens)]
    ids = torch.cat([seq[p] for seq, p in zip(seqs, pos)])
    windows = ngram_windows(seqs, pos, model.config.engram_max_ngram_size)
    return model.forward_batch(ids, batch, caches, engram_windows=windows, return_streams=True)


def _same(a, b, what):
    eq = a.view(torch.int16) == b.view(torch.int16)
    assert bool(eq.all()), f"{what}: {(~eq).sum().item()} of {eq.numel()} differ"


def test_finite_and_batch_invariant(model):
    seqs = _seqs(3, 1)
    lens = [300, 77, 513]
    out, streams = _run(model, seqs, lens, lens, model.new_caches(3, seed=1))
    assert bool(torch.isfinite(out.float()).all())
    offset = 0
    for r, n in enumerate(lens):
        alone, alone_streams = _run(model, [seqs[r]], [n], [n], model.new_caches(1, seed=7 + r))
        _same(out[offset : offset + n], alone, f"request {r}")
        for i, (a, b) in enumerate(zip(streams, alone_streams)):
            _same(a[offset : offset + n], b, f"request {r} layer {i}")
        offset += n


def test_prefill_then_decode_equals_one_prefill(model):
    seqs = _seqs(1, 2)
    n, first = 280, 250
    full, _ = _run(model, seqs, [n], [n], model.new_caches(1, seed=3))
    caches = model.new_caches(1, seed=5)
    a, _ = _run(model, seqs, [130], [130], caches)
    b, _ = _run(model, seqs, [first], [first - 130], caches)
    _same(torch.cat([a, b]), full[:first], "chunked prefill")
    for p in range(first, n):
        d, _ = _run(model, seqs, [p + 1], [1], caches)
        _same(d, full[p : p + 1], f"decode position {p}")
