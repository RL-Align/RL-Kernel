# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4.1 Engram hashing against DeepSeek's reference ``inference/engram.py``: the compressed
token map of the release tokenizer (size = engram_compressed_vocab_size) and the n-gram
hashes over a prefill + decode split. Needs the release metadata directory
(``DSV41_META``: config.json, tokenizer.json, inference/engram.py)."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import types

import pytest
import torch

META = pathlib.Path(
    os.environ.get("DSV41_META", "/work/u4478697/dsv4-lora-e2e/models/DeepSeek-V4.1-Flash-meta")
)
pytestmark = pytest.mark.skipif(
    not (META / "tokenizer.json").exists() or not (META / "inference/engram.py").exists(),
    reason="DSv4.1 release metadata (DSV41_META) not available",
)


@pytest.fixture(scope="module")
def ref():
    spec = importlib.util.spec_from_file_location("dsv41_ref_engram", META / "inference/engram.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def maps(ref):
    from tokenizers import Tokenizer

    from rl_engine.models.deepseek_v41.engram import build_compressed_token_map

    backend = Tokenizer.from_file(str(META / "tokenizer.json"))
    vocab = backend.get_vocab_size(with_added_tokens=True)
    tok = type("Tok", (), {"backend_tokenizer": backend, "__len__": lambda self: vocab})()
    return (
        build_compressed_token_map(str(META / "tokenizer.json")),
        ref.build_compressed_token_map(tok),
        tok,
    )


def test_token_map_equals_the_reference(maps):
    from rl_engine.models.deepseek_v41 import Dsv41Config, release_config

    (ours, size), (theirs, ref_size), _ = maps
    assert ours == theirs and size == ref_size
    assert size == Dsv41Config.from_dict(release_config()).engram_compressed_vocab_size


def test_hashes_equal_the_reference_over_prefill_and_decode(ref, maps):
    from rl_engine.models.deepseek_v41 import Dsv41Config, release_config
    from rl_engine.models.deepseek_v41.engram import NgramHasher, ngram_windows

    (token_map, _), _, tok = maps
    cfg = Dsv41Config.from_dict(release_config())
    args = types.SimpleNamespace(
        engram_layer_ids=cfg.engram_layer_ids,
        engram_max_ngram_size=cfg.engram_max_ngram_size,
        engram_n_heads=cfg.engram_n_heads,
        engram_vocab_size=cfg.engram_vocab_size,
        engram_num_embeddings=cfg.engram_num_embeddings,
        engram_head_dim=cfg.engram_head_dim,
        engram_compressed_vocab_size=cfg.engram_compressed_vocab_size,
        engram_pad_id=cfg.engram_pad_token_id,
        max_batch_size=2,
        max_seq_len=64,
    )
    layout = ref.EngramLayout.from_args(args)
    state = ref.NgramHashState(args, layout, tok)
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (2, 40), generator=gen)
    want = torch.cat(
        [state(ids[:, :33], 0)] + [state(ids[:, p : p + 1], p) for p in range(33, 40)], 1
    )

    hasher = NgramHasher(cfg, token_map, device="cpu")
    pos = torch.arange(40)
    got = hasher(ngram_windows([ids[0], ids[1]], [pos, pos], cfg.engram_max_ngram_size))
    assert torch.equal(got.view(2, 40, *got.shape[1:]), want)
