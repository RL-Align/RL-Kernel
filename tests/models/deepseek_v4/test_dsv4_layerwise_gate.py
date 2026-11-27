# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Layerwise exact-zero gate: rollout (continuous batching, chunked prefill, decode) ==
actor recompute (packed prefill, other order) == training forward, at every layer.

The model mixes every layer type: window-only, C4 with the indexer and C128 attention;
hash and scored routing; nonzero LoRA adapters on attention and MoE.
"""

from __future__ import annotations

import pytest
import torch

from tests.models.deepseek_v4.test_dsv4_model import MAX_LEN, _config, _ready

pytestmark = pytest.mark.skipif(
    not _ready(), reason="SM90, DeepGEMM, FlashMLA and a KERNEL_ALIGN_MOE_SM90=1 build"
)

PROMPTS = [300, 140, 517, 64]
GENERATE = [6, 9, 4, 12]


@pytest.fixture(scope="module")
def model():
    from rl_engine.models.deepseek_v4.loader import load_model, random_checkpoint
    from rl_engine.models.deepseek_v4.model import attach_lora

    cfg = _config(num_hash_layers=2)
    m = load_model(random_checkpoint(cfg, seed=0), cfg, max_model_len=MAX_LEN)
    attach_lora(m, rank=16, alpha=32.0, zero_b=False, seed=3)
    return m


@pytest.fixture(scope="module")
def ids(model):
    gen = torch.Generator().manual_seed(21)
    vocab = model.config.vocab_size
    return [torch.randint(vocab, (p + g,), generator=gen).cuda() for p, g in zip(PROMPTS, GENERATE)]


def test_rollout_schedule_mixes_prefill_and_decode():
    from rl_engine.models.deepseek_v4.layerwise_gate import rollout_schedule

    steps = rollout_schedule(PROMPTS, GENERATE, token_budget=128)
    lengths = [p + g for p, g in zip(PROMPTS, GENERATE)]
    covered = [0] * len(lengths)
    mixed = False
    for parts in steps:
        assert sum(hi - lo for _, lo, hi in parts) <= 128 + len(lengths)
        assert [r for r, _, _ in parts] == sorted({r for r, _, _ in parts})
        for r, lo, hi in parts:
            assert lo == covered[r] and hi > lo
            covered[r] = hi
        q = [hi - lo for _, lo, hi in parts]
        mixed |= 1 in q and max(q) > 1
    assert covered == lengths
    assert mixed  # some step decodes next to a prefill chunk


@pytest.mark.parametrize("budget", [128, 61])
def test_layerwise_gate_passes(model, ids, budget):
    from rl_engine.models.deepseek_v4.layerwise_gate import layerwise_gate

    report = layerwise_gate(model, ids, PROMPTS, token_budget=budget)
    assert report["passed"], report
    assert len(report["actor"]["layers"]) == len(model.layers)


def test_layerwise_gate_finds_the_first_bad_layer(model, ids):
    """Negative control: a one-element change in layer 2's FFN norm after the rollout
    shows up at layer 2, with layers 0 and 1 still byte-identical."""
    from rl_engine.models.deepseek_v4.layerwise_gate import compare, run_packed, run_rollout

    rollout = run_rollout(model, ids, PROMPTS, token_budget=128)
    norm = model.layers[2].hc_ffn.norm
    saved = norm.clone()
    norm[7] = norm[7] * 1.5
    try:
        rep = compare(rollout, run_packed(model, ids))
    finally:
        norm.copy_(saved)
    assert rep["first_layer_differs"] == 2, rep
    assert [r["tokens_differ"] for r in rep["layers"][:2]] == [0, 0]
    assert rep["layers"][2]["tokens_differ"] > 0
    assert not rep["passed"]
