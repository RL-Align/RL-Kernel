# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Dsv41Config on the release config.json: vLLM's field names and topology rules, the
design doc's layer map, RoPE per layer, Engram buckets (equal to the release's table sizes),
and the fail-closed support check."""

from __future__ import annotations

import pytest

from rl_engine.models.deepseek_v41 import (
    FULL,
    REINDEX,
    REUSE,
    WINDOW,
    Dsv41Config,
    release_config,
    toy_config,
)
from rl_engine.models.deepseek_v41.config import _is_prime


def flat() -> dict:
    """The release config as one flat dict (text_config + quantization_config)."""
    r = release_config()
    return {**r["text_config"], "quantization_config": r["quantization_config"]}


def test_release_config_parses_and_is_supported():
    cfg = Dsv41Config.from_dict(release_config())
    cfg.check_supported()
    assert cfg.model_type == "deepseek_v41_text" and cfg.hidden_size == 5120
    assert cfg.num_hidden_layers == 40 and len(cfg.compress_ratios) == 43
    assert cfg.rms_norm_eps == 1e-20 and cfg.hc_eps == 1e-6
    assert cfg.quantization_config["weight_block_size"] == [32, 32]
    assert (cfg.o_lora_rank, cfg.o_groups, cfg.vocab_size) == (1024, 8, 129280)


def test_engram_primes_reproduce_the_release_table_sizes():
    cfg = Dsv41Config.from_dict(release_config())
    rows = tuple(sum(p for per in layer for p in per) for layer in cfg.engram_primes())
    assert rows == cfg.engram_num_embeddings == (384006168, 384016682)


def test_release_roles_follow_the_design_doc_map():
    cfg = Dsv41Config.from_dict(release_config())
    roles = cfg.roles()
    assert [r.mode for r in roles[:3]] == [WINDOW, WINDOW, FULL]
    assert {i for i, r in enumerate(roles) if r.mode == FULL} == {2, 8, 14, 20}
    assert {i for i, r in enumerate(roles) if r.mode == REINDEX} == {24, 28, 32, 36}
    assert roles[19] == (2, REUSE, 14, 14, False, False, None)
    assert roles[20].ratio == 1 and roles[20].candidate_source
    assert roles[25] == (1, REUSE, 20, 24, False, False, None)
    assert [i for i, r in enumerate(roles) if r.uses_candidates] == [24, 28, 32, 36]
    assert [i for i, r in enumerate(roles) if r.engram is not None] == [1, 14]
    assert all(not cfg.hash_layer(i) for i in range(40))


def test_toy_config_has_one_layer_of_every_role():
    cfg = Dsv41Config.from_dict(toy_config())
    cfg.check_supported()
    got = [(r.ratio, r.mode, r.candidate_source, r.uses_candidates) for r in cfg.roles()]
    assert got == [
        (0, WINDOW, False, False),
        (2, FULL, False, False),
        (2, REUSE, False, False),
        (1, FULL, True, False),
        (1, REUSE, False, False),
        (1, REINDEX, False, True),
        (1, REUSE, False, False),
    ]
    assert cfg.layer_role(1).engram == 0


def test_text_config_and_unknown_keys():
    d = flat()
    d["dspark_markov_rank"] = 8
    cfg = Dsv41Config.from_dict({"text_config": d, "vision_n_layers": 4})
    assert cfg.hidden_size == 5120 and cfg.kv_source_layer_ids == (2, 8, 14, 20)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"compress_ratios": [0] * 2 + [2] * 18 + [1] * 19 + [4] + [0] * 3}, "0, 1 or 2"),
        ({"kv_source_layer_ids": [2, 8, 14, 20, 21]}, "KV sources must also be index"),
        ({"index_source_layer_ids": [0, 2, 8, 14, 20, 24, 28, 32, 36]}, "index_source_layer_ids"),
        (
            {"kv_source_layer_ids": [8, 14, 20], "index_source_layer_ids": [8, 14, 20]},
            "no preceding",
        ),
        (
            {"kv_source_layer_ids": [2, 8, 14], "index_source_layer_ids": [2, 8, 14]},
            "disagree on ratio",
        ),
        ({"candidate_source_layer_id": 21}, "must be an index source"),
        ({"engram_layer_ids": [1]}, "differ in length"),
    ],
)
def test_topology_rules(change, message):
    with pytest.raises(ValueError, match=message):
        Dsv41Config.from_dict({**flat(), **change})


@pytest.mark.parametrize(
    "change,message",
    [
        (
            {
                "quantization_config": {
                    **flat()["quantization_config"],
                    "weight_block_size": [128, 128],
                }
            },
            "weight_block_size",
        ),
        ({"hidden_size": 6144}, "hidden_size"),
        (
            {
                "quantization_config": {
                    **flat()["quantization_config"],
                    "engram_embed_quant": "mxfp4",
                }
            },
            "FP8 tables",
        ),
        ({"engram_num_embeddings": [10, 10]}, "bucket rows"),
        ({"candidate_topk_blocks": 4096}, "candidate_topk_blocks"),
        (
            {"rope_parameters": {"rope_type": "yarn", "factor": 16, "attention_factor": 1.0}},
            "llama-scaling",
        ),
        ({"model_type": "deepseek_v4"}, "model_type"),
        ({"rms_norm_eps": 0.0}, "rms_norm_eps"),
    ],
)
def test_check_supported_fails_closed(change, message):
    cfg = Dsv41Config.from_dict({**flat(), **change})
    with pytest.raises(ValueError, match=message):
        cfg.check_supported()


def test_candidates_from_another_kv_group_are_rejected():
    d = toy_config()
    d.update(
        compress_ratios=[0, 1, 1, 1, 1, 1, 1],
        kv_source_layer_ids=[1, 3],
        candidate_source_layer_id=1,
        index_source_layer_ids=[1, 3, 5],
    )
    with pytest.raises(ValueError, match="another KV group"):
        Dsv41Config.from_dict(d).check_supported()


def test_rope_per_layer():
    cfg = Dsv41Config.from_dict(release_config())
    assert cfg.rope_for_layer(0) == {"theta": 10000.0, "kind": "plain"}
    for layer in (2, 20, 39):  # ratio 2 and ratio 1 are both YaRN
        r = cfg.rope_for_layer(layer)
        assert r["kind"] == "yarn" and r["theta"] == 160000.0 and r["factor"] == 16.0
    nested = {
        "main": {"rope_type": "default"},
        "compress": {"rope_type": "yarn", "factor": 4, "original_max_position_embeddings": 4096},
    }
    cfg = Dsv41Config.from_dict({**flat(), "rope_parameters": nested})
    assert cfg.rope_for_layer(0)["kind"] == "plain"
    assert cfg.rope_for_layer(20)["factor"] == 4.0
    legacy = {k: v for k, v in flat().items() if k != "rope_scaling"}
    legacy["rope_scaling"] = {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
    }
    assert Dsv41Config.from_dict(legacy).rope_for_layer(30)["kind"] == "yarn"


def test_engram_primes_are_distinct_primes_above_the_vocabulary():
    cfg = Dsv41Config.from_dict(toy_config(engram_vocab_size=1009))
    flat = [p for layer in cfg.engram_primes() for per in layer for p in per]
    assert len(flat) == 3 * 8 and len(set(flat)) == len(flat)
    assert all(_is_prime(p) and p > 1008 for p in flat)
    assert flat == sorted(flat)
    assert cfg.engram_num_embeddings == (sum(flat),)
