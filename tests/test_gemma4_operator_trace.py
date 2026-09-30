# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Gemma 4 operator trace topology / kind / identity tests (CPU-only, no model)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from rl_engine.testing import gemma4_workload
from rl_engine.testing.gemma4_operator_trace import (
    FULL_ATTENTION,
    NO_BACKEND_KINDS,
    NODE_KINDS,
    OFFICIAL_FINGERPRINT,
    SLIDING_ATTENTION,
    Gemma4Spec,
    _OFFICIAL_NORM_RESIDUAL_ORDER,
    _OFFICIAL_QKV_NORM,
)
from rl_engine.testing.gemma4_workload import Gemma4Manifest, load_manifest

SLIDING_LAYER_SUFFIXES = (
    "input_layernorm",
    "q_proj",
    "q_norm",
    "rope_q",
    "k_proj",
    "v_proj",
    "k_norm",
    "rope_k",
    "v_norm",
    "attn",
    "o_proj",
    "post_attention_layernorm",
    "residual_attn",
    "pre_feedforward_layernorm",
    "gate_proj",
    "up_proj",
    "gelu_tanh_mul",
    "down_proj",
    "post_feedforward_layernorm",
    "residual_mlp",
    "layer_scalar",
)
# Full-attention layers project K and V once: kv_proj replaces the k_proj/v_proj pair.
FULL_LAYER_SUFFIXES = tuple(
    "kv_proj" if s == "k_proj" else s for s in SLIDING_LAYER_SUFFIXES if s != "v_proj"
)
FULL_LAYERS = tuple(range(5, 60, 6))

# Projection of per-layer node suffixes onto the manifest's coarse norm_residual_order.
_COARSE_BLOCK = {
    "input_layernorm": "input_layernorm",
    "post_attention_layernorm": "post_attention_layernorm",
    "residual_attn": "residual_add",
    "pre_feedforward_layernorm": "pre_feedforward_layernorm",
    "post_feedforward_layernorm": "post_feedforward_layernorm",
    "residual_mlp": "residual_add",
    **dict.fromkeys(
        (
            "q_proj",
            "q_norm",
            "rope_q",
            "k_proj",
            "v_proj",
            "kv_proj",
            "k_norm",
            "rope_k",
            "v_norm",
            "attn",
            "o_proj",
        ),
        "self_attn",
    ),
    **dict.fromkeys(("gate_proj", "up_proj", "gelu_tanh_mul", "down_proj"), "mlp"),
}


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture(scope="module")
def spec(manifest):
    return Gemma4Spec.from_manifest(manifest)


def _layer_suffixes(names: tuple[str, ...], layer: int) -> tuple[str, ...]:
    prefix = f"layers.{layer}."
    return tuple(n[len(prefix) :] for n in names if n.startswith(prefix))


def _mutated(manifest: Gemma4Manifest, mutate: Callable[[dict[str, Any]], None]) -> Gemma4Manifest:
    """Manifest copy with one edit, built directly so the loader's checks are bypassed."""
    raw = copy.deepcopy(manifest.raw)
    mutate(raw)
    return Gemma4Manifest(raw=raw, path=manifest.path)


def test_node_names_cover_full_topology(spec):
    names = spec.node_names()
    assert names[0] == "embedding"
    assert names[-4:] == ("final_layernorm", "lm_head", "logit_softcap", "logprob")
    # 1 embed + 50 sliding layers * 21 + 10 full layers * 20 + 4 tail
    assert len(names) == 1 + 50 * 21 + 10 * 20 + 4
    assert len(set(names)) == len(names)


def test_layer_types_follow_period_six_schedule(spec):
    assert len(spec.layer_types) == 60
    full = tuple(i for i, t in enumerate(spec.layer_types) if t == FULL_ATTENTION)
    assert full == FULL_LAYERS
    assert spec.layer_types.count(SLIDING_ATTENTION) == 50
    assert spec.layer_type(0) == SLIDING_ATTENTION
    assert spec.layer_type(5) == FULL_ATTENTION


def test_sliding_and_full_layers_differ_only_in_kv_projection(spec):
    names = spec.node_names()
    assert _layer_suffixes(names, 0) == SLIDING_LAYER_SUFFIXES
    assert _layer_suffixes(names, 5) == FULL_LAYER_SUFFIXES
    for layer in range(60):
        expected = FULL_LAYER_SUFFIXES if layer in FULL_LAYERS else SLIDING_LAYER_SUFFIXES
        assert _layer_suffixes(names, layer) == expected, layer
    with pytest.raises(KeyError, match="unknown node"):
        spec.node_kind("layers.0.nonexistent")
    with pytest.raises(KeyError, match="unknown node"):
        spec.node_kind("layers.0.attn.extra")


def test_full_attention_layers_project_k_and_v_once(spec):
    """K=V: one projection node feeds k_norm (then RoPE) and v_norm; there is no v_proj."""
    names = spec.node_names()
    for layer in range(60):
        prefix = f"layers.{layer}."
        suffixes = _layer_suffixes(names, layer)
        if layer in FULL_LAYERS:
            assert spec.node_kind(prefix + "kv_proj") == "det_gemm", layer
            assert prefix + "k_proj" not in names and prefix + "v_proj" not in names, layer
            source = "kv_proj"
        else:
            assert spec.node_kind(prefix + "k_proj") == "det_gemm", layer
            assert spec.node_kind(prefix + "v_proj") == "det_gemm", layer
            assert prefix + "kv_proj" not in names, layer
            source = "v_proj"
        # The V branch reads the raw projection; k_norm and rope_k act on the K branch only.
        order = [suffixes.index(s) for s in (source, "k_norm", "rope_k", "v_norm", "attn")]
        assert order == sorted(order), layer
    with pytest.raises(KeyError, match="as kv_proj"):
        spec.node_kind("layers.5.k_proj")
    with pytest.raises(KeyError, match="as kv_proj"):
        spec.node_kind("layers.5.v_proj")
    with pytest.raises(KeyError, match="separate k_proj and v_proj"):
        spec.node_kind("layers.0.kv_proj")


def test_every_node_has_a_kind_and_every_kind_is_used(spec):
    assert len(NODE_KINDS) == 12
    assert len(NO_BACKEND_KINDS) == 2
    assert not set(NODE_KINDS) & set(NO_BACKEND_KINDS)
    names = spec.node_names()
    kinds = {name: spec.node_kind(name) for name in names}
    assert set(kinds.values()) == set(NODE_KINDS) | set(NO_BACKEND_KINDS)
    assert "loss" not in names
    assert "masked_loss" not in kinds.values()
    assert kinds["embedding"] == "scaled_embedding"
    assert kinds["lm_head"] == "tied_lm_head"
    assert kinds["layers.0.attn"] == "attention_sliding"
    assert kinds["layers.0.rope_q"] == "rope_sliding"
    assert kinds["layers.5.attn"] == "attention_global"
    assert kinds["layers.5.rope_k"] == "rope_global"
    assert kinds["layers.0.v_norm"] == "qkv_norm"
    assert kinds["layers.0.post_attention_layernorm"] == "rms_norm"
    assert kinds["layers.0.layer_scalar"] == "layer_scale"
    assert kinds["layers.0.residual_attn"] == "residual_add"


def test_layer_order_matches_manifest_norm_residual_order(spec, manifest):
    coarse = manifest.model_identity["config_fingerprint"]["norm_residual_order"]
    names = spec.node_names()
    for layer in range(60):
        suffixes = _layer_suffixes(names, layer)
        assert suffixes[-1] == "layer_scalar", layer
        projected: list[str] = []
        for suffix in suffixes[:-1]:
            block = _COARSE_BLOCK[suffix]
            if not projected or projected[-1] != block:
                projected.append(block)
        assert projected == coarse, layer


def test_from_manifest_forbids_architecture_drift(manifest):
    def shrink_layers(raw):
        raw["model_identity"]["config_fingerprint"]["text_config"]["num_hidden_layers"] = 2

    def break_schedule(raw):
        text = raw["model_identity"]["config_fingerprint"]["text_config"]
        text["layer_types"][5] = SLIDING_ATTENTION

    def scale_v_norm(raw):
        fp = raw["model_identity"]["config_fingerprint"]
        fp["attention_qkv_norm"]["v"]["with_scale"] = True

    def drop_post_attention_norm(raw):
        raw["model_identity"]["config_fingerprint"]["norm_residual_order"].remove(
            "post_attention_layernorm"
        )

    for mutate in (shrink_layers, break_schedule, scale_v_norm, drop_post_attention_norm):
        with pytest.raises(ValueError, match="architecture shrink"):
            Gemma4Spec.from_manifest(_mutated(manifest, mutate))


def test_pins_agree_with_fingerprint_module():
    for key, expected in gemma4_workload._OFFICIAL_FINGERPRINT.items():
        assert OFFICIAL_FINGERPRINT[key] == expected, key
    assert _OFFICIAL_QKV_NORM == gemma4_workload._OFFICIAL_QKV_NORM
    assert _OFFICIAL_NORM_RESIDUAL_ORDER == gemma4_workload._OFFICIAL_NORM_RESIDUAL_ORDER


def test_spec_fields_match_manifest(spec, manifest):
    ident = manifest.model_identity
    text = ident["config_fingerprint"]["text_config"]
    for key, expected in OFFICIAL_FINGERPRINT.items():
        if hasattr(spec, key):
            assert getattr(spec, key) == expected, key
    assert spec.rope_parameters == text["rope_parameters"]
    assert spec.rope_parameters[SLIDING_ATTENTION]["rope_type"] == "default"
    assert spec.rope_parameters[FULL_ATTENTION]["rope_type"] == "proportional"
    assert spec.workload_id == manifest.workload_id
    assert spec.model_id == ident["model_id"]
    assert spec.revision == ident["revision"]
    assert spec.weight_content_hash == ident["weight_snapshot"]["content_hash"]
    assert spec.has_v_proj(SLIDING_ATTENTION) is True
    assert spec.has_v_proj(FULL_ATTENTION) is False
