# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Gemma 4 architecture fingerprint (#415) manifest tests (CPU-only)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SCRIPT = REPO_ROOT / "scripts" / "gemma4_reference.py"


def _load_pure_workload_module():
    path = REPO_ROOT / "rl_engine/testing/gemma4_workload.py"
    spec = importlib.util.spec_from_file_location("_gemma4_workload_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gemma4 = _load_pure_workload_module()
Gemma4WorkloadError = gemma4.Gemma4WorkloadError
default_manifest_path = gemma4.default_manifest_path
load_manifest = gemma4.load_manifest
reference_payload = gemma4.reference_payload
validate_manifest = gemma4.validate_manifest


def _raw_manifest():
    return json.loads(default_manifest_path().read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def test_default_manifest_path_exists():
    path = default_manifest_path()
    assert path.is_file()
    assert path.name == "gemma4_manifest.json"


def test_manifest_loads_and_validates(manifest):
    assert manifest.workload_id.startswith("gemma4-31b-it-dense")
    assert manifest.version == "gemma4-arch-fingerprint-v1"
    validate_manifest(manifest.raw)


def test_model_identity_is_full_gemma4_31b(manifest):
    identity = manifest.model_identity
    assert identity["model_id"] == "google/gemma-4-31B-it"
    assert identity["revision"] == "842da3794eaa0b77d5f08bae87a17459d91ff475"
    assert identity["semantics_source"] == {
        "package": "transformers",
        "version": "5.13.1",
        "git_commit": "4626421dc6b741a329300682a6408246ee465490",
        "modeling_file": "src/transformers/models/gemma4/modeling_gemma4.py",
    }
    assert identity["architecture"] == "Gemma4ForConditionalGeneration"
    assert identity["exit_forbids_architecture_shrink"] is True
    fp = identity["config_fingerprint"]
    assert fp["dtype"] == "bfloat16"
    assert fp["audio_config"] is None
    assert fp["tie_word_embeddings"] is True
    text = fp["text_config"]
    assert text["num_hidden_layers"] == 60
    assert text["hidden_size"] == 5376
    assert text["intermediate_size"] == 21504
    assert text["num_attention_heads"] == 32
    assert text["num_key_value_heads"] == 16
    assert text["head_dim"] == 256
    assert text["num_global_key_value_heads"] == 4
    assert text["global_head_dim"] == 512
    assert text["sliding_window"] == 1024
    assert text["vocab_size"] == 262144
    assert text["max_position_embeddings"] == 262144
    assert text["hidden_activation"] == "gelu_pytorch_tanh"
    assert text["final_logit_softcapping"] == 30.0
    assert text["tie_word_embeddings"] is True
    assert text["attention_k_eq_v"] is True
    assert text["use_bidirectional_attention"] == "vision"
    assert text["enable_moe_block"] is False
    assert text["hidden_size_per_layer_input"] == 0
    assert text["num_kv_shared_layers"] == 0
    weight = identity["weight_snapshot"]
    assert weight["total_size_bytes"] > 0
    assert weight["pin_method"]
    assert len(weight["shards"]) == 2
    assert weight["content_hash"] == (
        "e02b23eea7a4b409b4e4034304aa6a0a1ce1528df80ba12ba7664ccc88d6f302"
    )


def test_layer_types_follow_period_six_schedule(manifest):
    layer_types = manifest.model_identity["config_fingerprint"]["text_config"]["layer_types"]
    assert len(layer_types) == 60
    full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    assert full == list(range(5, 60, 6))
    assert layer_types.count("sliding_attention") == 50
    assert layer_types.count("full_attention") == 10


def test_rope_parameters_pinned_per_layer_type(manifest):
    rope = manifest.model_identity["config_fingerprint"]["text_config"]["rope_parameters"]
    assert rope["sliding_attention"] == {"rope_theta": 10000.0, "rope_type": "default"}
    assert rope["full_attention"] == {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
        "rope_type": "proportional",
    }


def test_attention_qkv_norm_pinned(manifest):
    fp = manifest.model_identity["config_fingerprint"]
    assert fp["attention_qkv_norm"] == {
        "q": {"norm": "rmsnorm", "with_scale": True, "rope": True},
        "k": {"norm": "rmsnorm", "with_scale": True, "rope": True},
        "v": {"norm": "rmsnorm", "with_scale": False, "rope": False},
    }
    assert fp["attention_qkv_norm_note"]


def test_norm_residual_order_pinned(manifest):
    fp = manifest.model_identity["config_fingerprint"]
    assert fp["norm_residual_order"] == [
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "residual_add",
        "pre_feedforward_layernorm",
        "mlp",
        "post_feedforward_layernorm",
        "residual_add",
    ]
    assert fp["norm_residual_order_note"]


def test_manifest_edit_without_identity_regen_rejected():
    raw = _raw_manifest()
    raw["model_identity"]["config_fingerprint"]["text_config"]["rms_norm_eps"] = 1e-5
    with pytest.raises(Gemma4WorkloadError, match="fixture_identity_sha256"):
        validate_manifest(raw)


def test_missing_semantics_source_rejected():
    raw = _raw_manifest()
    del raw["model_identity"]["semantics_source"]["git_commit"]
    with pytest.raises(Gemma4WorkloadError, match="semantics_source missing 'git_commit'"):
        validate_manifest(raw)


def test_v_norm_scale_cannot_be_enabled():
    raw = _raw_manifest()
    raw["model_identity"]["config_fingerprint"]["attention_qkv_norm"]["v"]["with_scale"] = True
    with pytest.raises(Gemma4WorkloadError, match="attention_qkv_norm"):
        validate_manifest(raw)


def test_norm_residual_order_deviation_rejected():
    raw = _raw_manifest()
    del raw["model_identity"]["config_fingerprint"]["norm_residual_order"][2]
    with pytest.raises(Gemma4WorkloadError, match="norm_residual_order"):
        validate_manifest(raw)


def test_missing_weight_hash_rejected():
    raw = _raw_manifest()
    del raw["model_identity"]["weight_snapshot"]["content_hash"]
    with pytest.raises(Gemma4WorkloadError, match="content_hash"):
        validate_manifest(raw)


def test_architecture_shrink_rejected():
    raw = _raw_manifest()
    raw["model_identity"]["config_fingerprint"]["text_config"]["num_hidden_layers"] = 2
    with pytest.raises(Gemma4WorkloadError, match="does not match official"):
        validate_manifest(raw)


def test_k_eq_v_cannot_be_disabled():
    raw = _raw_manifest()
    raw["model_identity"]["config_fingerprint"]["text_config"]["attention_k_eq_v"] = False
    with pytest.raises(Gemma4WorkloadError, match="attention_k_eq_v"):
        validate_manifest(raw)


def test_layer_types_shrink_rejected():
    raw = _raw_manifest()
    text = raw["model_identity"]["config_fingerprint"]["text_config"]
    text["layer_types"] = text["layer_types"][:59]
    with pytest.raises(Gemma4WorkloadError, match="layer_types must list exactly"):
        validate_manifest(raw)


def test_layer_schedule_deviation_rejected():
    raw = _raw_manifest()
    text = raw["model_identity"]["config_fingerprint"]["text_config"]
    text["layer_types"][0] = "full_attention"
    with pytest.raises(Gemma4WorkloadError, match="5x sliding"):
        validate_manifest(raw)


def test_reference_payload_contains_required_fields(manifest):
    payload = reference_payload(manifest)
    assert payload["workload_id"] == manifest.workload_id
    assert payload["version"] == manifest.version
    assert payload["fixture_identity_sha256"] == manifest.raw["fixture_identity_sha256"]
    assert payload["model_id"] == "google/gemma-4-31B-it"
    assert payload["revision"] == manifest.model_identity["revision"]
    assert payload["semantics_source"] == manifest.model_identity["semantics_source"]
    assert payload["config_fingerprint"] == manifest.model_identity["config_fingerprint"]
    assert payload["weight_snapshot"] == manifest.model_identity["weight_snapshot"]


def test_gemma4_reference_cli_emits_identity():
    proc = subprocess.run(
        [sys.executable, str(REFERENCE_SCRIPT), "--emit-json", "-"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["workload_id"].startswith("gemma4-31b-it-dense")
    assert payload["revision"] == "842da3794eaa0b77d5f08bae87a17459d91ff475"
    assert len(payload["fixture_identity_sha256"]) == 64
