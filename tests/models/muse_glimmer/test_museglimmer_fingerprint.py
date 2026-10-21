# SPDX-License-Identifier: Apache-2.0
"""Muse-Glimmer-30B architecture fingerprint, strict loader and operator trace (RFC #501)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from rl_engine.models.muse_glimmer import workload as w
from rl_engine.models.muse_glimmer.loader import (
    check_runtime_binding,
    read_safetensors_header,
    verify_checkpoint,
    verify_config,
    verify_weight_map,
)
from rl_engine.models.muse_glimmer.operator_trace import (
    NODE_KIND_ROWS,
    RFC_WS1_ROWS,
    MuseGlimmerSpec,
)

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture()
def raw():
    return copy.deepcopy(w.load_manifest().raw)


def _resealed(raw):
    raw["fixture_identity_sha256"] = w.manifest_identity_hash(raw)
    return raw


# --- manifest -------------------------------------------------------------------------


def test_manifest_loads_and_pins_revision():
    m = w.load_manifest()
    ident = m.model_identity
    assert ident["model_id"] == "meta-models/Muse-Glimmer-30B"
    assert ident["revision"] == "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
    assert ident["architecture"] == "MuseGlimmerForConditionalGeneration"
    assert ident["semantics_source"]["git_commit"] == "856157a2f3e9594954310df18fdccc31ffddebe9"
    assert m.raw["fixture_identity_sha256"] == w.manifest_identity_hash(m.raw)


def test_layer_schedule_is_three_local_one_global():
    text = w.load_manifest().text_config
    full = [i for i, t in enumerate(text["layer_types"]) if t == w.FULL_ATTENTION]
    assert full == list(range(3, 52, 4))
    assert len(full) == 13
    assert all(text["layer_rope_theta"][i] == 0 for i in full)


def test_padded_vocab_and_row_parallel_shards():
    assert w.padded_vocab_size(202048) == 202752
    assert 202048 % (128 * 8) != 0
    shards = w.OFFICIAL_PARALLEL_LAYOUT["row_parallel_shards"]
    assert shards["o_proj"] * 8 == 32 * 128
    assert shards["down_proj"] * 8 == 19968


def test_weight_schema_counts():
    schema = w.text_weight_schema()
    assert len(schema) == 52 * 12 + 3
    assert schema["model.language_model.layers.0.self_attn.k_proj.weight"] == (256, 6656)
    assert schema["model.language_model.layers.51.self_attn.gate_proj.weight"] == (4096, 6656)
    assert schema["lm_head.weight"] == (202048, 6656)
    assert w.load_manifest().model_identity["weight_snapshot"]["text_tensor_count"] == len(schema)


def test_identity_hash_detects_unsealed_edit(raw):
    raw["model_identity"]["config_fingerprint"]["text_config"]["max_position_embeddings"] = 131072
    raw["workload_id"] = "edited"
    with pytest.raises(w.MuseGlimmerWorkloadError, match="fixture_identity_sha256"):
        w.validate_manifest(raw)


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_hidden_layers", 48),
        ("num_key_value_heads", 8),
        ("sliding_window", 4096),
        ("qk_scale_factor", 43.7840518911),
        ("post_norm_eps", 1e-5),
        ("output_multiplier", 1.0),
        ("final_logit_softcapping", 30.0),
        ("tie_word_embeddings", True),
    ],
)
def test_text_config_drift_rejected(raw, key, value):
    raw["model_identity"]["config_fingerprint"]["text_config"][key] = value
    with pytest.raises(w.MuseGlimmerWorkloadError, match=key):
        w.validate_manifest(_resealed(raw))


def test_layer_schedule_drift_rejected(raw):
    text = raw["model_identity"]["config_fingerprint"]["text_config"]
    text["layer_types"][3] = w.SLIDING_ATTENTION
    with pytest.raises(w.MuseGlimmerWorkloadError, match="schedule"):
        w.validate_manifest(_resealed(raw))


def test_rope_on_global_layer_rejected(raw):
    text = raw["model_identity"]["config_fingerprint"]["text_config"]
    text["layer_rope_theta"][3] = 500000.0
    with pytest.raises(w.MuseGlimmerWorkloadError, match="NoPE"):
        w.validate_manifest(_resealed(raw))


@pytest.mark.parametrize(
    "section,path,value",
    [
        ("frozen_semantics", ("logits", "order"), ["lm_head", "tanh"]),
        ("frozen_semantics", ("output_gate", "granularity"), "per_head"),
        ("frozen_semantics", ("embedding", "scale_by_sqrt_hidden"), True),
        ("projection_grouping", ("attention",), [["q_proj", "k_proj", "v_proj"], ["gate_proj"]]),
        ("parallel_layout", ("padded_vocab_size",), 202048),
    ],
)
def test_frozen_sections_rejected_on_drift(raw, section, path, value):
    target = raw[section]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(w.MuseGlimmerWorkloadError, match=section):
        w.validate_manifest(_resealed(raw))


def test_weight_snapshot_tamper_rejected(raw):
    raw["model_identity"]["weight_snapshot"]["shards"][0]["sha256"] = "0" * 64
    with pytest.raises(w.MuseGlimmerWorkloadError, match="content_hash"):
        w.validate_manifest(_resealed(raw))


# --- strict loader --------------------------------------------------------------------


def _write_header_only_shard(
    path: Path, tensors: dict[str, tuple[int, ...]], dtype: str = "BF16", data: bytes = b""
):
    header = {
        name: {"dtype": dtype, "shape": list(shape), "data_offsets": [0, len(data)]}
        for name, shape in tensors.items()
    }
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


@pytest.fixture()
def fake_checkpoint(tmp_path):
    m = w.load_manifest()
    schema = w.text_weight_schema()
    names = sorted(schema)
    shard_files = [s["filename"] for s in m.model_identity["weight_snapshot"]["shards"]]
    split = len(names) // 2
    parts = {shard_files[0]: names[:split], shard_files[1]: names[split:]}
    for filename, part in parts.items():
        _write_header_only_shard(tmp_path / filename, {n: schema[n] for n in part})
    weight_map = {n: f for f, part in parts.items() for n in part}
    weight_map["model.vision_tower.ln_pre.weight"] = shard_files[1]
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    config = {
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "text_config": m.text_config,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_loader_accepts_matching_checkpoint(fake_checkpoint):
    report = verify_checkpoint(fake_checkpoint, require_pinned_bytes=False)
    assert report.text_tensor_count == 627
    assert len(report.headers_checked) == 2


def test_loader_requires_pinned_bytes(fake_checkpoint):
    with pytest.raises(w.MuseGlimmerWorkloadError, match="config.json sha256"):
        verify_checkpoint(fake_checkpoint)


def test_loader_rejects_wrong_tensor_shape(fake_checkpoint):
    # Keep every other tensor of the shard so verify_checkpoint reaches the shape check.
    target = "model.language_model.layers.0.self_attn.k_proj.weight"
    index = json.loads((fake_checkpoint / "model.safetensors.index.json").read_text())
    shard_name = index["weight_map"][target]
    schema = w.text_weight_schema()
    tensors = {
        n: schema[n] for n, f in index["weight_map"].items() if f == shard_name and n in schema
    }
    tensors[target] = (1024, 6656)
    _write_header_only_shard(fake_checkpoint / shard_name, tensors)
    with pytest.raises(w.MuseGlimmerWorkloadError, match="!= pinned"):
        verify_checkpoint(fake_checkpoint, require_pinned_bytes=False)


def test_loader_rejects_truncated_shard_without_sha256(fake_checkpoint, raw):
    # Regression: a shard cut short behind an intact header must fail closed by default,
    # without opting into SHA-256. Pin the fake files' hashes and sizes, then truncate.
    index = json.loads((fake_checkpoint / "model.safetensors.index.json").read_text())
    schema = w.text_weight_schema()
    snapshot = raw["model_identity"]["weight_snapshot"]
    shard = fake_checkpoint / snapshot["shards"][0]["filename"]
    tensors = {
        n: schema[n] for n, f in index["weight_map"].items() if f == shard.name and n in schema
    }
    _write_header_only_shard(shard, tensors, data=b"\0" * 64)
    raw["model_identity"]["config_sha256"] = hashlib.sha256(
        (fake_checkpoint / "config.json").read_bytes()
    ).hexdigest()
    snapshot["index_sha256"] = hashlib.sha256(
        (fake_checkpoint / "model.safetensors.index.json").read_bytes()
    ).hexdigest()
    for record in snapshot["shards"]:
        record["size_bytes"] = (fake_checkpoint / record["filename"]).stat().st_size
    snapshot["weight_files_total_size_bytes"] = sum(r["size_bytes"] for r in snapshot["shards"])
    snapshot["content_hash"] = w.weight_snapshot_hash(snapshot["shards"])
    w.validate_manifest(_resealed(raw))
    m = w.MuseGlimmerManifest(raw=raw)
    verify_checkpoint(fake_checkpoint, m)
    shard.write_bytes(shard.read_bytes()[:-16])
    assert read_safetensors_header(shard)  # the header itself still parses
    with pytest.raises(w.MuseGlimmerWorkloadError, match="size differs from the pin"):
        verify_checkpoint(fake_checkpoint, m, verify_shard_sha256=False)


@pytest.mark.parametrize(
    "payload, match",
    [
        (struct.pack("<Q", 1 << 40), "header length"),
        (struct.pack("<Q", 64) + b"{}", "truncated"),
        (struct.pack("<Q", 3) + b"{x}", "invalid safetensors header"),
        (struct.pack("<Q", 2) + b"[]", "JSON object"),
    ],
)
def test_read_safetensors_header_fails_closed(tmp_path, payload, match):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(payload)
    with pytest.raises(w.MuseGlimmerWorkloadError, match=match):
        read_safetensors_header(path)


def test_loader_rejects_config_drift():
    m = w.load_manifest()
    text = dict(m.text_config, use_cache=False)
    with pytest.raises(w.MuseGlimmerWorkloadError, match="use_cache"):
        verify_config({"architectures": [m.model_identity["architecture"]], "text_config": text}, m)
    with pytest.raises(w.MuseGlimmerWorkloadError, match="architectures"):
        verify_config(
            {"architectures": ["TransformersForCausalLM"], "text_config": m.text_config}, m
        )


def test_loader_rejects_fused_projection_and_unpinned_shard():
    m = w.load_manifest()
    shard = m.model_identity["weight_snapshot"]["shards"][0]["filename"]
    weight_map = {n: shard for n in w.text_weight_schema()}
    fused = dict(weight_map)
    fused["model.language_model.layers.0.self_attn.qkv_proj.weight"] = shard
    with pytest.raises(w.MuseGlimmerWorkloadError, match="fused projection"):
        verify_weight_map(fused, m)
    missing = dict(weight_map)
    del missing["model.language_model.layers.7.self_attn.gate_proj.weight"]
    with pytest.raises(w.MuseGlimmerWorkloadError, match="missing"):
        verify_weight_map(missing, m)
    unpinned = dict(weight_map, **{"lm_head.weight": "model-00003-of-00003.safetensors"})
    with pytest.raises(w.MuseGlimmerWorkloadError, match="unpinned"):
        verify_weight_map(unpinned, m)


def test_runtime_binding_fails_closed():
    grouping = w.OFFICIAL_PROJECTION_GROUPING["attention"]
    check_runtime_binding(
        model_class="MuseGlimmerForConditionalGeneration",
        attention_projection_grouping=grouping,
        speculative_decoding=False,
    )
    with pytest.raises(w.MuseGlimmerWorkloadError, match="Transformers-fallback"):
        check_runtime_binding(
            model_class="TransformersForCausalLM",
            attention_projection_grouping=grouping,
            speculative_decoding=False,
        )
    with pytest.raises(w.MuseGlimmerWorkloadError, match="grouping"):
        check_runtime_binding(
            model_class="MuseGlimmerForConditionalGeneration",
            attention_projection_grouping=[["q_proj", "k_proj", "v_proj"], ["gate_proj"]],
            speculative_decoding=False,
        )
    with pytest.raises(w.MuseGlimmerWorkloadError, match="speculative"):
        check_runtime_binding(
            model_class="MuseGlimmerForConditionalGeneration",
            attention_projection_grouping=grouping,
            speculative_decoding=True,
        )


@pytest.mark.skipif(
    not os.environ.get("MUSEGLIMMER_CHECKPOINT"), reason="set MUSEGLIMMER_CHECKPOINT"
)
def test_loader_on_real_checkpoint():
    report = verify_checkpoint(
        os.environ["MUSEGLIMMER_CHECKPOINT"],
        verify_shard_sha256=os.environ.get("MUSEGLIMMER_VERIFY_SHA256") == "1",
    )
    assert report.text_tensor_count == 627


# --- operator trace -------------------------------------------------------------------


def test_trace_covers_every_ws1_row():
    spec = MuseGlimmerSpec.from_manifest()
    assert set(NODE_KIND_ROWS.values()) == RFC_WS1_ROWS
    assert "fused_add_rmsnorm" not in RFC_WS1_ROWS  # renamed in the RFC revision
    assert "local_gated_attention_train" not in RFC_WS1_ROWS  # merged with the rollout row
    assert {node.row for node in spec.trace()} == RFC_WS1_ROWS


def test_trace_node_counts():
    spec = MuseGlimmerSpec.from_manifest()
    nodes = spec.trace()
    assert len(nodes) == 5 + 39 * 21 + 13 * 19
    assert nodes[0].name == "embedding"
    assert [n.name for n in nodes[-4:]] == ["final_norm", "lm_head", "logit_softcap", "logprob"]
    kinds = Counter(n.kind for n in nodes)
    assert kinds["det_gemm"] == 52 * 8
    assert kinds["rope_local"] == 39 * 2
    assert kinds["attention_local"] == 39
    assert kinds["attention_global"] == 13
    assert kinds["centered_rms_norm"] == 52 * 4


def test_global_layers_have_no_rope_and_gate_follows_layer_type():
    spec = MuseGlimmerSpec.from_manifest()
    assert "layers.3.rope_q" not in spec.node_names()
    assert spec.node_kind("layers.2.rope_q") == "rope_local"
    assert spec.node_kind("layers.3.attn") == "attention_global"
    assert spec.node_kind("layers.3.output_gate") == "output_gate_global"
    assert spec.node_kind("layers.4.output_gate") == "output_gate_local"
    assert spec.node_kind("lm_head") == "lm_head"
    for bad in ("layers.3.rope_k", "layers.52.attn", "layers.0.qkv_proj", "nope"):
        with pytest.raises(KeyError):
            spec.node_kind(bad)


def test_gemm_shapes_match_weight_schema():
    spec = MuseGlimmerSpec.from_manifest()
    schema = w.text_weight_schema()
    names = {
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "attn_gate_proj": "self_attn.gate_proj",
        "o_proj": "self_attn.o_proj",
        "mlp_gate_proj": "mlp.gate_proj",
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj",
    }
    for node, tensor in names.items():
        assert spec.gemm_shapes()[node] == schema[f"model.language_model.layers.0.{tensor}.weight"]


def test_reference_cli_emits_json():
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/museglimmer_reference.py"), "--emit-json", "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(out.stdout)
    assert payload["revision"] == "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
    assert payload["parallel_layout"]["padded_vocab_size"] == 202752
