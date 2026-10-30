# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Muse Glimmer architecture fingerprint.

This module freezes the meta-models/Muse-Glimmer-30B text-model identity used by every
later Muse Glimmer gate (RFC #501). It pins the checkpoint revision, the published config,
the semantics the config does not state, the projection grouping, and the canonical TP /
padded vocabulary. It does not run the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from rl_engine.config.workload import manifest_identity_hash, weight_snapshot_hash

_MANIFEST_PATH = Path(__file__).with_name("museglimmer_manifest.json")

_REQUIRED_TOP_LEVEL = (
    "version",
    "workload_id",
    "model_identity",
    "frozen_semantics",
    "projection_grouping",
    "parallel_layout",
    "fixture_identity_sha256",
)

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"
LAYER_PERIOD = 4
FULL_SLOT = 3  # layer i is full_attention iff i % LAYER_PERIOD == FULL_SLOT

OFFICIAL_FINGERPRINT: dict[str, Any] = {
    "num_hidden_layers": 52,
    "hidden_size": 6656,
    "intermediate_size": 19968,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "sliding_window": 2048,
    "vocab_size": 202048,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-05,
    "post_norm_eps": 1e-08,
    "qk_scale_factor": 3.87,
    "output_multiplier": 0.19611613513818404,
    "final_logit_softcapping": 20.0,
    "hidden_activation": "silu",
    "attention_bias": False,
    "tie_word_embeddings": False,
    "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
}

# Semantics read from modeling_muse_glimmer.py at semantics_source. None of these are HF
# config flags; each one is a to-freeze item from RFC #501.
OFFICIAL_FROZEN_SEMANTICS: dict[str, Any] = {
    "embedding": {
        "form": "weightless_rmsnorm_after_lookup",
        "lookup_dtype": "bfloat16",
        "norm_compute_dtype": "float32",
        "norm_eps_source": "rms_norm_eps",
        "rsqrt_form": "pow_neg_half",
        "scale_by_sqrt_hidden": False,
        "output_dtype": "bfloat16",
    },
    "decoder_norm": {
        "form": "centered_rmsnorm_one_plus_weight",
        "compute_dtype": "float32",
        "rsqrt_form": "rsqrt",
        "weight_applied_before_cast": True,
        "eps": {
            "input_layernorm": "rms_norm_eps",
            "post_attention_layernorm": "post_norm_eps",
            "pre_feedforward_layernorm": "rms_norm_eps",
            "post_feedforward_layernorm": "post_norm_eps",
        },
    },
    "final_norm": {
        "form": "rmsnorm_times_weight",
        "compute_dtype": "float32",
        "rsqrt_form": "pow_neg_half",
        "eps_source": "rms_norm_eps",
    },
    "post_norm_eps_governs": ["post_attention_layernorm", "post_feedforward_layernorm"],
    "norm_residual_order": [
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "residual_add",
        "pre_feedforward_layernorm",
        "mlp",
        "post_feedforward_layernorm",
        "residual_add",
    ],
    "qk_norm": {
        "form": "weightless_rmsnorm_per_head",
        "applies_to": ["q", "k"],
        "compute_dtype": "float32",
        "rsqrt_form": "pow_neg_half",
        "eps_source": "rms_norm_eps",
        "q_scale": "qk_scale_factor",
        "q_scale_applied_after_cast": True,
        "k_scale": None,
        "order": ["proj", "qk_norm", "q_scale", "rope", "kv_cache_write"],
    },
    "attention": {
        "softmax_scale": "head_dim**-0.5",
        "effective_q_scale": "qk_scale_factor * head_dim**-0.5",
        "gqa_group_size": 16,
    },
    "rope": {
        "layers": SLIDING_ATTENTION,
        "layer_rope_theta_zero_means": "nope",
        "type": "default",
        "rotated_channels": 128,
        "layout": "rotate_half",
        "cos_sin_compute_dtype": "float32",
        "cos_sin_cast_dtype": "bfloat16",
    },
    "output_gate": {
        "form": "sigmoid",
        "granularity": "elementwise",
        "width": 4096,
        "input": "input_layernorm_output",
        "applied_to": "flattened_attention_output_before_o_proj",
        "bias": False,
    },
    "local_window": {
        "mask": "attend iff 0 <= q_pos - k_pos < sliding_window",
        "positions": "absolute",
        "keys_visible_including_self": 2048,
        "past_tokens_retained": 2047,
        "eviction": "drop_oldest_beyond_window",
    },
    "global_layers": {"mask": "causal", "positional_encoding": "none"},
    "ffn": {"form": "swiglu", "activation": "silu", "order": "down(silu(gate(x)) * up(x))"},
    "logits": {
        "formula": "T * tanh(z * output_multiplier / T)",
        "T": "final_logit_softcapping",
        "order": ["lm_head", "mul_output_multiplier", "div_T", "tanh", "mul_T"],
        "reference_dtype": "bfloat16",
    },
}

# One projection grouping for both runtimes. The checkpoint stores four separate matrices;
# Megatron Bridge fuses Q/K/V/gate and vLLM fuses Q/K/V, so both un-fuse in strict mode.
OFFICIAL_PROJECTION_GROUPING: dict[str, Any] = {
    "attention": [["q_proj"], ["k_proj"], ["v_proj"], ["gate_proj"]],
    "mlp": [["gate_proj"], ["up_proj"]],
    "fail_closed": ["qkv_proj", "qkvg_proj", "linear_qkv", "gate_up_proj", "linear_fc1"],
}

CANONICAL_TP = 8
VOCAB_TILE = 128

OFFICIAL_PARALLEL_LAYOUT: dict[str, Any] = {
    "canonical_tp": CANONICAL_TP,
    "vocab_tile": VOCAB_TILE,
    "vocab_alignment": VOCAB_TILE * CANONICAL_TP,
    "padded_vocab_size": 202752,
    "padded_logits": "excluded_from_logsumexp",
    "softcap_before_vocab_partials": True,
    "row_parallel_shards": {"o_proj": 512, "down_proj": 2496},
    "kv_heads_per_canonical_rank": "replicated",
}


class MuseGlimmerWorkloadError(ValueError):
    """Raised when the Muse Glimmer fingerprint manifest is missing a pin or inconsistent."""


@dataclass
class MuseGlimmerManifest:
    """Validated in-memory view of museglimmer_manifest.json."""

    raw: dict[str, Any]
    path: Path = field(default=_MANIFEST_PATH)

    @property
    def version(self) -> str:
        return str(self.raw["version"])

    @property
    def workload_id(self) -> str:
        return str(self.raw["workload_id"])

    @property
    def model_identity(self) -> dict[str, Any]:
        return dict(self.raw["model_identity"])

    @property
    def text_config(self) -> dict[str, Any]:
        return dict(self.raw["model_identity"]["config_fingerprint"]["text_config"])


def default_manifest_path() -> Path:
    return _MANIFEST_PATH


def expected_layer_types(num_layers: int | None = None) -> list[str]:
    n = int(OFFICIAL_FINGERPRINT["num_hidden_layers"]) if num_layers is None else num_layers
    return [
        FULL_ATTENTION if i % LAYER_PERIOD == FULL_SLOT else SLIDING_ATTENTION for i in range(n)
    ]


def padded_vocab_size(
    vocab_size: int, canonical_tp: int = CANONICAL_TP, tile: int = VOCAB_TILE
) -> int:
    alignment = tile * canonical_tp
    return -(-vocab_size // alignment) * alignment


def text_weight_schema(text_config: Mapping[str, Any] | None = None) -> dict[str, tuple[int, ...]]:
    """Every text-model tensor name and shape the pinned checkpoint must contain."""
    cfg = text_config if text_config is not None else OFFICIAL_FINGERPRINT
    h = int(cfg["hidden_size"])
    ffn = int(cfg["intermediate_size"])
    q = int(cfg["num_attention_heads"]) * int(cfg["head_dim"])
    kv = int(cfg["num_key_value_heads"]) * int(cfg["head_dim"])
    vocab = int(cfg["vocab_size"])
    schema: dict[str, tuple[int, ...]] = {
        "model.language_model.embed_tokens.weight": (vocab, h),
        "model.language_model.norm.weight": (h,),
        "lm_head.weight": (vocab, h),
    }
    per_layer = {
        "input_layernorm.weight": (h,),
        "post_attention_layernorm.weight": (h,),
        "pre_feedforward_layernorm.weight": (h,),
        "post_feedforward_layernorm.weight": (h,),
        "self_attn.q_proj.weight": (q, h),
        "self_attn.k_proj.weight": (kv, h),
        "self_attn.v_proj.weight": (kv, h),
        "self_attn.gate_proj.weight": (q, h),
        "self_attn.o_proj.weight": (h, q),
        "mlp.gate_proj.weight": (ffn, h),
        "mlp.up_proj.weight": (ffn, h),
        "mlp.down_proj.weight": (h, ffn),
    }
    for index in range(int(cfg["num_hidden_layers"])):
        for suffix, shape in per_layer.items():
            schema[f"model.language_model.layers.{index}.{suffix}"] = shape
    return schema


def load_manifest(path: str | Path | None = None) -> MuseGlimmerManifest:
    manifest_path = Path(path) if path is not None else _MANIFEST_PATH
    with manifest_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise MuseGlimmerWorkloadError("manifest root must be a JSON object")
    validate_manifest(raw)
    return MuseGlimmerManifest(raw=raw, path=manifest_path)


def validate_manifest(raw: Mapping[str, Any]) -> None:
    """Hard-fail if any required fingerprint pin is missing or inconsistent."""
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in raw]
    if missing:
        raise MuseGlimmerWorkloadError(f"manifest missing top-level keys: {missing}")
    _validate_model_identity(raw["model_identity"])
    for key, expected in (
        ("frozen_semantics", OFFICIAL_FROZEN_SEMANTICS),
        ("projection_grouping", OFFICIAL_PROJECTION_GROUPING),
        ("parallel_layout", OFFICIAL_PARALLEL_LAYOUT),
    ):
        if raw[key] != expected:
            raise MuseGlimmerWorkloadError(
                f"{key} does not match the pinned Muse-Glimmer-30B contract; change "
                "workload_id/version and the pin in workload.py together"
            )
    vocab = int(raw["model_identity"]["config_fingerprint"]["text_config"]["vocab_size"])
    layout = raw["parallel_layout"]
    if (
        padded_vocab_size(vocab, layout["canonical_tp"], layout["vocab_tile"])
        != layout["padded_vocab_size"]
    ):
        raise MuseGlimmerWorkloadError(
            "parallel_layout.padded_vocab_size is not the aligned vocabulary"
        )
    expected_identity = manifest_identity_hash(raw)
    if raw["fixture_identity_sha256"] != expected_identity:
        raise MuseGlimmerWorkloadError(
            "fixture_identity_sha256 does not match manifest; change workload_id/version "
            "and regenerate the identity for any numerics-affecting edit"
        )


def _is_sha(value: Any, length: int) -> bool:
    text = str(value)
    return len(text) == length and all(c in "0123456789abcdef" for c in text)


def _validate_model_identity(identity: Mapping[str, Any]) -> None:
    for key in (
        "model_id",
        "revision",
        "architecture",
        "semantics_source",
        "config_sha256",
        "config_fingerprint",
        "weight_snapshot",
    ):
        if key not in identity:
            raise MuseGlimmerWorkloadError(f"model_identity missing {key!r}")
    if not _is_sha(identity["revision"], 40):
        raise MuseGlimmerWorkloadError("model_identity.revision must be a lowercase 40-hex commit")
    if not _is_sha(identity["config_sha256"], 64):
        raise MuseGlimmerWorkloadError("model_identity.config_sha256 must be a lowercase SHA-256")
    if identity["architecture"] != "MuseGlimmerForConditionalGeneration":
        raise MuseGlimmerWorkloadError(
            f"architecture {identity['architecture']!r} is not MuseGlimmerForConditionalGeneration"
        )
    source = identity["semantics_source"]
    for key in ("package", "version", "git_commit", "modeling_file", "modeling_file_sha256"):
        if not isinstance(source, Mapping) or not source.get(key):
            raise MuseGlimmerWorkloadError(f"semantics_source missing {key!r}")
    if not _is_sha(source["git_commit"], 40):
        raise MuseGlimmerWorkloadError(
            "semantics_source.git_commit must be a lowercase 40-hex git SHA"
        )
    if not _is_sha(source["modeling_file_sha256"], 64):
        raise MuseGlimmerWorkloadError(
            "semantics_source.modeling_file_sha256 must be a lowercase SHA-256"
        )
    fp = identity["config_fingerprint"]
    if not isinstance(fp, Mapping) or not isinstance(fp.get("text_config"), Mapping):
        raise MuseGlimmerWorkloadError("config_fingerprint.text_config must be an object")
    validate_text_config(fp["text_config"])
    if not identity.get("exit_forbids_architecture_shrink", False):
        raise MuseGlimmerWorkloadError("exit_forbids_architecture_shrink must be true")
    _validate_weight_snapshot(identity["weight_snapshot"])


def validate_text_config(text: Mapping[str, Any]) -> None:
    """Check an HF ``text_config`` against the official pins. Shared with the strict loader."""
    for key, expected in OFFICIAL_FINGERPRINT.items():
        if key not in text:
            raise MuseGlimmerWorkloadError(f"text_config missing {key!r}")
        if text[key] != expected:
            raise MuseGlimmerWorkloadError(
                f"text_config {key}={text[key]!r} does not match official "
                f"Muse-Glimmer-30B pin {expected!r}; architecture shrink is forbidden"
            )
    num_layers = int(text["num_hidden_layers"])
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != num_layers:
        raise MuseGlimmerWorkloadError(
            f"layer_types must list exactly num_hidden_layers={num_layers} entries; "
            "architecture shrink is forbidden"
        )
    expected_types = expected_layer_types(num_layers)
    if list(layer_types) != expected_types:
        bad = [
            i for i, (a, e) in enumerate(zip(layer_types, expected_types, strict=True)) if a != e
        ]
        raise MuseGlimmerWorkloadError(
            f"layer_types deviates from the 3x sliding + 1x full schedule at layers {bad[:8]}"
        )
    thetas = text.get("layer_rope_theta")
    theta = float(OFFICIAL_FINGERPRINT["rope_parameters"]["rope_theta"])
    expected_thetas = [0 if t == FULL_ATTENTION else theta for t in expected_types]
    if not isinstance(thetas, list) or [float(t) for t in thetas] != expected_thetas:
        raise MuseGlimmerWorkloadError(
            "layer_rope_theta must be rope_theta on sliding layers and 0 (NoPE) on full layers"
        )


def _validate_weight_snapshot(weight: Mapping[str, Any]) -> None:
    for key in (
        "pin_method",
        "index_file",
        "index_sha256",
        "tensor_total_size_bytes",
        "weight_files_total_size_bytes",
        "content_hash_algorithm",
        "content_hash",
        "shards",
    ):
        if key not in weight:
            raise MuseGlimmerWorkloadError(f"weight_snapshot missing {key!r}")
    shards = weight["shards"]
    if not isinstance(shards, list) or not shards:
        raise MuseGlimmerWorkloadError("weight_snapshot.shards must be a non-empty list")
    filenames = [str(shard.get("filename", "")) for shard in shards]
    if len(set(filenames)) != len(filenames) or any(not name for name in filenames):
        raise MuseGlimmerWorkloadError(
            "weight_snapshot shard filenames must be unique and non-empty"
        )
    if int(weight["weight_files_total_size_bytes"]) != sum(int(s["size_bytes"]) for s in shards):
        raise MuseGlimmerWorkloadError("weight_snapshot file total does not match shard sizes")
    for shard in shards:
        if not _is_sha(shard.get("sha256", ""), 64):
            raise MuseGlimmerWorkloadError("every weight shard must pin a lowercase SHA-256")
    if not _is_sha(weight["index_sha256"], 64):
        raise MuseGlimmerWorkloadError("weight_snapshot.index_sha256 must be a lowercase SHA-256")
    if weight["content_hash_algorithm"] != "sha256-of-sorted-shard-records-v1":
        raise MuseGlimmerWorkloadError("unsupported weight_snapshot content_hash_algorithm")
    if weight["content_hash"] != weight_snapshot_hash(shards):
        raise MuseGlimmerWorkloadError("weight_snapshot content_hash does not match shard records")


def reference_payload(manifest: MuseGlimmerManifest | None = None) -> dict[str, Any]:
    """Payload emitted by tools/validation/models/museglimmer_reference.py (no model forward)."""
    m = manifest if manifest is not None else load_manifest()
    identity = m.model_identity
    return {
        "workload_id": m.workload_id,
        "version": m.version,
        "fixture_identity_sha256": str(m.raw["fixture_identity_sha256"]),
        "model_id": identity["model_id"],
        "revision": identity["revision"],
        "semantics_source": identity["semantics_source"],
        "config_fingerprint": identity["config_fingerprint"],
        "frozen_semantics": m.raw["frozen_semantics"],
        "projection_grouping": m.raw["projection_grouping"],
        "parallel_layout": m.raw["parallel_layout"],
        "weight_snapshot": identity["weight_snapshot"],
    }


__all__ = [
    "CANONICAL_TP",
    "FULL_ATTENTION",
    "LAYER_PERIOD",
    "MuseGlimmerManifest",
    "MuseGlimmerWorkloadError",
    "OFFICIAL_FINGERPRINT",
    "OFFICIAL_FROZEN_SEMANTICS",
    "OFFICIAL_PARALLEL_LAYOUT",
    "OFFICIAL_PROJECTION_GROUPING",
    "SLIDING_ATTENTION",
    "VOCAB_TILE",
    "default_manifest_path",
    "expected_layer_types",
    "load_manifest",
    "manifest_identity_hash",
    "padded_vocab_size",
    "reference_payload",
    "text_weight_schema",
    "validate_manifest",
    "validate_text_config",
    "weight_snapshot_hash",
]
