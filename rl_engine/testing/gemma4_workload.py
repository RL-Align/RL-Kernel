# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Gemma 4 architecture fingerprint.

This module freezes the google/gemma-4-31B-it text-model identity used by later Gemma 4
gates. It does not run the full model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

_MANIFEST_PATH = Path(__file__).with_name("gemma4_manifest.json")

_REQUIRED_TOP_LEVEL = (
    "version",
    "workload_id",
    "model_identity",
    "fixture_identity_sha256",
)

_OFFICIAL_FINGERPRINT = {
    "num_hidden_layers": 60,
    "hidden_size": 5376,
    "intermediate_size": 21504,
    "num_attention_heads": 32,
    "num_key_value_heads": 16,
    "head_dim": 256,
    "num_global_key_value_heads": 4,
    "global_head_dim": 512,
    "sliding_window": 1024,
    "attention_k_eq_v": True,
    "num_kv_shared_layers": 0,
    "vocab_size": 262144,
}

_SLIDING_ATTENTION = "sliding_attention"
_FULL_ATTENTION = "full_attention"
_LAYER_PERIOD = 6
_FULL_SLOT = 5  # layer i is full_attention iff i % _LAYER_PERIOD == _FULL_SLOT


class Gemma4WorkloadError(ValueError):
    """Raised when the Gemma 4 fingerprint manifest is missing a pin or inconsistent."""


@dataclass
class Gemma4Manifest:
    """Validated in-memory view of gemma4_manifest.json."""

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


def default_manifest_path() -> Path:
    return _MANIFEST_PATH


def load_manifest(path: str | Path | None = None) -> Gemma4Manifest:
    manifest_path = Path(path) if path is not None else _MANIFEST_PATH
    with manifest_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise Gemma4WorkloadError("manifest root must be a JSON object")
    validate_manifest(raw)
    return Gemma4Manifest(raw=raw, path=manifest_path)


def validate_manifest(raw: Mapping[str, Any]) -> None:
    """Hard-fail if any required fingerprint pin is missing or inconsistent."""
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in raw]
    if missing:
        raise Gemma4WorkloadError(f"manifest missing top-level keys: {missing}")
    _validate_model_identity(raw["model_identity"])
    expected_identity = manifest_identity_hash(raw)
    if raw["fixture_identity_sha256"] != expected_identity:
        raise Gemma4WorkloadError(
            "fixture_identity_sha256 does not match manifest; change workload_id/version "
            "and regenerate the identity for any numerics-affecting edit"
        )


def _validate_model_identity(identity: Mapping[str, Any]) -> None:
    for key in ("model_id", "revision", "config_fingerprint", "weight_snapshot"):
        if key not in identity:
            raise Gemma4WorkloadError(f"model_identity missing {key!r}")
    fp = identity["config_fingerprint"]
    if not isinstance(fp, Mapping) or not isinstance(fp.get("text_config"), Mapping):
        raise Gemma4WorkloadError("config_fingerprint.text_config must be an object")
    text = fp["text_config"]
    for key, expected in _OFFICIAL_FINGERPRINT.items():
        if key not in text:
            raise Gemma4WorkloadError(f"text_config missing {key!r}")
        if text[key] != expected:
            raise Gemma4WorkloadError(
                f"text_config {key}={text[key]!r} does not match official "
                f"Gemma-4-31B-it pin {expected!r}; architecture shrink is forbidden"
            )
    layer_types = text.get("layer_types")
    num_layers = int(text["num_hidden_layers"])
    if not isinstance(layer_types, list) or len(layer_types) != num_layers:
        raise Gemma4WorkloadError(
            f"layer_types must list exactly num_hidden_layers={num_layers} entries; "
            "architecture shrink is forbidden"
        )
    expected_types = [
        _FULL_ATTENTION if i % _LAYER_PERIOD == _FULL_SLOT else _SLIDING_ATTENTION
        for i in range(num_layers)
    ]
    if list(layer_types) != expected_types:
        bad = [i for i, (a, e) in enumerate(zip(layer_types, expected_types)) if a != e]
        raise Gemma4WorkloadError(
            f"layer_types deviates from the 5x sliding + 1x full schedule at layers {bad[:8]}"
        )
    if not identity.get("exit_forbids_architecture_shrink", False):
        raise Gemma4WorkloadError("exit_forbids_architecture_shrink must be true")
    weight = identity["weight_snapshot"]
    for key in (
        "pin_method",
        "total_size_bytes",
        "index_file",
        "index_sha256",
        "content_hash_algorithm",
        "content_hash",
        "shards",
        "weight_files_total_size_bytes",
    ):
        if key not in weight:
            raise Gemma4WorkloadError(f"weight_snapshot missing {key!r}")
    shards = weight["shards"]
    if not isinstance(shards, list) or not shards:
        raise Gemma4WorkloadError("weight_snapshot.shards must be a non-empty list")
    filenames = [str(shard.get("filename", "")) for shard in shards]
    if len(set(filenames)) != len(filenames) or any(not name for name in filenames):
        raise Gemma4WorkloadError("weight_snapshot shard filenames must be unique and non-empty")
    if int(weight["weight_files_total_size_bytes"]) != sum(int(s["size_bytes"]) for s in shards):
        raise Gemma4WorkloadError("weight_snapshot file total does not match shard sizes")
    for shard in shards:
        digest = str(shard.get("sha256", ""))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise Gemma4WorkloadError("every weight shard must pin a lowercase SHA-256")
    index_digest = str(weight["index_sha256"])
    if len(index_digest) != 64 or any(c not in "0123456789abcdef" for c in index_digest):
        raise Gemma4WorkloadError("weight_snapshot.index_sha256 must be a lowercase SHA-256")
    expected_content_hash = weight_snapshot_hash(shards)
    if weight["content_hash_algorithm"] != "sha256-of-sorted-shard-records-v1":
        raise Gemma4WorkloadError("unsupported weight_snapshot content_hash_algorithm")
    if weight["content_hash"] != expected_content_hash:
        raise Gemma4WorkloadError("weight_snapshot content_hash does not match shard records")


def _manifest_identity_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    # Hash every declared section so future manifest keys cannot escape identity.
    return {k: v for k, v in raw.items() if k != "fixture_identity_sha256"}


def manifest_identity_hash(raw: Mapping[str, Any]) -> str:
    blob = json.dumps(
        _manifest_identity_payload(raw), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def weight_snapshot_hash(shards: Sequence[Mapping[str, Any]]) -> str:
    """Hash canonical filename/SHA-256/size records for all weight shards."""
    records = sorted((str(s["filename"]), str(s["sha256"]), int(s["size_bytes"])) for s in shards)
    blob = "".join(f"{name}\t{digest}\t{size}\n" for name, digest, size in records)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def reference_payload(manifest: Gemma4Manifest | None = None) -> dict[str, Any]:
    """Payload emitted by scripts/gemma4_reference.py (no model forward)."""
    m = manifest if manifest is not None else load_manifest()
    identity = m.model_identity
    return {
        "workload_id": m.workload_id,
        "version": m.version,
        "fixture_identity_sha256": str(m.raw["fixture_identity_sha256"]),
        "model_id": identity["model_id"],
        "revision": identity["revision"],
        "config_fingerprint": identity["config_fingerprint"],
        "weight_snapshot": identity["weight_snapshot"],
    }
