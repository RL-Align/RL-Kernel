# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Strict Muse Glimmer checkpoint loader checks.

Verifies a local Muse-Glimmer-30B snapshot against the pinned fingerprint before either
runtime loads it, and rejects runtime bindings that strict mode does not support. Every
check fails closed with :class:`MuseGlimmerWorkloadError`. Nothing here imports torch.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_engine.models.muse_glimmer.workload import (
    OFFICIAL_PROJECTION_GROUPING,
    MuseGlimmerManifest,
    MuseGlimmerWorkloadError,
    load_manifest,
    text_weight_schema,
    validate_text_config,
)

_TEXT_PREFIXES = ("model.language_model.", "lm_head.")
_HASH_CHUNK = 64 << 20

# Model classes that implement Muse Glimmer natively. Anything else, including the
# Transformers fallback classes in vLLM and SGLang, is rejected.
NATIVE_MODEL_CLASSES = frozenset(
    {
        "MuseGlimmerForConditionalGeneration",
        "MuseGlimmerForCausalLM",
    }
)


@dataclass(frozen=True)
class CheckpointReport:
    """What :func:`verify_checkpoint` checked, for provenance records."""

    path: str
    revision: str
    config_sha256: str
    index_sha256: str
    text_tensor_count: int
    headers_checked: tuple[str, ...]
    shards_hashed: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


_MAX_HEADER_BYTES = (
    100 << 20
)  # far above any real header; a corrupt length prefix must not allocate


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """Read only the JSON header of a safetensors file (fails closed on a damaged header)."""
    with Path(path).open("rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise MuseGlimmerWorkloadError(f"{path}: not a safetensors file")
        (length,) = struct.unpack("<Q", prefix)
        if length > _MAX_HEADER_BYTES:
            raise MuseGlimmerWorkloadError(f"{path}: safetensors header length {length} is invalid")
        blob = fh.read(length)
    if len(blob) != length:
        raise MuseGlimmerWorkloadError(f"{path}: truncated safetensors header")
    try:
        header = json.loads(blob)
    except ValueError as exc:
        raise MuseGlimmerWorkloadError(f"{path}: invalid safetensors header") from exc
    if not isinstance(header, dict):
        raise MuseGlimmerWorkloadError(f"{path}: safetensors header must be a JSON object")
    header.pop("__metadata__", None)
    return header


def verify_config(config: Mapping[str, Any], manifest: MuseGlimmerManifest) -> None:
    """Check a parsed config.json against the manifest; every text_config key must match."""
    architectures = list(config.get("architectures") or [])
    expected_arch = manifest.model_identity["architecture"]
    if architectures != [expected_arch]:
        raise MuseGlimmerWorkloadError(
            f"config architectures={architectures!r}, expected [{expected_arch!r}]"
        )
    text = config.get("text_config")
    if not isinstance(text, Mapping):
        raise MuseGlimmerWorkloadError("config.json has no text_config")
    validate_text_config(text)
    pinned = manifest.text_config
    drift = sorted(k for k in set(text) | set(pinned) if text.get(k) != pinned.get(k))
    if drift:
        raise MuseGlimmerWorkloadError(
            f"text_config differs from the pinned config at keys {drift}"
        )


def verify_weight_map(weight_map: Mapping[str, str], manifest: MuseGlimmerManifest) -> None:
    """Check the index weight map: exact text tensor set, no fused projections, pinned shards."""
    text_keys = {k for k in weight_map if k.startswith(_TEXT_PREFIXES)}
    fused = sorted(
        k
        for k in text_keys
        if any(f".{name}." in k for name in OFFICIAL_PROJECTION_GROUPING["fail_closed"])
    )
    if fused:
        raise MuseGlimmerWorkloadError(
            f"fused projection tensors {fused[:4]} are not the pinned Q/K/V/gate grouping"
        )
    schema = text_weight_schema(manifest.text_config)
    missing = sorted(set(schema) - text_keys)
    extra = sorted(text_keys - set(schema))
    if missing or extra:
        raise MuseGlimmerWorkloadError(
            f"text tensors differ from the pinned schema: missing={missing[:4]} extra={extra[:4]}"
        )
    pinned_shards = {s["filename"] for s in manifest.model_identity["weight_snapshot"]["shards"]}
    unknown = sorted(set(weight_map.values()) - pinned_shards)
    if unknown:
        raise MuseGlimmerWorkloadError(f"weight map references unpinned shards {unknown}")


def verify_tensor_headers(
    headers: Mapping[str, Mapping[str, Any]], manifest: MuseGlimmerManifest
) -> None:
    """Check dtype and shape of every text tensor found in the given safetensors headers."""
    schema = text_weight_schema(manifest.text_config)
    dtype = manifest.model_identity["weight_snapshot"]["text_tensor_dtype"]
    for name, info in headers.items():
        if name not in schema:
            continue
        if info.get("dtype") != dtype or tuple(info.get("shape", ())) != schema[name]:
            raise MuseGlimmerWorkloadError(
                f"{name}: {info.get('dtype')} {tuple(info.get('shape', ()))} != pinned "
                f"{dtype} {schema[name]}"
            )


def verify_checkpoint(
    path: str | Path,
    manifest: MuseGlimmerManifest | None = None,
    *,
    require_pinned_bytes: bool = True,
    verify_shard_sha256: bool = False,
) -> CheckpointReport:
    """Verify a local checkpoint directory against the fingerprint.

    ``require_pinned_bytes`` also compares config.json and the index file byte-for-byte
    (SHA-256) and every shard's size with the pinned revision, so a truncated shard with an
    intact header fails closed without hashing. ``verify_shard_sha256`` hashes every shard
    (about 60 GB); without it, shard identity rests on the pinned index, headers and sizes.
    """
    m = manifest if manifest is not None else load_manifest()
    root = Path(path)
    snapshot = m.model_identity["weight_snapshot"]

    config_path = root / "config.json"
    index_path = root / snapshot["index_file"]
    for required in (config_path, index_path):
        if not required.is_file():
            raise MuseGlimmerWorkloadError(f"checkpoint is missing {required.name}")
    config_sha = _sha256_file(config_path)
    index_sha = _sha256_file(index_path)
    if require_pinned_bytes:
        if config_sha != m.model_identity["config_sha256"]:
            raise MuseGlimmerWorkloadError(
                f"config.json sha256 {config_sha} is not the pinned revision "
                f"{m.model_identity['revision']}"
            )
        if index_sha != snapshot["index_sha256"]:
            raise MuseGlimmerWorkloadError(
                f"{index_path.name} sha256 {index_sha} is not the pinned revision "
                f"{m.model_identity['revision']}"
            )

    verify_config(json.loads(config_path.read_text(encoding="utf-8")), m)
    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise MuseGlimmerWorkloadError(f"{index_path.name} has no weight_map")
    verify_weight_map(weight_map, m)

    headers: dict[str, Mapping[str, Any]] = {}
    checked: list[str] = []
    hashed: list[str] = []
    for shard in snapshot["shards"]:
        shard_path = root / shard["filename"]
        if not shard_path.is_file():
            raise MuseGlimmerWorkloadError(f"checkpoint is missing shard {shard['filename']}")
        headers.update(read_safetensors_header(shard_path))
        checked.append(shard["filename"])
        if require_pinned_bytes and shard_path.stat().st_size != int(shard["size_bytes"]):
            raise MuseGlimmerWorkloadError(f"{shard['filename']}: size differs from the pin")
        if verify_shard_sha256:
            if _sha256_file(shard_path) != shard["sha256"]:
                raise MuseGlimmerWorkloadError(f"{shard['filename']}: sha256 differs from the pin")
            hashed.append(shard["filename"])
    missing = sorted(set(text_weight_schema(m.text_config)) - set(headers))
    if missing:
        raise MuseGlimmerWorkloadError(f"shard headers are missing text tensors {missing[:4]}")
    verify_tensor_headers(headers, m)

    return CheckpointReport(
        path=str(root),
        revision=str(m.model_identity["revision"]),
        config_sha256=config_sha,
        index_sha256=index_sha,
        text_tensor_count=len(text_weight_schema(m.text_config)),
        headers_checked=tuple(checked),
        shards_hashed=tuple(hashed),
    )


def check_runtime_binding(
    *,
    model_class: str,
    attention_projection_grouping: Sequence[Sequence[str]],
    speculative_decoding: bool,
) -> None:
    """Fail closed on a runtime binding strict mode does not support."""
    if model_class not in NATIVE_MODEL_CLASSES:
        raise MuseGlimmerWorkloadError(
            f"model class {model_class!r} is not a native Muse Glimmer class; "
            "Transformers-fallback classes are not strict"
        )
    grouping = [list(group) for group in attention_projection_grouping]
    if grouping != OFFICIAL_PROJECTION_GROUPING["attention"]:
        raise MuseGlimmerWorkloadError(
            f"attention projection grouping {grouping!r} is not the pinned "
            f"{OFFICIAL_PROJECTION_GROUPING['attention']!r}"
        )
    if speculative_decoding:
        raise MuseGlimmerWorkloadError(
            "speculative decoding is not supported in strict mode (RFC #501)"
        )


__all__ = [
    "CheckpointReport",
    "NATIVE_MODEL_CLASSES",
    "check_runtime_binding",
    "read_safetensors_header",
    "verify_checkpoint",
    "verify_config",
    "verify_tensor_headers",
    "verify_weight_map",
]
