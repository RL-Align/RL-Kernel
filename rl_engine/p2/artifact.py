# SPDX-License-Identifier: Apache-2.0
"""Portable sealed artifacts. SHA-256 integrity seals are NOT signatures."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import platform
import struct
import sys
import tempfile
from pathlib import Path

import torch

from .contract import (
    MODES,
    OPERATORS,
    REFERENCE_PROFILE,
    SCHEMA_VERSION,
    ContractError,
    Status,
    canonical,
    check_compatibility,
    digest,
    require,
    schema_guard,
    validate_runtime_policy,
)
from .fixtures import catalog, required_evidence, selector_recipe
from .negative import evaluate_negative, negative_cases
from .planner import validate_state_sequence

ARTIFACT_VERSION = "p2-sealed-artifact.v1"
DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "int64": 8, "uint8": 1, "bool": 1}


def source_fingerprint() -> str:
    root = Path(__file__).parent
    return digest(
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    )


def provenance() -> dict:
    # No hostname, address, username, environment dump or private checkpoint path.
    return {
        "kind": "actual",
        "implementation": source_fingerprint(),
        "backend": "cpu",
        "device": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "byteorder": sys.byteorder,
        "execution": "torch_eager_reference",
        "kernel": None,
        "tile": None,
        "warp": None,
        "stage": None,
        "actual_unroll": None,
        "vectorization": None,
        "fast_math": None,
        "fma_reassociation": None,
        "register_spill": None,
        "readback_scope": "Python/PyTorch runtime only; native kernel fields unavailable",
    }


def _walk_tensors(value) -> None:
    if isinstance(value, dict):
        if value.get("schema") == "p2-selector-weight.v1":
            require(
                value == selector_recipe(*value["shape"], value["salt"]),
                Status.IDENTITY_DRIFT,
                "selector weight recipe",
            )
            return
        if "dtype" in value or "shape" in value or "sha256" in value:
            require(
                set(value) == {"dtype", "shape", "endian", "sha256", "data"},
                Status.SCHEMA_MISMATCH,
                "tensor record keys",
            )
            require(
                value["dtype"] in DTYPE_BYTES and value["endian"] == "little",
                Status.SCHEMA_MISMATCH,
                "tensor format",
            )
            require(
                isinstance(value["shape"], list)
                and all(type(d) is int and d >= 0 for d in value["shape"]),
                Status.SCHEMA_MISMATCH,
                "tensor shape",
            )
            raw = base64.b64decode(value["data"], validate=True)
            size = math.prod(value["shape"]) * DTYPE_BYTES[value["dtype"]]
            require(
                len(raw) == size and hashlib.sha256(raw).hexdigest() == value["sha256"],
                Status.CORRUPT_ARTIFACT,
                "tensor payload",
            )
            if value["dtype"] == "float32" and raw:
                require(
                    all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", raw)),
                    Status.NON_FINITE,
                    "recorded tensor",
                )
            if value["dtype"] == "bfloat16" and raw:
                require(
                    all((v[0] & 0x7F80) != 0x7F80 for v in struct.iter_unpack("<H", raw)),
                    Status.NON_FINITE,
                    "recorded BF16",
                )
            if value["dtype"] == "bool":
                require(all(v in (0, 1) for v in raw), Status.SCHEMA_MISMATCH, "bool bytes")
            return
        for child in value.values():
            _walk_tensors(child)
    elif isinstance(value, list):
        for child in value:
            _walk_tensors(child)


@schema_guard
def validate_payload(payload: dict) -> dict:
    """Derive a scoped verdict from sealed evidence, never trust a supplied PASS."""
    require(
        payload["schema"] == SCHEMA_VERSION and payload["profile"] == REFERENCE_PROFILE,
        Status.SCHEMA_MISMATCH,
        "payload",
    )
    check_compatibility(payload["contract"])
    validate_runtime_policy(payload["runtime_policy"])
    require(payload["catalog"] == catalog(), Status.IDENTITY_DRIFT, "fixture catalog")
    require(
        payload["required_evidence"] == required_evidence(),
        Status.INCOMPLETE_ARTIFACT,
        "required-evidence matrix",
    )
    require(
        payload["negative_cases"] == negative_cases(),
        Status.IDENTITY_DRIFT,
        "negative fixture corpus",
    )
    for record in payload["negative_cases"]:
        require(
            evaluate_negative(record) == record["expected_status"],
            Status.BYTE_MISMATCH,
            record["case_id"],
        )
    prov = payload["provenance"]
    require(
        prov["kind"] == "actual"
        and prov["backend"] == "cpu"
        and prov["implementation"]
        and prov["torch"]
        and prov["python"]
        and prov["byteorder"] == "little",
        Status.MISSING_PROVENANCE,
        "actual reference runtime",
    )
    recordings = payload["recordings"]
    require(
        set(recordings) == {op.name for op in OPERATORS},
        Status.INCOMPLETE_ARTIFACT,
        "15 boundary recordings",
    )
    for name, recording in recordings.items():
        require(
            recording["operator"] == name
            and recording["profile"] == REFERENCE_PROFILE
            and recording["evidence_level"] == "synthetic_reference",
            Status.IDENTITY_DRIFT,
            "recording identity",
        )
        require(
            recording["checksum"]
            == digest({k: v for k, v in recording.items() if k != "checksum"}),
            Status.CORRUPT_ARTIFACT,
            "boundary checksum",
        )
    _walk_tensors(recordings)
    require(
        set(payload["sequences"]) == {"C0", "C4", "C128"},
        Status.INCOMPLETE_ARTIFACT,
        "named layer fixtures",
    )
    modes = {}
    for layer, sequence in payload["sequences"].items():
        snapshots = sequence["snapshots"]
        require(
            type(sequence["tokens"]) is int
            and sequence["tokens"] > 0
            and len(snapshots) == sequence["tokens"],
            Status.INCOMPLETE_ARTIFACT,
            "per-token state snapshots",
        )
        identity = snapshots[0]["identity"]
        require(identity["layer"] == layer, Status.IDENTITY_DRIFT, "layer/state identity")
        state_hashes = validate_state_sequence(snapshots, identity)
        require(
            set(sequence["mode_state_hashes"]) == set(MODES),
            Status.INCOMPLETE_ARTIFACT,
            "four independent modes",
        )
        for mode, hashes in sequence["mode_state_hashes"].items():
            require(
                hashes == state_hashes,
                Status.STATE_BYTES_MISMATCH,
                f"{layer}/{mode}",
            )
            modes[f"{layer}/{mode}"] = Status.PASS.value
    return {
        "status": "PASS",
        "scope": "synthetic_reference_only",
        "state_modes": modes,
        "operator_recordings": len(recordings),
        "negative_cases": len(payload["negative_cases"]),
        "live_ws1": "UNSUPPORTED_CAPABILITY",
        "live_ws2": "UNSUPPORTED_CAPABILITY",
        "integration": "UNSUPPORTED_CAPABILITY",
        "performance": "NOT_CERTIFIED",
    }


def resume_identity(payload_checksum: str) -> str:
    """Bind resume to ALL inputs, states, boundaries and producer provenance."""
    return digest({"schema": "p2-resume.v1", "payload_checksum": payload_checksum})


def seal(directory: str | Path, payload: dict) -> dict:
    """Create a fresh artifact atomically; never overwrite another run."""
    report = validate_payload(payload)
    target = Path(directory)
    require(
        not target.exists() and not target.is_symlink(),
        Status.INCOMPLETE_ARTIFACT,
        "refusing to overwrite artifact",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(payload)
    payload_checksum = hashlib.sha256(raw).hexdigest()
    manifest = {
        "schema": ARTIFACT_VERSION,
        "payload_file": "payload.json",
        "sha256": payload_checksum,
        "bytes": len(raw),
        "resume_identity": resume_identity(payload_checksum),
        "completion": True,
    }
    # Sibling staging directory and rename: a crash never exposes a completed run.
    with tempfile.TemporaryDirectory(prefix=".p2-stage-", dir=target.parent) as staging:
        stage = Path(staging) / "artifact"
        stage.mkdir()
        for name, content in (
            ("payload.json", raw),
            ("manifest.json", canonical(manifest)),
            ("seal.sha256", digest(manifest).encode("ascii")),
        ):
            with (stage / name).open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        stage.rename(target)
    return report


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, Status.CORRUPT_ARTIFACT, "duplicate JSON key")
        result[key] = value
    return result


def _json(raw: bytes) -> dict:
    def bad_constant(value):
        raise ContractError(Status.NON_FINITE, value)

    return json.loads(raw, object_pairs_hook=_object, parse_constant=bad_constant)


def read_artifact(directory: str | Path) -> tuple[dict, dict]:
    try:
        target = Path(directory)
        require(
            target.is_dir() and not target.is_symlink(),
            Status.INCOMPLETE_ARTIFACT,
            "artifact directory",
        )
        names = {"payload.json", "manifest.json", "seal.sha256"}
        require(
            {p.name for p in target.iterdir()} == names,
            Status.INCOMPLETE_ARTIFACT,
            "artifact inventory",
        )
        require(
            all((target / name).is_file() and not (target / name).is_symlink() for name in names),
            Status.CORRUPT_ARTIFACT,
            "no symlink payloads",
        )
        m = _json((target / "manifest.json").read_bytes())
        require(
            set(m) == {"schema", "payload_file", "sha256", "bytes", "resume_identity", "completion"}
            and m["schema"] == ARTIFACT_VERSION
            and m["payload_file"] == "payload.json",
            Status.SCHEMA_MISMATCH,
            "artifact manifest",
        )
        require(m["completion"] is True, Status.INCOMPLETE_ARTIFACT, "completion seal")
        require(
            (target / "seal.sha256").read_text("ascii") == digest(m),
            Status.CORRUPT_ARTIFACT,
            "manifest seal",
        )
        raw = (target / "payload.json").read_bytes()
        require(
            type(m["bytes"]) is int
            and len(raw) == m["bytes"]
            and hashlib.sha256(raw).hexdigest() == m["sha256"],
            Status.CORRUPT_ARTIFACT,
            "payload checksum",
        )
        payload = _json(raw)
        require(
            m["resume_identity"] == resume_identity(m["sha256"]),
            Status.IDENTITY_DRIFT,
            "resume identity",
        )
        return payload, validate_payload(payload)
    except ContractError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        OverflowError,
    ) as exc:
        raise ContractError(Status.CORRUPT_ARTIFACT, type(exc).__name__) from exc
