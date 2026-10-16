# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 M10 export (draft) for same-fixture pairing with T06 and the T09 byte verdict.

Format (``rlkernel.p7.t05.m10_export.v1-draft``; replaced by the T01 schema once frozen):

``m10.json``
    Canonical JSON: schema version, side, row keys in local row order
    (``[sample_id, position]``, ``null`` for padding rows), BOOL active mask in
    the same order, byte encoding, active-byte SHA-256, the validated evidence
    identity and the optional upstream evidence.
``m10_active.bin``
    FP32 selected logprob of the active rows only, in local row order, each
    value written as the little-endian bytes of its ``int32`` view.

Files are created exclusively (never overwritten). The local comparison below
is a T05 self-test of the export format; the M10 byte verdict belongs to T09.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from rl_engine.alignment.cross_config.adapters.megatron_training import T05ContractError
from rl_engine.alignment.cross_config.adapters.megatron_training_gate import TrainingM10Evidence
from rl_engine.alignment.cross_config.adapters.megatron_training_upstream import UpstreamEvidence

M10_EXPORT_SCHEMA_VERSION = "rlkernel.p7.t05.m10_export.v1-draft"
META_NAME = "m10.json"
BYTES_NAME = "m10_active.bin"
BYTE_ENCODING = {
    "dtype": "float32",
    "view": "int32",
    "endianness": "little",
    "selection": "active rows only, local row order",
}
SELF_TEST_AUTHORITY = "none: T05 export self-test; the M10 byte verdict belongs to T09"


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise T05ContractError("CORRUPT_ARTIFACT", str(path), "refusing to overwrite") from exc


def export_m10(
    evidence: TrainingM10Evidence,
    directory: Path | str,
    *,
    side: str = "training",
    upstream: Optional[UpstreamEvidence] = None,
) -> Path:
    if sys.byteorder != "little":
        raise T05ContractError("UNSUPPORTED_CAPABILITY", "byteorder", sys.byteorder)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    raw = evidence.active_raw_bytes
    if hashlib.sha256(raw).hexdigest() != evidence.active_bytes_sha256:
        raise T05ContractError("CORRUPT_ARTIFACT", "active_raw_bytes", "evidence digest mismatch")
    mask = evidence.active_token_mask.tolist()
    meta = {
        "schema_version": M10_EXPORT_SCHEMA_VERSION,
        "side": side,
        "row_keys": [list(key) if key is not None else None for key in evidence.row_keys],
        "active_token_mask": [bool(value) for value in mask],
        "active_count": int(sum(mask)),
        "byte_encoding": dict(BYTE_ENCODING),
        "active_bytes_sha256": evidence.active_bytes_sha256,
        "evidence": evidence.to_dict(),
        "evidence_fingerprint": evidence.fingerprint,
        "upstream": None if upstream is None else upstream.to_dict(),
        "upstream_fingerprint": None if upstream is None else upstream.fingerprint,
    }
    _write_new(target / BYTES_NAME, raw)
    _write_new(target / META_NAME, _canonical(meta) + b"\n")
    return target


@dataclass(frozen=True)
class LoadedM10:
    side: str
    row_keys: tuple[Optional[tuple[str, int]], ...]
    active_token_mask: torch.Tensor
    active_values: torch.Tensor  # FP32 [active_count], local row order
    active_raw_bytes: bytes
    meta: Mapping[str, Any]

    @property
    def active_keys(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            key  # type: ignore[misc]
            for key, active in zip(self.row_keys, self.active_token_mask.tolist(), strict=True)
            if active
        )


def load_m10(directory: Path | str) -> LoadedM10:
    source = Path(directory)
    meta_path, bytes_path = source / META_NAME, source / BYTES_NAME
    missing = [path.name for path in (meta_path, bytes_path) if not path.is_file()]
    if missing:
        raise T05ContractError("INCOMPLETE_ARTIFACT", str(source), f"missing {missing}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise T05ContractError("CORRUPT_ARTIFACT", META_NAME, str(exc)) from exc
    if meta.get("schema_version") != M10_EXPORT_SCHEMA_VERSION:
        raise T05ContractError(
            "SCHEMA_MISMATCH", "schema_version", repr(meta.get("schema_version"))
        )
    if meta.get("byte_encoding") != BYTE_ENCODING:
        raise T05ContractError("SCHEMA_MISMATCH", "byte_encoding", repr(meta.get("byte_encoding")))
    raw = bytes_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta.get("active_bytes_sha256"):
        raise T05ContractError("CORRUPT_ARTIFACT", BYTES_NAME, "SHA-256 does not match m10.json")
    mask = meta.get("active_token_mask")
    keys = meta.get("row_keys")
    if not isinstance(mask, list) or not isinstance(keys, list) or len(mask) != len(keys):
        raise T05ContractError("CORRUPT_ARTIFACT", META_NAME, "row keys and mask differ in length")
    active_count = sum(1 for value in mask if value)
    if active_count != meta.get("active_count") or len(raw) != 4 * active_count:
        raise T05ContractError(
            "CORRUPT_ARTIFACT", BYTES_NAME, "byte length does not match the mask"
        )
    if any(key is None and active for key, active in zip(keys, mask, strict=True)):
        raise T05ContractError("CORRUPT_ARTIFACT", META_NAME, "active padding row")
    values = torch.frombuffer(bytearray(raw), dtype=torch.int32).view(torch.float32).clone()
    return LoadedM10(
        side=str(meta.get("side")),
        row_keys=tuple(None if key is None else (str(key[0]), int(key[1])) for key in keys),
        active_token_mask=torch.tensor(mask, dtype=torch.bool),
        active_values=values,
        active_raw_bytes=raw,
        meta=meta,
    )


@dataclass(frozen=True)
class SelfTestReport:
    """Local export-format self-test; carries no verdict authority."""

    identical: bool
    compared_tokens: int
    first_difference: Optional[Mapping[str, Any]]
    authority: str = SELF_TEST_AUTHORITY


def self_test_compare(left: LoadedM10, right: LoadedM10) -> SelfTestReport:
    """Compare two exports token by token on raw FP32 bytes (e.g. two T05 runs)."""

    left_keys, right_keys = left.active_keys, right.active_keys
    if set(left_keys) != set(right_keys) or len(left_keys) != len(right_keys):
        raise T05ContractError(
            "IDENTITY_DRIFT", "active_keys", "exports cover different active global tokens"
        )
    right_bits = dict(zip(right_keys, right.active_values.view(torch.int32).tolist(), strict=True))
    left_bits = left.active_values.view(torch.int32).tolist()
    for key, bits in zip(left_keys, left_bits, strict=True):
        if bits != right_bits[key]:
            return SelfTestReport(
                identical=False,
                compared_tokens=len(left_keys),
                first_difference={
                    "token": list(key),
                    "left_bits": f"0x{bits & 0xFFFFFFFF:08x}",
                    "right_bits": f"0x{right_bits[key] & 0xFFFFFFFF:08x}",
                },
            )
    return SelfTestReport(identical=True, compared_tokens=len(left_keys), first_difference=None)


__all__ = [
    "BYTES_NAME",
    "BYTE_ENCODING",
    "M10_EXPORT_SCHEMA_VERSION",
    "META_NAME",
    "SELF_TEST_AUTHORITY",
    "LoadedM10",
    "SelfTestReport",
    "export_m10",
    "load_m10",
    "self_test_compare",
]
