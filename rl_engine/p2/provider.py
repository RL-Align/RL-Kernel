# SPDX-License-Identifier: Apache-2.0
"""Recorded-to-live replacement seam with explicit capability and state-first gates."""

from __future__ import annotations

import copy
from typing import Protocol

from .contract import (
    ABI_VERSION,
    MODES,
    REFERENCE_PROFILE,
    SCHEMA_VERSION,
    Identity,
    Status,
    canonical,
    check_compatibility,
    digest,
    manifest,
    require,
    schema_guard,
    validate_runtime_policy,
)
from .planner import compare_snapshots, validate_state_sequence


class Provider(Protocol):
    def describe(self) -> dict: ...

    def run(self, case_id: str, mode: str) -> dict: ...


@schema_guard
def validate_envelope(envelope: dict) -> None:
    require(
        envelope["schema"] == SCHEMA_VERSION and envelope["foundation_abi"] == ABI_VERSION,
        Status.SCHEMA_MISMATCH,
        "TraceEnvelope",
    )
    Identity(**envelope["identity"]).validate()
    require(envelope["mode"] in MODES, Status.UNSUPPORTED_CAPABILITY, "mode")
    require(
        isinstance(envelope["case_id"], str) and bool(envelope["case_id"]),
        Status.SCHEMA_MISMATCH,
        "case id",
    )
    require(
        envelope["profile"] == envelope["identity"]["profile"],
        Status.IDENTITY_DRIFT,
        "envelope/profile",
    )
    require(envelope["fallback"] is False, Status.SILENT_FALLBACK, "fallback")
    require(envelope["route"] in ("recorded", "natural"), Status.SCHEMA_MISMATCH, "route")
    require(
        bool(envelope["provenance"])
        and envelope["provenance"].get("kind") in ("recorded", "actual"),
        Status.MISSING_PROVENANCE,
        "provenance",
    )
    provenance = envelope["provenance"]
    if provenance["kind"] == "recorded":
        fingerprint = provenance.get("artifact_checksum")
    else:
        require(
            all(
                isinstance(provenance.get(k), str) and bool(provenance[k])
                for k in ("backend", "python", "torch", "byteorder")
            ),
            Status.MISSING_PROVENANCE,
            "actual runtime readback fields",
        )
        fingerprint = provenance.get("implementation")
    require(
        isinstance(fingerprint, str)
        and len(fingerprint) == 64
        and all(c in "0123456789abcdef" for c in fingerprint),
        Status.MISSING_PROVENANCE,
        "source/recording fingerprint",
    )
    validate_runtime_policy(envelope["runtime_policy"])
    validate_state_sequence(envelope["states"], envelope["identity"])


@schema_guard
def compare_envelopes(reference: dict, candidate: dict, *, require_natural: bool = False) -> dict:
    """No attention attribution is produced until *all* identity/state gates pass."""
    for key in ("schema", "foundation_abi", "case_id", "identity", "profile"):
        require(reference[key] == candidate[key], Status.IDENTITY_DRIFT, key)
    validate_envelope(reference)
    validate_envelope(candidate)
    require(
        len(reference["states"]) == len(candidate["states"]),
        Status.STATE_BYTES_MISMATCH,
        "token boundary count",
    )
    for left, right in zip(reference["states"], candidate["states"], strict=False):
        compare_snapshots(left, right)
    if require_natural:
        require(candidate["route"] == "natural", Status.NATURAL_ROUTE_MISMATCH, "Natural Route")
    require(
        canonical(reference["boundaries"]) == canonical(candidate["boundaries"]),
        Status.BYTE_MISMATCH,
        "M2 boundary bytes",
    )
    return {
        "status": Status.PASS.value,
        "state_status": Status.PASS.value,
        "mode": candidate["mode"],
        "scope": candidate["profile"],
    }


class RecordedProvider:
    """Immutable recordings; selection never substitutes another case or mode."""

    def __init__(self, envelopes: list[dict]):
        self._records = {}
        for envelope in envelopes:
            validate_envelope(envelope)
            key = envelope["case_id"], envelope["mode"]
            require(key not in self._records, Status.SCHEMA_MISMATCH, "duplicate recording")
            self._records[key] = copy.deepcopy(envelope)

    @classmethod
    def from_artifact(cls, directory) -> RecordedProvider:
        from .artifact import read_artifact

        payload, _ = read_artifact(directory)
        checksum = digest(payload)
        envelopes = []
        for layer, sequence in payload["sequences"].items():
            for mode in MODES:
                envelopes.append(
                    {
                        "schema": SCHEMA_VERSION,
                        "foundation_abi": ABI_VERSION,
                        "identity": sequence["snapshots"][0]["identity"],
                        "profile": REFERENCE_PROFILE,
                        "case_id": f"P2-F-LAYER-{layer.lower()}.v1",
                        "mode": mode,
                        "fallback": False,
                        "route": "recorded",
                        "provenance": {
                            "kind": "recorded",
                            "artifact_checksum": checksum,
                            "producer": payload["provenance"],
                        },
                        "runtime_policy": payload["runtime_policy"],
                        "states": sequence["snapshots"],
                        # Independent catalog, NOT a fabricated full-chain execution.
                        "boundaries": {
                            "independent_recordings": {
                                name: record["checksum"]
                                for name, record in payload["recordings"].items()
                            }
                        },
                    }
                )
        return cls(envelopes)

    def describe(self) -> dict:
        return {
            "contract": manifest(),
            "profile": REFERENCE_PROFILE,
            "kind": "recorded",
            "capabilities": sorted([list(k) for k in self._records]),
        }

    def run(self, case_id: str, mode: str) -> dict:
        key = case_id, mode
        require(key in self._records, Status.UNSUPPORTED_CAPABILITY, f"{case_id}/{mode}")
        return copy.deepcopy(self._records[key])


class ProviderRegistry:
    """Explicit routes only. No exceptions swallowed and no recorded fallback."""

    def __init__(self):
        self._providers: dict[str, Provider] = {}

    def register(self, route: str, provider: Provider) -> None:
        require(
            route in ("recorded", "live") and route not in self._providers,
            Status.SCHEMA_MISMATCH,
            "route registration",
        )
        description = provider.describe()
        check_compatibility(description["contract"])
        require(
            description["profile"] == REFERENCE_PROFILE,
            Status.UNSUPPORTED_CAPABILITY,
            "unregistered comparison profile",
        )
        require(description["kind"] == route, Status.SILENT_FALLBACK, "provider kind")
        self._providers[route] = provider

    def run(self, route: str, case_id: str, mode: str) -> dict:
        require(
            route in self._providers, Status.UNSUPPORTED_CAPABILITY, f"missing {route} provider"
        )
        envelope = self._providers[route].run(case_id, mode)
        validate_envelope(envelope)
        require(
            envelope["case_id"] == case_id and envelope["mode"] == mode,
            Status.IDENTITY_DRIFT,
            "provider returned different case/mode",
        )
        require(
            envelope["route"] == ("natural" if route == "live" else "recorded"),
            Status.NATURAL_ROUTE_MISMATCH,
            "provider route",
        )
        if route == "live":
            require(
                envelope["provenance"]["kind"] == "actual",
                Status.MISSING_PROVENANCE,
                "live needs actual readback",
            )
        return envelope
