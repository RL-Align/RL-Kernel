# SPDX-License-Identifier: Apache-2.0
"""Input-bound recorded provider and explicit candidate replacement interface."""

import copy
from dataclasses import asdict
from typing import Protocol

from .contract import (
    CombinePlan,
    Context,
    PROFILE,
    OPERATORS,
    SavedForward,
    TRACE_SCHEMA,
    check_compatibility,
    digest,
    exact_keys,
    manifest,
    require,
)
from .recordings import verify_recordings


class Provider(Protocol):
    def describe(self) -> dict: ...

    def run(self, operator: str, case_id: str, inputs: dict) -> dict: ...


def input_plan(inputs):
    context = Context(**inputs["context"])
    if "plan" in inputs:
        plan = CombinePlan.from_dict(inputs["plan"])
        plan.validate(context)
    else:
        plan = SavedForward(**inputs["saved"]).restore(context, inputs["expected_fingerprint"])
        require(
            inputs["shared_boundary"] == plan.gradient_boundary,
            "GRADIENT_BOUNDARY_MISMATCH",
            "saved gradient identity",
        )
    return plan


def validate_envelope(envelope, *, check_bytes=True):
    exact_keys(
        envelope,
        (
            "schema",
            "profile",
            "operator",
            "case_id",
            "input_sha256",
            "plan_fingerprint",
            "order_hash",
            "kind",
            "fallback",
            "provenance",
            "stages",
            "boundary_hashes",
            "producer_verdict",
            "context",
            "boundary",
            "phase",
        ),
        "TraceEnvelope",
    )
    require(
        envelope["schema"] == TRACE_SCHEMA and envelope["profile"] == PROFILE,
        "SCHEMA_MISMATCH",
        "envelope schema/profile",
    )
    require(envelope["fallback"] is False, "SILENT_FALLBACK", "fallback")
    require(envelope["kind"] in ("recorded", "live"), "SCHEMA_MISMATCH", "provider kind")
    spec = next((op for op in OPERATORS if op["name"] == envelope["operator"]), None)
    require(spec is not None, "UNSUPPORTED_CAPABILITY", "operator")
    require(
        envelope["boundary"] == spec["boundary"] and envelope["phase"] == spec["phase"],
        "INVALID_DISCRETE_PLAN",
        "operator event",
    )
    Context(**envelope["context"]).validate()
    for key in ("input_sha256", "plan_fingerprint", "order_hash"):
        value = envelope[key]
        require(
            type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            "SCHEMA_MISMATCH",
            key,
        )
    provenance = envelope["provenance"]
    require(type(provenance) is dict and bool(provenance), "MISSING_PROVENANCE", "provenance")
    if envelope["kind"] == "live":
        require(provenance.get("readback_kind") == "actual", "MISSING_PROVENANCE", "readback")
        require(
            all(
                type(provenance.get(k)) is str and bool(provenance[k])
                for k in ("backend", "device", "implementation_sha256")
            ),
            "MISSING_PROVENANCE",
            "live fields",
        )
    else:
        require(
            type(provenance.get("artifact_sha256")) is str
            and len(provenance["artifact_sha256"]) == 64,
            "MISSING_PROVENANCE",
            "recording fingerprint",
        )
    if not check_bytes:
        return
    require(
        envelope["boundary_hashes"] == {k: digest(v) for k, v in envelope["stages"].items()},
        "CORRUPT_ARTIFACT",
        "boundary hash",
    )
    require(
        envelope["producer_verdict"] == "REFERENCE_BYTES_PASS",
        "BYTE_MISMATCH",
        "producer verdict is not reference pass",
    )


def compare_envelopes(reference, candidate):
    # Identity/discrete metadata precede numerical attribution.
    for value in (reference, candidate):
        validate_envelope(value, check_bytes=False)
    for key in (
        "schema",
        "profile",
        "operator",
        "case_id",
        "input_sha256",
        "plan_fingerprint",
        "order_hash",
        "context",
        "boundary",
        "phase",
    ):
        require(reference[key] == candidate[key], "IDENTITY_DRIFT", key)
    for value in (reference, candidate):
        validate_envelope(value)
    require(reference["stages"] == candidate["stages"], "BYTE_MISMATCH", "boundary bytes")
    return {"status": "REFERENCE_BYTES_PASS", "production_certified": False}


class RecordedProvider:
    def __init__(self, recordings, artifact_sha256):
        verify_recordings(recordings)
        self._records = {(r["operator"], r["case_id"]): copy.deepcopy(r) for r in recordings}
        self._artifact_sha256 = artifact_sha256

    @classmethod
    def from_artifact(cls, directory):
        from .artifact import read_artifact

        payload, _ = read_artifact(directory)
        return cls(payload["operator_recordings"], digest(payload))

    def describe(self):
        return {
            "contract": manifest(),
            "profile": PROFILE,
            "kind": "recorded",
            "capabilities": [list(k) for k in sorted(self._records)],
        }

    def run(self, operator, case_id, inputs):
        key = operator, case_id
        require(key in self._records, "UNSUPPORTED_CAPABILITY", str(key))
        r = self._records[key]
        plan = input_plan(inputs)
        require(digest(inputs) == digest(r["inputs"]), "IDENTITY_DRIFT", "recorded input")
        result = {
            "schema": TRACE_SCHEMA,
            "profile": PROFILE,
            "operator": operator,
            "case_id": case_id,
            "input_sha256": digest(inputs),
            "plan_fingerprint": plan.fingerprint,
            "order_hash": plan.order_hash,
            "kind": "recorded",
            "fallback": False,
            "provenance": {"artifact_sha256": self._artifact_sha256},
            "stages": copy.deepcopy(r["expected"]),
            "boundary_hashes": {k: digest(v) for k, v in r["expected"].items()},
            "producer_verdict": "REFERENCE_BYTES_PASS",
            "context": asdict(plan.context),
            "boundary": r["boundary"],
            "phase": r["phase"],
        }
        validate_envelope(result)
        return result


class ProviderRegistry:
    """Local test seam, not KernelRegistry. Missing live implementations fail closed."""

    def __init__(self):
        self._providers = {}

    def register(self, route, provider):
        require(
            route in ("recorded", "live") and route not in self._providers,
            "SCHEMA_MISMATCH",
            "provider registration",
        )
        description = provider.describe()
        check_compatibility(description["contract"])
        require(description["profile"] == PROFILE, "UNSUPPORTED_CAPABILITY", "profile")
        require(description["kind"] == route, "SILENT_FALLBACK", "provider kind")
        self._providers[route] = provider

    def run(self, route, operator, case_id, inputs):
        require(route in self._providers, "UNSUPPORTED_CAPABILITY", "missing " + route)
        plan = input_plan(inputs)
        result = self._providers[route].run(operator, case_id, copy.deepcopy(inputs))
        validate_envelope(result, check_bytes=False)
        require(result["kind"] == route, "SILENT_FALLBACK", "returned provider kind")
        for key, expected in {
            "operator": operator,
            "case_id": case_id,
            "input_sha256": digest(inputs),
            "plan_fingerprint": plan.fingerprint,
            "order_hash": plan.order_hash,
            "context": asdict(plan.context),
        }.items():
            require(result[key] == expected, "IDENTITY_DRIFT", key)
        validate_envelope(result)
        return result
