"""Executable sanitized negative fixtures, shared by CLI and tests."""

import struct
from copy import deepcopy
from dataclasses import replace

import numpy as np

from .assembler import validate_case
from .checker import (
    check_actual_provenance,
    check_anchor,
    compare_recordings,
    validate_plan,
    validate_recording,
)
from .contract import SAVED_ROUTE_SCHEMA, P3Error, P3OpCtxHost, P3Verdict, SavedRouteSealedV1
from .fixtures import catalog
from .oracle import hash_route_fwd, learned_route_fwd, stable_topk6
from .provider import readback_verdict, validate_saved
from .recordings import record_case


def run_negative_fixtures():
    case = catalog()[0]
    recording = record_case(case)
    bundle = recording["bundle"]
    saved = SavedRouteSealedV1(**deepcopy(recording["saved"]["route"]))
    ctx = P3OpCtxHost(
        "p3-synthetic",
        "recorded",
        0,
        case.row_active.copy(),
        bundle["RoutePlan"]["route_artifact_fingerprint"],
    )
    fixtures = []

    def capture(name, expected, fn):
        try:
            actual = fn()
        except P3Error as exc:
            actual = exc.verdict
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected.name}, got {actual}")
        fixtures.append(
            {
                "fixture_id": name,
                "expected": expected.name,
                "actual": actual.name,
                "result": "EXPECTED_REJECTION",
            }
        )

    capture("missing-miles-anchor", P3Verdict.MISSING_PROVENANCE, check_anchor)
    capture(
        "wrong-layer-xor",
        P3Verdict.INVALID_DISCRETE_PLAN,
        lambda: validate_case(replace(case, absolute_layer=0)),
    )
    capture(
        "duplicate-global-token",
        P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING,
        lambda: validate_case(
            replace(case, global_token_id=np.array([101, 101, 305, -1], np.int64))
        ),
    )
    capture(
        "local-shard-topk",
        P3Verdict.FORBIDDEN_LOCAL_SHARD_TOPK,
        lambda: stable_topk6(np.zeros((1, 128), np.float32)),
    )
    capture(
        "topk-nan",
        P3Verdict.NON_FINITE,
        lambda: stable_topk6(np.full((1, 256), np.nan, np.float32)),
    )
    s = np.ones((4, 256), np.float32)
    capture(
        "table-index-out-of-range",
        P3Verdict.HASH_TABLE_INDEX_OUT_OF_RANGE,
        lambda: hash_route_fwd(np.full(4, 4, np.int64), s, case.table),
    )
    capture(
        "table-sentinel-invalid",
        P3Verdict.HASH_TABLE_MISMATCH,
        lambda: hash_route_fwd(case.input_token_id, s, np.full((4, 6), 256, np.int32)),
    )
    capture(
        "bias-wrong-dtype",
        P3Verdict.SCHEMA_MISMATCH,
        lambda: learned_route_fwd(s, case.bias.astype(np.float16)),
    )
    capture(
        "raw-saved",
        P3Verdict.SCHEMA_MISMATCH,
        lambda: validate_saved(saved.payload, SAVED_ROUTE_SCHEMA, ctx),
    )
    old = deepcopy(saved)
    old.header["operator_abi"] = "p3-op-abi.v3"
    capture(
        "old-saved-abi",
        P3Verdict.SCHEMA_MISMATCH,
        lambda: validate_saved(old, SAVED_ROUTE_SCHEMA, ctx),
    )
    drift = deepcopy(saved)
    drift.header["route_artifact_fingerprint"] = "ab" * 32
    capture(
        "saved-identity-drift",
        P3Verdict.IDENTITY_DRIFT,
        lambda: validate_saved(drift, SAVED_ROUTE_SCHEMA, ctx),
    )
    flipped = deepcopy(saved)
    flipped.payload["p"].view(np.uint32)[0, 0] ^= 1
    capture(
        "saved-bitflip",
        P3Verdict.CORRUPT_ARTIFACT,
        lambda: validate_saved(flipped, SAVED_ROUTE_SCHEMA, ctx),
    )
    padding_flip = deepcopy(saved)
    padding_flip.payload["p"].view(np.uint32)[3, 0] ^= 1
    capture(
        "padding-saved-bitflip",
        P3Verdict.CORRUPT_ARTIFACT,
        lambda: validate_saved(padding_flip, SAVED_ROUTE_SCHEMA, ctx),
    )
    schema = deepcopy(bundle)
    schema["RoutePlan"]["core"][0]["core_schema_version"] = "RoutePlanCore.v0"
    capture("old-core-schema", P3Verdict.SCHEMA_MISMATCH, lambda: validate_plan(schema))
    slots = deepcopy(bundle)
    slots["RoutePlan"]["core"][1]["topk_index"] = 0
    capture("duplicate-slot", P3Verdict.TOPK_ORDER_MISMATCH, lambda: validate_plan(slots))
    padding = deepcopy(bundle)
    padding["RoutePlan"]["core"][-1]["route_weight"] = np.float32(-0.0)
    capture(
        "padding-negative-zero", P3Verdict.INVALID_DISCRETE_PLAN, lambda: validate_plan(padding)
    )
    for status in (0, 3, -1, 49):
        capture(
            f"illegal-device-status-{status}",
            P3Verdict.CORRUPT_ARTIFACT,
            lambda status=status: readback_verdict(struct.pack("<iiQ", status, 0, 11), 11),
        )
    capture(
        "modified-status-reserved",
        P3Verdict.CORRUPT_ARTIFACT,
        lambda: readback_verdict(struct.pack("<iiQ", 0x7FFFFFFF, 1, 11), 11),
    )
    capture(
        "stale-invocation-echo",
        P3Verdict.INCOMPLETE_ARTIFACT,
        lambda: readback_verdict(struct.pack("<iiQ", 0x7FFFFFFF, 0, 10), 11),
    )
    capture(
        "silent-fallback",
        P3Verdict.SILENT_FALLBACK,
        lambda: check_actual_provenance(
            {
                "requested_backend": "cuda",
                "actual_backend": "cpu",
                "fast_math": False,
                "fallback_reason": None,
            }
        ),
    )
    capture(
        "missing-actual-provenance",
        P3Verdict.MISSING_PROVENANCE,
        lambda: check_actual_provenance({"requested_backend": "cuda"}),
    )
    minimal = {
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "fast_math": False,
        "fallback_reason": None,
    }
    capture(
        "missing-kernel-build",
        P3Verdict.MISSING_PROVENANCE,
        lambda: check_actual_provenance(minimal),
    )
    failed = deepcopy(recording)
    failed["operators"]["learned_route_fwd"]["verdict"] = "NON_FINITE"
    capture(
        "failed-operator", P3Verdict.UPSTREAM_EVIDENCE_MISSING, lambda: validate_recording(failed)
    )
    hash_rec = record_case(catalog()[4])
    uniform = deepcopy(hash_rec)
    uniform["operators"]["hash_route_fwd"]["payload"]["weights"][:] = np.float32(0.25)
    capture(
        "hash-no-gate-uniform-weight",
        P3Verdict.ROUTE_WEIGHT_BYTES_MISMATCH,
        lambda: P3Verdict[compare_recordings(hash_rec, uniform)["verdict"]],
    )
    biased = deepcopy(recording)
    route = biased["operators"]["learned_route_fwd"]["payload"]
    from .oracle import normalize

    route["weights"] = normalize(
        route["ids"], biased["operators"]["router_sqrt_softplus_fwd"]["payload"]["s"] + case.bias
    )["weights"]
    capture(
        "inplace-bias-weight",
        P3Verdict.ROUTE_WEIGHT_BYTES_MISMATCH,
        lambda: P3Verdict[compare_recordings(recording, biased)["verdict"]],
    )
    finite_capacity = deepcopy(bundle)
    for row in finite_capacity["RoutePlan"]["core"][:6]:
        row["capacity"] = 1
    capture(
        "unsupported-finite-capacity",
        P3Verdict.INVALID_DISCRETE_PLAN,
        lambda: validate_plan(finite_capacity),
    )
    from .boundary import consume

    capture(
        "unapproved-foundation-binding",
        P3Verdict.UPSTREAM_CONTRACT_MISMATCH,
        lambda: consume(bundle, "P4", expected_identity=case.identity(), require_foundation=True),
    )
    return {"schema_version": "p3-negative-fixtures.v2", "fixtures": fixtures}
