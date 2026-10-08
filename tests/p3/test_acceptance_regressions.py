"""Regression coverage for the nine failures independently reproduced in the v22 audit.

Run from the repository root with PYTHONPATH=. These do not edit implementation
or existing evidence; all mutated artifacts are created beneath pytest tmp_path.
"""

import hashlib
import json
import os
from copy import deepcopy

import numpy as np
import pytest

from rl_engine.p3.artifact import create, verify
from rl_engine.p3.assembler import combine_seed
from rl_engine.p3.checker import (
    check_actual_provenance,
    compare_plans,
    compare_recordings,
    validate_events,
    validate_plan,
)
from rl_engine.p3.contract import ENVELOPE_FIELDS, P3Error, P3OpCtxHost, P3Verdict
from rl_engine.p3.fixtures import catalog
from rl_engine.p3.provider import RecordedProvider
from rl_engine.p3.recordings import record_case
from rl_engine.p3.serialization import encode, fingerprint


@pytest.fixture(scope="module")
def recording():
    return record_case(catalog()[0])


def test_missing_kernel_build_provenance_is_rejected():
    minimal = {
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "fast_math": False,
        "fallback_reason": None,
    }
    with pytest.raises(P3Error):
        check_actual_provenance(minimal)


def test_failed_operator_cannot_produce_case_pass(recording):
    changed = deepcopy(recording)
    changed["operators"]["learned_route_fwd"]["verdict"] = "NON_FINITE"
    result = compare_recordings(recording, changed)
    assert result["verdict"] != "CASE_PASS", result


def test_old_core_schema_cannot_produce_case_pass(recording):
    changed = deepcopy(recording)
    changed["bundle"]["RoutePlan"]["core"][0]["core_schema_version"] = "RoutePlanCore.v0"
    result = compare_recordings(recording, changed)
    assert result["verdict"] != "CASE_PASS", result


def test_boundary_producer_schema_is_checked(recording):
    events = deepcopy(recording["events"])
    events[0]["producer_schema"] = "unknown-old-producer.v0"
    with pytest.raises(P3Error):
        validate_events(events)


def test_nonfirst_slot_model_drift_is_rejected(recording):
    original = recording["bundle"]
    changed = deepcopy(original)
    plan = changed["RoutePlan"]
    plan["core"][1]["selection_source"] = "tid2eid.table_slot"
    plan["core"][1]["bias_fingerprint"] = "ab" * 32
    # A producer can publish an internally consistent file checksum. The checker
    # must still reject Core fields contradicting the case and other token slots.
    envelope_body = [{k: env[k] for k in ENVELOPE_FIELDS} for env in plan["envelope"]]
    digest = fingerprint(
        {"case_id": plan["identity"]["case_id"], "core": plan["core"], "envelope": envelope_body}
    )
    plan["route_artifact_fingerprint"] = digest
    for env in plan["envelope"]:
        env["route_artifact_fingerprint"] = digest
    changed["CombinePlanSeed"] = combine_seed(plan)
    with pytest.raises(P3Error):
        validate_plan(changed)
        compare_plans(original, changed)


def test_recorded_provider_refuses_failed_operator(recording):
    changed = deepcopy(recording)
    operator = changed["operators"]["router_sqrt_softplus_fwd"]
    operator["verdict"] = "NON_FINITE"
    ctx = P3OpCtxHost("audit", "recorded", 0, changed["row_active"].copy())
    result = RecordedProvider(changed).router_sqrt_softplus_fwd(ctx, *operator["inputs"])
    assert result.verdict != P3Verdict.PASS, result.verdict


def test_recorded_zero_active_still_checks_input_schema():
    case = catalog()[6]
    provider = RecordedProvider(record_case(case))
    ctx = P3OpCtxHost("audit", "recorded", 0, case.row_active.copy())
    invalid_logits = np.zeros((len(case.z), 128), dtype=np.float64)
    result = provider.router_sqrt_softplus_fwd(ctx, invalid_logits, case.round_policy)
    assert result.verdict == P3Verdict.SCHEMA_MISMATCH, result.verdict


def test_verifier_rejects_unsupported_certification_claim(tmp_path):
    path = tmp_path / "claim"
    data = create(path)
    gate = next(e for e in data["evidence_matrix"] if e["gate"] == "Miles anchor / recorded L3b")
    gate["verdict"] = "PASS"
    artifact_file = path / "artifact.json"
    artifact_file.write_text(json.dumps(encode(data), indent=2) + "\n")
    seal_file = path / "seal.json"
    seal = json.loads(seal_file.read_text())
    seal["files"]["artifact.json"] = hashlib.sha256(artifact_file.read_bytes()).hexdigest()
    seal_file.write_text(json.dumps(seal, indent=2) + "\n")
    with pytest.raises(P3Error):
        verify(path)


@pytest.mark.skipif(os.getenv("P3_RUN_GPU") != "1", reason="requires actual H100")
def test_cuda_runner_does_not_reuse_invocations_between_calls(tmp_path):
    from rl_engine.p3.__main__ import cuda_slice

    first = cuda_slice(state_dir=tmp_path)
    second = cuda_slice(state_dir=tmp_path)
    a = {r["provenance"]["invocation_id"] for r in first}
    b = {r["provenance"]["invocation_id"] for r in second}
    assert a.isdisjoint(b), {
        "reused_ids": sorted(a & b),
        "first_records": len(first),
        "second_records": len(second),
    }
