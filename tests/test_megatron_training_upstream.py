# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P2b: recorded P1–P6 TraceEnvelope correlation, forwarded verbatim."""

import copy

import pytest

from rl_engine.alignment.cross_config.adapters.megatron_training import T05ContractError
from rl_engine.alignment.cross_config.adapters.megatron_training_upstream import (
    UpstreamCorrelationKey,
    UpstreamRequirement,
    correlate_upstream,
)

pytestmark = pytest.mark.unit

KEY = UpstreamCorrelationKey(
    case_id="case-1", run_id="run-a", rank="tp0", global_tokens=(("s0", 0), ("s0", 1), ("s1", 3))
)
P3 = UpstreamRequirement("P3", "P3.route.v1", "M4", "3", 2, "forward", "moe_ffn")
P1 = UpstreamRequirement("P1", "P1.mhc.v1", "M0.pre", "1", 0, "forward", "attention")


def _envelope(requirement, **overrides):
    payload = {
        "producer": requirement.producer,
        "upstream_contract_id": requirement.upstream_contract_id,
        "producer_schema_version": requirement.pinned_schema_version,
        "boundary": requirement.boundary,
        "run_id": "run-a",
        "case_id": "case-1",
        "absolute_layer": requirement.absolute_layer,
        "sublayer_site": requirement.sublayer_site,
        "pass_name": requirement.pass_name,
        "event_index": 7,
        "global_token_ids": [["s0", 0], ["s0", 1], ["s1", 3], ["s1", 4]],
        "rank": "tp0",
        "requested_backend": "strict",
        "actual_backend": "strict",
        "kernel_build": "build-1",
        "comparison_mode": "byte_exact",
        "producer_verdict": "PASS",
        "fixture_checksum": "abc",
        "producer_private": {"route_weight_sha256": "deadbeef"},
    }
    payload.update(overrides)
    return payload


def _expect(status, field, envelopes, requirements=(P3,)):
    with pytest.raises(T05ContractError) as info:
        correlate_upstream(KEY, requirements, envelopes)
    assert info.value.status == status, str(info.value)
    assert info.value.field_name.startswith(field), str(info.value)


def test_envelopes_are_forwarded_verbatim_and_ordered_by_runtime_events():
    p3 = _envelope(P3, producer_verdict="BYTE_MISMATCH")
    p1 = _envelope(P1)
    evidence = correlate_upstream(KEY, (P3, P1), [p3, p1])
    assert [item.requirement.producer for item in evidence.forwarded] == ["P1", "P3"]
    assert evidence.forwarded[1].payload == p3, "producer verdict must not be re-judged"
    assert evidence.forwarded[1].payload["producer_private"] == {"route_weight_sha256": "deadbeef"}
    p3["producer_verdict"] = "PASS"
    assert (
        evidence.forwarded[1].payload["producer_verdict"] == "BYTE_MISMATCH"
    ), "copied, not aliased"
    again = correlate_upstream(KEY, (P1, P3), [p1, _envelope(P3, producer_verdict="BYTE_MISMATCH")])
    assert again.fingerprint == evidence.fingerprint


def test_missing_boundary_is_upstream_evidence_missing():
    _expect("UPSTREAM_EVIDENCE_MISSING", "P3:M4", [_envelope(P1)])


def test_no_requirements_is_missing():
    with pytest.raises(T05ContractError) as info:
        correlate_upstream(KEY, (), [])
    assert info.value.status == "UPSTREAM_EVIDENCE_MISSING"


@pytest.mark.parametrize(
    "field", ["fixture_checksum", "kernel_build", "actual_backend", "event_index"]
)
def test_missing_required_field(field):
    payload = _envelope(P3)
    del payload[field]
    _expect("UPSTREAM_EVIDENCE_MISSING", f"P3:M4.{field}", [payload])


def test_missing_verdict_has_its_own_status():
    _expect(
        "UPSTREAM_VERDICT_MISSING", "P3:M4.producer_verdict", [_envelope(P3, producer_verdict="")]
    )


def test_schema_pin_mismatch():
    _expect(
        "UPSTREAM_CONTRACT_MISMATCH",
        "P3:M4.producer_schema_version",
        [_envelope(P3, producer_schema_version="4")],
    )


def test_unknown_comparison_mode():
    _expect(
        "UPSTREAM_CONTRACT_MISMATCH",
        "P3:M4.comparison_mode",
        [_envelope(P3, comparison_mode="ulp")],
    )


def test_tolerance_mode_needs_tolerance_contract():
    payload = _envelope(P3, comparison_mode="upstream_contract")
    _expect("UPSTREAM_EVIDENCE_MISSING", "P3:M4.tolerance_contract_version", [payload])
    ok = _envelope(P3, comparison_mode="upstream_contract", tolerance_contract_version="t1")
    correlate_upstream(KEY, (P3,), [ok])


@pytest.mark.parametrize("field, value", [("run_id", "run-b"), ("case_id", "case-9")])
def test_wrong_run_or_case_is_stale(field, value):
    _expect("STALE_RUN_METADATA", f"P3:M4.{field}", [_envelope(P3, **{field: value})])


def test_wrong_rank_is_identity_drift():
    _expect("IDENTITY_DRIFT", "P3:M4.rank", [_envelope(P3, rank="tp1")])


def test_uncovered_tokens_are_identity_drift():
    payload = _envelope(P3, global_token_ids=[["s0", 0], ["s0", 1]])
    _expect("IDENTITY_DRIFT", "P3:M4.global_token_ids", [payload])


def test_duplicate_envelopes_for_one_boundary_are_rejected():
    _expect("UPSTREAM_CONTRACT_MISMATCH", "P3:M4", [_envelope(P3), _envelope(P3, event_index=8)])


def test_other_layer_or_pass_does_not_satisfy_requirement():
    _expect("UPSTREAM_EVIDENCE_MISSING", "P3:M4", [_envelope(P3, absolute_layer=3)])
    _expect("UPSTREAM_EVIDENCE_MISSING", "P3:M4", [_envelope(P3, pass_name="backward")])


def test_required_fields_are_injectable():
    payload = _envelope(P3)
    del payload["kernel_build"]
    fields = (
        "producer_verdict",
        "producer_schema_version",
        "comparison_mode",
        "run_id",
        "case_id",
        "rank",
        "global_token_ids",
        "absolute_layer",
        "sublayer_site",
        "pass_name",
        "event_index",
    )
    evidence = correlate_upstream(KEY, (P3,), [payload], required_fields=fields)
    assert evidence.required_fields == fields


def test_key_rejects_duplicate_tokens():
    with pytest.raises(T05ContractError) as info:
        UpstreamCorrelationKey("c", "r", "tp0", (("s0", 0), ("s0", 0)))
    assert info.value.status == "AMBIGUOUS_GLOBAL_TOKEN_MAPPING"


def test_input_envelopes_are_not_mutated():
    payload = _envelope(P3)
    before = copy.deepcopy(payload)
    correlate_upstream(KEY, (P3,), [payload])
    assert payload == before
