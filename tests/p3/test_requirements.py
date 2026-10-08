"""Named regression evidence for authenticated KLR #1/#5/#41–#46 requirements."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import pytest

from rl_engine.p3 import bitmath, oracle
from rl_engine.p3.boundary import consume
from rl_engine.p3.checker import compare_recordings, validate_plan, validate_recording
from rl_engine.p3.contract import ROUND_POLICIES, P3Error, P3Verdict
from rl_engine.p3.diagnostics import ulp_distance
from rl_engine.p3.fixtures import catalog
from rl_engine.p3.provider import InvocationAllocator
from rl_engine.p3.recordings import record_case


def case_named(name, policy):
    return next(c for c in catalog() if c.case_id == name + "." + policy)


@pytest.mark.parametrize("policy", ROUND_POLICIES)
def test_true_selection_cutoff_changes_under_bias_bf16_round(policy):
    case = case_named("cutoff-6-7-bias-ulp", policy)
    rec = record_case(case)
    route = rec["operators"]["learned_route_fwd"]["payload"]
    score = rec["operators"]["router_sqrt_softplus_fwd"]["payload"]["s"]
    rounded = oracle.learned_route_fwd(score, bitmath.apply(case.bias, "bf16"))
    assert np.array_equal(route["ids"][0], [0, 1, 2, 3, 4, 6])
    assert np.array_equal(rounded["ids"][0], [0, 1, 2, 3, 4, 5])
    assert np.all(rec["torch_paired"]["cutoff_6_7_margin"] > 0)
    assert np.all(rec["torch_paired"]["cutoff_6_7_ulp"] <= 2)
    # Both paths gather the identical pre-bias score bytes.
    assert np.array_equal(route["saved_route"]["a"], rounded["saved_route"]["a"])


@pytest.mark.parametrize("policy", ROUND_POLICIES)
def test_eighth_ninth_bias_precision_has_a_distinct_named_case(policy):
    case = case_named("bias-precision-8-9", policy)
    score = oracle.router_sqrt_softplus_fwd(case.z, policy)["s"]
    q = score + case.bias
    rounded_q = score + bitmath.apply(case.bias, "bf16")
    order = np.argsort(-q[0], kind="stable")
    rounded = np.argsort(-rounded_q[0], kind="stable")
    assert list(order[7:9]) == [8, 7]
    assert list(rounded[7:9]) == [7, 8]
    assert list(order[:6]) == list(rounded[:6])  # 8/9 is not the Top-6 cutoff.


@pytest.mark.parametrize("policy", ROUND_POLICIES)
def test_dropless_hotspot_retains_all_active_slots_and_canonical_padding(policy):
    rec = record_case(case_named("dropless-hotspot-padding", policy))
    plan = rec["bundle"]["RoutePlan"]
    valid = [r for r in plan["core"] if r["valid"]]
    assert len(valid) == 18
    assert all(r["capacity"] == -1 for r in valid)
    assert len([r for r in valid if r["logical_expert_id"] == 0]) == 3
    padding = [r for r in plan["core"] if not r["valid"]]
    assert len(padding) == 6
    assert all(r["invalid_reason"] == "padding" for r in padding)
    assert all(r["route_weight"].view(np.uint32) == 0 for r in padding)


@pytest.mark.parametrize(
    "fault", ["hash-uniform-no-gate", "learned-inplace-bias", "selection-gradient"]
)
def test_wrong_implementations_have_named_rejections(fault):
    case = (
        catalog()[4] if fault.startswith("hash") else case_named("pre-bias-weight", "fp32_direct")
    )
    base = record_case(case)
    changed = deepcopy(base)
    branch = case.router_mode
    if fault == "hash-uniform-no-gate":
        changed["operators"]["hash_route_fwd"]["payload"]["weights"][:] = np.float32(0.25)
        expected = "ROUTE_WEIGHT_BYTES_MISMATCH"
    elif fault == "learned-inplace-bias":
        route = changed["operators"]["learned_route_fwd"]["payload"]
        s = changed["operators"]["router_sqrt_softplus_fwd"]["payload"]["s"]
        route["weights"] = oracle.normalize(route["ids"], s + case.bias)["weights"]
        expected = "ROUTE_WEIGHT_BYTES_MISMATCH"
    else:
        ids = changed["operators"][branch + "_route_fwd"]["payload"]["ids"][0]
        unselected = next(e for e in range(256) if e not in ids)
        changed["operators"][branch + "_route_bwd"]["payload"]["ds"] = changed["operators"][
            branch + "_route_bwd"
        ]["payload"]["ds"].copy()
        changed["operators"][branch + "_route_bwd"]["payload"]["ds"][0, unselected] = 1
        expected = "GRADIENT_BYTES_MISMATCH"
    assert compare_recordings(base, changed)["verdict"] == expected


@pytest.mark.parametrize("consumer", ["P4", "P6", "P7"])
def test_consumer_contract_is_independently_readable_and_rejects_drift(consumer):
    from rl_engine.p3.serialization import decode, encode

    bundle = decode(json.loads(json.dumps(encode(record_case(catalog()[0])["bundle"]))))
    identity = bundle["RoutePlan"]["identity"]
    assert consume(bundle, consumer, expected_identity=identity)
    for kwargs, verdict in [
        ({"expected_boundary": "p3-local-boundary.v0"}, P3Verdict.SCHEMA_MISMATCH),
        ({"capacity_policy": "finite-capacity"}, P3Verdict.UNSUPPORTED_CAPABILITY),
        ({"require_foundation": True}, P3Verdict.UPSTREAM_CONTRACT_MISMATCH),
        ({"expected_identity": {**identity, "weight_id": "other"}}, P3Verdict.IDENTITY_DRIFT),
    ]:
        with pytest.raises(P3Error) as exc:
            consume(bundle, consumer, **{"expected_identity": identity, **kwargs})
        assert exc.value.verdict == verdict


def test_ulp_diagnostics_preserve_strict_signed_zero_and_one_ulp_semantics():
    a = np.array([-1, -0.0, 0.0, 1], np.float32)
    b = np.nextafter(a, np.float32(np.inf))
    assert np.array_equal(ulp_distance(a, b), [1, 1, 1, 1])
    assert ulp_distance(np.array([-0.0], np.float32), np.array([0.0], np.float32))[0] == 0
    assert np.array([-0.0], np.float32).tobytes() != np.array([0.0], np.float32).tobytes()


def test_persistent_attempt_allocation_concurrent_restart_and_failure(tmp_path, monkeypatch):
    path = tmp_path / "invocations.json"
    with ThreadPoolExecutor(max_workers=8) as pool:
        allocators = list(
            pool.map(lambda _: InvocationAllocator.new_attempt(path, "run", "engine", 0), range(16))
        )
    assert sorted(a.attempt_id for a in allocators) == list(range(1, 17))
    latest = max(allocators, key=lambda a: a.attempt_id)
    assert latest.reserve() == (16 << 32) | 1
    restarted = InvocationAllocator.new_attempt(path, "run", "engine", 0)
    assert restarted.reserve() == (17 << 32) | 1
    original = path.read_bytes()

    def fail(*args):
        raise OSError("injected durable commit failure")

    monkeypatch.setattr(InvocationAllocator, "_commit", fail)
    with pytest.raises(P3Error) as exc:
        InvocationAllocator.new_attempt(path, "run", "engine", 0)
    assert exc.value.verdict == P3Verdict.CORRUPT_ARTIFACT
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "missing", ["schema_version", "operators", "events", "bundle", "provenance"]
)
def test_malformed_recordings_are_typed_rejections(missing):
    rec = record_case(catalog()[0])
    del rec[missing]
    with pytest.raises(P3Error):
        validate_recording(rec)


def test_finite_capacity_and_corrupt_diagnostics_are_not_silently_accepted():
    rec = record_case(catalog()[0])
    bad = deepcopy(rec["bundle"])
    for row in bad["RoutePlan"]["core"][:6]:
        row["capacity"] = 1
    with pytest.raises(P3Error):
        validate_plan(bad)
    rec["torch_paired"]["max_score_ulp"] += 1
    with pytest.raises(P3Error) as exc:
        validate_recording(rec)
    assert exc.value.verdict == P3Verdict.CORRUPT_ARTIFACT
