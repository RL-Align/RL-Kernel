from dataclasses import replace

import numpy as np
import pytest

from rl_engine.p3 import bitmath, oracle
from rl_engine.p3.assembler import assemble
from rl_engine.p3.checker import compare_plans
from rl_engine.p3.contract import P3Error, P3OpResult, P3Verdict
from rl_engine.p3.fixtures import catalog
from rl_engine.p3.recordings import record_case
from rl_engine.p3.serialization import canonical

CASES = catalog()
ACTIVE_CASES = [c for c in CASES if c.row_active.any()]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
def test_source_recordings_and_zero_active(case):
    recording = record_case(case)
    if not case.row_active.any():
        assert recording["producer_verdict"] == "ZERO_ACTIVE_TOKENS"
        assert not recording["saved"] and not recording["operators"]
        return
    assert len(recording["operators"]) == 5
    assert recording["torch_paired"]["status"] == "RECORDED_DIAGNOSTIC"
    assert recording["torch_paired"]["max_abs_score_error"] < 2e-6
    assert recording["torch_paired"]["ids_match"]
    assert len(recording["bundle"]["CombinePlanSeed"]["tokens"]) == int(case.row_active.sum())


@pytest.mark.parametrize("case", ACTIVE_CASES, ids=lambda c: c.case_id)
def test_repeat_and_dual_recorded_engines(case):
    baseline = record_case(case)["bundle"]
    assert (
        compare_plans(baseline, record_case(case)["bundle"], same_config=True)
        == P3Verdict.CASE_PASS
    )
    alternate = record_case(
        case, engine_id="synthetic-inference", run_id="different-run", attempt_id=2
    )["bundle"]
    assert compare_plans(baseline, alternate) == P3Verdict.CASE_PASS
    assert (
        baseline["RoutePlan"]["route_artifact_fingerprint"]
        != alternate["RoutePlan"]["route_artifact_fingerprint"]
    )


@pytest.mark.parametrize("case", ACTIVE_CASES, ids=lambda c: c.case_id)
def test_batch_pack_padding_per_token_invariant(case):
    baseline = record_case(case)["bundle"]["RoutePlan"]["per_token_fingerprints"]
    reordered = record_case(case.rows([2, 3, 0, 1]))["bundle"]
    assert reordered["RoutePlan"]["per_token_fingerprints"] == baseline
    unpadded = record_case(case.rows([0, 1, 2]))["bundle"]
    assert unpadded["RoutePlan"]["per_token_fingerprints"] == baseline
    for row in range(3):
        single = record_case(case.rows([row]))["bundle"]
        token = str(case.global_token_id[row])
        assert single["RoutePlan"]["per_token_fingerprints"] == {token: baseline[token]}


@pytest.mark.parametrize("case", ACTIVE_CASES, ids=lambda c: c.case_id)
def test_placement_changes_only_envelope(case):
    rec = record_case(case)
    score = rec["operators"]["router_sqrt_softplus_fwd"]["payload"]
    route = rec["operators"][case.router_mode + "_route_fwd"]["payload"]
    provenance = rec["provenance"]
    rotated = assemble(
        case,
        P3OpResult(P3Verdict.PASS, score, provenance),
        P3OpResult(P3Verdict.PASS, route, provenance),
        placement=list(range(128, 256)) + list(range(128)),
        placement_version="rotate.v1",
    )
    assert compare_plans(rec["bundle"], rotated) == P3Verdict.CASE_PASS
    assert canonical(rec["bundle"]["RoutePlan"]["core"]) == canonical(rotated["RoutePlan"]["core"])


@pytest.mark.parametrize("seed", range(5))
def test_topk_independent_naive_total_order(seed):
    rng = np.random.default_rng(seed)
    q = rng.integers(-4, 5, size=(31, 256)).astype(np.float32)
    q[0] = 0
    q[1, ::2] = np.float32(-0.0)
    q[2, 6] = np.nextafter(q[2, 5], np.float32(np.inf))
    expected = np.array(
        [sorted(range(256), key=lambda e: (-float(row[e]), e))[:6] for row in q], np.int32
    )
    assert np.array_equal(oracle.stable_topk6(q), expected)
    assert np.array_equal(oracle.stable_topk6(q[0:1]), [[0, 1, 2, 3, 4, 5]])


def test_hash_original_slots_and_duplicate_gradient_accumulation():
    table = np.array([[9, 1, 9, 2, 1, 9]], np.int32)
    s = np.linspace(0.1, 3, 256, dtype=np.float32)[None]
    fwd = oracle.hash_route_fwd(np.array([0], np.int64), s, table)
    assert np.array_equal(fwd["ids"], table)
    dw = np.array([[1, 2, -1, -3, 5, 0.25]], np.float32)
    bwd = oracle.route_bwd(dw, fwd["saved_route"])
    reference = np.zeros((1, 256), np.float32)
    for slot in range(6):
        reference[0, table[0, slot]] += bwd["da"][0, slot]
    assert canonical(bwd["ds"]) == canonical(reference)
    # Finite difference holds selection fixed and includes repeated expert slots.
    for expert in (1, 2, 9):
        h = np.float32(0.0002)
        plus, minus = s.copy(), s.copy()
        plus[0, expert] += h
        minus[0, expert] -= h
        numeric = np.sum(
            (
                oracle.hash_route_fwd(np.array([0], np.int64), plus, table)["weights"]
                - oracle.hash_route_fwd(np.array([0], np.int64), minus, table)["weights"]
            )
            * dw
        ) / (2 * h)
        assert bwd["ds"][0, expert] == pytest.approx(float(numeric), rel=0.003, abs=0.002)


def test_bias_selection_is_independent_and_weights_pre_bias():
    s = np.ones((1, 256), np.float32)
    bias = np.zeros(256, np.float32)
    bias[200] = 100
    original = s.copy()
    result = oracle.learned_route_fwd(s, bias)
    assert result["ids"][0, 0] == 200
    assert canonical(s) == canonical(original)
    assert np.array_equal(result["saved_route"]["a"], np.ones((1, 6), np.float32))
    assert np.array_equal(result["weights"], np.full((1, 6), 0.25, np.float32))


def test_fixed_tree_is_not_linear_sum():
    a = np.array([[1e20, 1, -1e20, 1, 1, 1]], np.float32)
    expected = ((a[:, 0] + a[:, 1]) + (a[:, 2] + a[:, 3])) + (a[:, 4] + a[:, 5])
    assert canonical(bitmath.sum6(a)) == canonical(expected)
    assert bitmath.sum6(a)[0] == 2


def test_backward_zero_guard_and_selected_underflow():
    saved = {"z_prime": np.full((1, 256), -1000, np.float32), "s": np.zeros((1, 256), np.float32)}
    ds = np.full((1, 256), -0.0, np.float32)
    dz = oracle.router_sqrt_softplus_bwd(ds, saved)["dz"]
    assert not dz.view(np.uint32).any()
    ds[0, 7] = 1
    with pytest.raises(P3Error) as exc:
        oracle.router_sqrt_softplus_bwd(ds, saved)
    assert exc.value.verdict == P3Verdict.NON_FINITE


def test_rounding_threshold_and_fp32_policy_separation():
    z = np.full((1, 256), np.nextafter(np.float32(20), np.float32(21)), np.float32)
    direct = oracle.router_sqrt_softplus_fwd(z, "fp32_direct")
    bf16 = oracle.router_sqrt_softplus_fwd(z, "bf16_round_then_widen")
    assert direct["saved_score"]["z_prime"][0, 0] > 20
    assert bf16["saved_score"]["z_prime"][0, 0] == 20
    assert canonical(direct) != canonical(bf16)


def test_model_identity_is_checked_before_numerics():
    base = record_case(CASES[0])["bundle"]
    other = record_case(replace(CASES[0], checkpoint_id="other-checkpoint"))["bundle"]
    with pytest.raises(P3Error) as exc:
        compare_plans(base, other)
    assert exc.value.verdict == P3Verdict.IDENTITY_DRIFT


def test_host_flush_to_zero_is_rejected():
    import torch

    try:
        if not torch.set_flush_denormal(True):
            pytest.skip("host lacks flush-to-zero control")
        with pytest.raises(P3Error) as exc:
            bitmath.apply(np.array([-100], np.float32), "exp")
        assert exc.value.verdict == P3Verdict.UNSUPPORTED_CAPABILITY
    finally:
        torch.set_flush_denormal(False)
