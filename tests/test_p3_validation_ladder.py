# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for T09 validation ladder (L1/L2/L3a/L3b) + synthetic producer.

Contract refs: §2.4 (semantic/artifact hash), §2.5 (byte gates), §4-T09,
§7 WS1 Gate. Synthetic producer usage is authorized by §5.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from rl_engine.moe.naive_topk6 import K
from rl_engine.moe.p3_verdicts import P3Verdict
from rl_engine.moe.validation.fingerprint import (
    Artifact,
    RouteIdentity,
    RouteRow,
    per_token_semantic_hashes,
    route_artifact_hash,
    route_semantic_hash,
)
from rl_engine.moe.validation.ladder import (
    run_l1_repeat,
    run_l2_invariance,
    run_l3a_oracle,
    run_l3b_dual_engine,
)
from rl_engine.moe.validation.synthetic_producer import (
    make_artifact,
    make_learned_route_rows,
    repack_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ident(**over):
    base = dict(
        checkpoint_id="ckpt-x", weight_id="w-x",
        table_fingerprint="00" * 32, bias_fingerprint="11" * 32,
    )
    base.update(over)
    return RouteIdentity(**base)


def _learned_case(t_rows=4, seed=7):
    rows, meta = make_learned_route_rows("cx", seed=seed, t_rows=t_rows, layers=(3,))
    return rows, meta


def _wrap(rows, ident, **kw):
    kw.setdefault("run_id", "r1")
    kw.setdefault("engine_id", "e1")
    kw.setdefault("attempt_id", 1)
    return make_artifact("cx", rows, ident, **kw)


# --- L1 ------------------------------------------------------------------------

def test_l1_repeat_identical_artifacts_pass():
    rows, _ = _learned_case()
    a = _wrap(rows, _ident())
    b = _wrap(rows, _ident())
    rep = run_l1_repeat("cx", a, b)
    assert rep.passed
    assert rep.extra["artifact_hash"]


def test_l1_repeat_padding_change_fails_artifact_hash():
    """Padding is audited ONLY by L1's artifact gate (§2.5)."""
    rows, _ = _learned_case()
    a = _wrap(rows, _ident())
    b = _wrap(rows, _ident(), padding_rows=2)
    rep = run_l1_repeat("cx", a, b)
    assert not rep.passed
    assert rep.verdict is P3Verdict.ROUTE_ARTIFACT_FINGERPRINT_MISMATCH


def test_l1_attempt_id_change_fails():
    rows, _ = _learned_case()
    a = _wrap(rows, _ident())
    b = make_artifact("cx", rows, _ident(), run_id="r1", engine_id="e1", attempt_id=2)
    rep = run_l1_repeat("cx", a, b)
    assert not rep.passed
    assert rep.verdict is P3Verdict.ROUTE_ARTIFACT_FINGERPRINT_MISMATCH


# --- L2 ------------------------------------------------------------------------

def test_l2_padding_and_repack_are_invariant():
    rows, meta = _learned_case()
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    base = _wrap(rows, ident)
    pert = make_artifact("cx", repack_rows(rows, batch_size=2), ident,
                         run_id="r2", engine_id="miles", attempt_id=2,
                         placement_offset=3, padding_rows=5)
    rep = run_l2_invariance("cx", base, pert, variant="pad+repack")
    assert rep.passed, rep.detail


def test_l2_missing_token_fails_incomplete():
    rows, meta = _learned_case()
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    base = _wrap(rows, ident)
    dropped = [r for r in rows if r.global_token_id != 1]
    pert = _wrap(dropped, ident, padding_rows=6)  # padding can't mask a loss
    rep = run_l2_invariance("cx", base, pert, variant="drop")
    assert not rep.passed
    assert rep.verdict is P3Verdict.INCOMPLETE_ARTIFACT
    assert rep.extra["missing"] == [1]


def test_l2_extra_token_fails_ambiguous_mapping():
    rows, meta = _learned_case()
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    base = _wrap(rows, ident)
    ghost = dataclasses.replace(rows[0], global_token_id=99, input_token_id=99)
    pert = _wrap(rows + [ghost], ident)
    rep = run_l2_invariance("cx", base, pert, variant="ghost")
    assert not rep.passed
    assert rep.verdict is P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING


def test_l2_weight_bit_flip_is_first_mismatch():
    rows, meta = _learned_case()
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    base = _wrap(rows, ident)
    flipped = list(rows)
    flipped[0] = dataclasses.replace(flipped[0], route_weight=flipped[0].route_weight + 1e-6)
    pert = _wrap(flipped, ident)
    rep = run_l2_invariance("cx", base, pert, variant="flip")
    assert not rep.passed
    assert rep.verdict is P3Verdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH
    assert rep.extra["first_token"] == rows[0].global_token_id


# --- L3a -----------------------------------------------------------------------

def test_l3a_byte_exact_passes():
    rows, meta = _learned_case()
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    rep = run_l3a_oracle("cx", ident, rows, list(rows))
    assert rep.passed


def test_l3a_ignores_padding_rows():
    """§2.5: padding is audited only by the same-config artifact gate."""
    rows, meta = _learned_case()
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    pad = RouteRow(global_token_id=-1, input_token_id=-1, absolute_layer=3,
                   router_mode="learned", topk_index=0, logical_expert_id=-1,
                   valid=False, invalid_reason="padding", route_weight=0.0,
                   weight_score=0.0, selection_score=0.0)
    rep = run_l3a_oracle("cx", ident, rows + [pad], rows)
    assert rep.passed


def test_l3a_expert_id_mismatch_reports_topk_order():
    rows, meta = _learned_case()
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    bad = list(rows)
    bad[7] = dataclasses.replace(bad[7], logical_expert_id=(bad[7].logical_expert_id + 1) % 256)
    rep = run_l3a_oracle("cx", ident, rows, bad)
    assert not rep.passed
    assert rep.verdict is P3Verdict.TOPK_ORDER_MISMATCH
    assert rep.first_mismatch is not None
    assert rep.first_mismatch.key.global_token_id == bad[7].global_token_id


def test_l3a_row_count_mismatch_fails_incomplete():
    rows, meta = _learned_case()
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    rep = run_l3a_oracle("cx", ident, rows, rows[:-1])
    assert not rep.passed
    assert rep.verdict is P3Verdict.INCOMPLETE_ARTIFACT


# --- L3b -----------------------------------------------------------------------

def _meta(router_mode="learned", layer=3):
    return {
        "case_id": "cx", "checkpoint_id": "c", "weight_id": "w",
        "absolute_layer": layer, "router_mode": router_mode,
        "table_fingerprint": "00" * 32, "bias_fingerprint": "11" * 32,
        "logit_round_point": "fp32_direct", "tie_break_policy": "p3_canonical",
        "capacity_policy": "dropless_v1",
    }


def test_l3b_identical_engines_pass_four_stages():
    import torch

    from rl_engine.moe.validation.first_mismatch import MismatchKey, TraceEvent

    events = [
        TraceEvent(key=MismatchKey(absolute_layer=3, site="topk",
                                   pass_direction="forward", event_index=i,
                                   global_token_id=i // K, rank=0),
                   payload={"logical_expert_id": i % 256})
        for i in range(12)
    ]
    w = torch.full((12,), 0.25)
    dz = torch.full((12,), 1.0)
    rep = run_l3b_dual_engine("cx", _meta(), dict(_meta()), events, list(events),
                              w, w.clone(), dz, dz.clone())
    assert rep.passed, rep.detail


def test_l3b_identity_drift_halts_walk():
    import torch

    from rl_engine.moe.validation.first_mismatch import MismatchKey, TraceEvent

    events = []
    lhs_meta = _meta()
    rhs_meta = _meta()
    rhs_meta["tie_break_policy"] = "other"
    w = torch.zeros(0)
    rep = run_l3b_dual_engine("cx", lhs_meta, rhs_meta, events, events,
                              w, w.clone(), w, w.clone())
    assert not rep.passed
    assert rep.verdict is P3Verdict.IDENTITY_DRIFT
    assert rep.extra["walked"] == ["identity"]


# --- fingerprints ---------------------------------------------------------------

def test_semantic_hash_excludes_padding_and_envelope():
    rows, meta = _learned_case()
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    a = _wrap(rows, ident)
    b = _wrap(rows, ident, run_id="r9", engine_id="miles", attempt_id=9,
              placement_offset=7, padding_rows=3)
    sa = per_token_semantic_hashes(a.rows, a.identity)
    sb = per_token_semantic_hashes(b.rows, b.identity)
    assert sa == sb
    assert -1 not in sa
    assert route_semantic_hash(a.rows, ident) == route_semantic_hash(b.rows, ident)


def test_artifact_hash_sensitive_to_envelope():
    rows, meta = _learned_case()
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    a = _wrap(rows, ident)
    b = _wrap(rows, ident, attempt_id=2)
    assert route_artifact_hash(a) != route_artifact_hash(b)


def test_minus_zero_vs_plus_zero_distinguished():
    h1 = route_semantic_hash(
        [RouteRow(0, 0, 3, "learned", 0, 1, True, None, 0.0, 0.0, 0.0)], _ident())
    h2 = route_semantic_hash(
        [RouteRow(0, 0, 3, "learned", 0, 1, True, None, -0.0, 0.0, 0.0)], _ident())
    assert h1 != h2


# --- CLI -----------------------------------------------------------------------

@pytest.mark.parametrize("args,expect_pass", [
    (["--cases", "smoke"], True),
    (["--cases", "learned_basic", "--ladders", "L1"], True),
    (["--cases", "hash_basic", "--ladders", "L2,L3a"], True),
    (["--ladders", "L9"], None),           # bad ladder -> usage error 2
    (["--cases", "nope"], None),           # bad case -> usage error 2
])
def test_check_p3_cli(args, expect_pass):
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_p3.py"), *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if expect_pass is None:
        assert proc.returncode == 2
        assert "error:" in proc.stderr
    else:
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "check_p3:" in proc.stdout


def test_check_p3_cli_json_flag():
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_p3.py"),
         "--cases", "learned_basic", "--ladders", "L1", "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert '"ladder": "L1"' in proc.stdout
