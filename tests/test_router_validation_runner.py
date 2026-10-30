# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the router validation runner: WS1 stages + WS2 checks.

Covers L1/L2/L3a/L3b (WS1), rank completeness + cross-config ownership
(WS2), and the synthetic producer helpers.
Synthetic producer usage keeps everything testable before the fixture
manifest lands.
"""

from __future__ import annotations

import dataclasses
import pytest
import torch

from rl_engine.moe.naive_topk import K
from rl_engine.moe.t01_verdicts import RouterVerdict
from rl_engine.moe.validation.t01_fingerprint import (
    RouteIdentity,
    RouteRow,
    per_token_semantic_hashes,
    route_artifact_hash,
    route_semantic_hash,
)
from rl_engine.moe.validation.runner import (
    TOKEN_PARTITION,
    TOKEN_REPLICA,
    RankArtifact,
    check_rank_completeness,
    run_l1_repeat,
    run_l2_invariance,
    run_l3a_oracle,
    run_l3b_dual_engine,
    run_ws2_cross_config,
)
from rl_engine.moe.validation.t01_synthetic_producer import (
    shard_rows,
    make_artifact,
    make_learned_route_rows,
    repack_rows,
)


# Runner coverage: L1-L3b boundaries, WS2, and whole-token sharding helpers.
def _ident(**over):
    base = dict(
        checkpoint_id="ckpt-x",
        weight_id="w-x",
        table_fingerprint="00" * 32,
        bias_fingerprint="11" * 32,
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


# --- L1: same-configuration repeats must be byte-identical --------------------
def test_l1_repeat_identical_artifacts_pass():
    rows, _ = _learned_case()
    a = _wrap(rows, _ident())
    b = _wrap(rows, _ident())
    rep = run_l1_repeat("cx", a, b)
    assert rep.passed
    assert rep.extra["artifact_hash"]


def test_l1_repeat_padding_change_fails_artifact_hash():
    """Padding is audited ONLY by L1's artifact gate."""
    rows, _ = _learned_case()
    a = _wrap(rows, _ident())
    b = _wrap(rows, _ident(), padding_rows=2)
    rep = run_l1_repeat("cx", a, b)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.ROUTE_ARTIFACT_FINGERPRINT_MISMATCH


def test_l1_attempt_id_change_fails():
    rows, _ = _learned_case()
    a = _wrap(rows, _ident())
    b = make_artifact("cx", rows, _ident(), run_id="r1", engine_id="e1", attempt_id=2)
    rep = run_l1_repeat("cx", a, b)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.ROUTE_ARTIFACT_FINGERPRINT_MISMATCH


def test_l1_rejects_misaligned_rows_and_envelopes():
    rows, _ = _learned_case()
    valid = _wrap(rows, _ident())
    malformed = dataclasses.replace(valid, envelopes=valid.envelopes[:-1])
    rep = run_l1_repeat("cx", valid, malformed)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.CORRUPT_ARTIFACT


# --- producer helpers: shards never split a token ------------------------------
def test_shard_rows_keeps_routing_units_whole():
    """Sharding must never split one token's slot rows across ranks."""
    rows, _meta = _learned_case(t_rows=5)  # odd count catches naive row slicing
    shards = shard_rows(rows, 2)
    assert len(shards) == 2
    # Every shard contains whole tokens and token sets are disjoint.
    owners: dict[int, int] = {}
    for idx, shard in enumerate(shards):
        for r in shard:
            assert r.global_token_id != -1
            prev = owners.setdefault(r.global_token_id, idx)
            assert prev == idx, f"token {r.global_token_id} split across shards {prev} and {idx}"
    # No rows are lost or duplicated; all slots of a token travel together.
    assert sum(len(s) for s in shards) == len(rows)
    by_token: dict[int, set[int]] = {}
    for r in rows:
        by_token.setdefault(r.global_token_id, set()).add(r.topk_index)
    for token, slots in by_token.items():
        shard = next(s for s in shards if any(r.global_token_id == token for r in s))
        assert {r.topk_index for r in shard if r.global_token_id == token} == slots


def test_shard_rows_roundtrip_covers_all_tokens():
    """Union of shards == base token set (nothing dropped, nothing invented)."""
    rows, _meta = _learned_case(t_rows=7)
    shards = shard_rows(rows, 3)
    seen = {r.global_token_id for s in shards for r in s}
    expected = {r.global_token_id for r in rows}
    assert seen == expected
    assert all(shard for shard in shards[:2])  # 7 tokens over 3 shards


def test_learned_route_weights_use_fp32_arithmetic():
    rows, _ = make_learned_route_rows("cx", seed=123, t_rows=2, layers=(3,))
    by_token = [row for row in rows if row.global_token_id == 0]
    scores = torch.tensor([row.weight_score for row in by_token], dtype=torch.float32)
    pair_sums = torch.stack(
        (
            scores[0] + scores[1],
            scores[2] + scores[3],
            scores[4] + scores[5],
        )
    )
    total = (pair_sums[0] + pair_sums[1]) + pair_sums[2]
    zeta = total + torch.tensor(1e-20, dtype=torch.float32)
    expected = (scores / zeta) * torch.tensor(1.5, dtype=torch.float32)
    actual = torch.tensor([row.route_weight for row in by_token], dtype=torch.float32)
    assert torch.equal(actual, expected)


# --- L2: perturbation invariance -----------------------------------------------
def test_l2_padding_and_repack_are_invariant():
    rows, meta = _learned_case()
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    base = _wrap(rows, ident)
    pert = make_artifact(
        "cx",
        repack_rows(rows, batch_size=2),
        ident,
        run_id="r2",
        engine_id="miles",
        attempt_id=2,
        placement_offset=3,
        padding_rows=5,
    )
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
    assert rep.verdict is RouterVerdict.INCOMPLETE_ARTIFACT
    assert rep.extra["missing"] == [[3, 1]]  # (layer=3, token=1) unit


def test_l2_extra_token_fails_ambiguous_mapping():
    rows, meta = _learned_case()
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    base = _wrap(rows, ident)
    ghost = dataclasses.replace(rows[0], global_token_id=99, input_token_id=99)
    pert = _wrap(rows + [ghost], ident)
    rep = run_l2_invariance("cx", base, pert, variant="ghost")
    assert not rep.passed
    assert rep.verdict is RouterVerdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING


def test_l2_weight_bit_flip_is_first_mismatch():
    rows, meta = _learned_case()
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    base = _wrap(rows, ident)
    flipped = list(rows)
    flipped[0] = dataclasses.replace(flipped[0], route_weight=flipped[0].route_weight + 1e-6)
    pert = _wrap(flipped, ident)
    rep = run_l2_invariance("cx", base, pert, variant="flip")
    assert not rep.passed
    assert rep.verdict is RouterVerdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH
    assert rep.extra["first_token"] == rows[0].global_token_id


def test_l2_multilayer_row_reversal_is_invariant():
    """The semantic unit is (layer, token): a multi-layer artifact must be
    stable under ANY physical row permutation, and the same token routed at
    two layers must hash as two separate decisions (never merged)."""
    rows, meta = make_learned_route_rows("cx", seed=5, t_rows=3, layers=(3, 4))
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    base = _wrap(rows, ident)
    pert = _wrap(list(reversed(rows)), ident)  # full physical reversal
    rep = run_l2_invariance("cx", base, pert, variant="full-reversal")
    assert rep.passed, rep.detail
    # Two layers by three tokens produce six distinct routing decisions.
    assert rep.extra["units"] == 6


def test_l2_missing_layer_reports_missing_unit():
    """Dropping one layer's rows for a token is INCOMPLETE_ARTIFACT with the
    (layer, token) unit named — not a vague fingerprint mismatch."""
    rows, meta = make_learned_route_rows("cx", seed=5, t_rows=3, layers=(3, 4))
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    base = _wrap(rows, ident)
    dropped = [r for r in rows if not (r.absolute_layer == 4 and r.global_token_id == 1)]
    pert = _wrap(dropped, ident)
    rep = run_l2_invariance("cx", base, pert, variant="drop-layer")
    assert not rep.passed
    assert rep.verdict is RouterVerdict.INCOMPLETE_ARTIFACT
    assert rep.extra["missing"] == [[4, 1]]  # (layer=4, token=1)


# --- L3a -----------------------------------------------------------------------
def test_l3a_byte_exact_passes():
    rows, _ = _learned_case()
    rep = run_l3a_oracle("cx", rows, list(rows))
    assert rep.passed


def test_l3a_ignores_padding_rows():
    """Padding is audited only by the same-config artifact gate."""
    rows, _ = _learned_case()
    pad = RouteRow(
        global_token_id=-1,
        input_token_id=-1,
        absolute_layer=3,
        router_mode="learned",
        topk_index=0,
        logical_expert_id=-1,
        valid=False,
        invalid_reason="padding",
        route_weight=0.0,
        weight_score=0.0,
        selection_score=0.0,
    )
    rep = run_l3a_oracle("cx", rows + [pad], rows)
    assert rep.passed


def test_l3a_expert_id_mismatch_reports_topk_order():
    rows, _ = _learned_case()
    bad = list(rows)
    bad[7] = dataclasses.replace(bad[7], logical_expert_id=(bad[7].logical_expert_id + 1) % 256)
    rep = run_l3a_oracle("cx", rows, bad)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.TOPK_ORDER_MISMATCH
    assert rep.first_mismatch is not None
    assert rep.first_mismatch.key.global_token_id == bad[7].global_token_id


def test_l3a_row_count_mismatch_fails_incomplete():
    rows, _ = _learned_case()
    rep = run_l3a_oracle("cx", rows, rows[:-1])
    assert not rep.passed
    assert rep.verdict is RouterVerdict.INCOMPLETE_ARTIFACT


def test_l3a_signed_zero_is_a_byte_mismatch():
    rows, _ = _learned_case()
    oracle = list(rows)
    candidate = list(rows)
    oracle[0] = dataclasses.replace(oracle[0], route_weight=0.0)
    candidate[0] = dataclasses.replace(candidate[0], route_weight=-0.0)
    rep = run_l3a_oracle("cx", oracle, candidate)
    assert not rep.passed
    assert rep.verdict is RouterVerdict.ROUTE_WEIGHT_BYTES_MISMATCH
    assert rep.first_mismatch is not None
    assert rep.first_mismatch.owner == "route-plan"
    assert rep.first_mismatch.key.site == "weight"


def test_l3a_multilayer_sort_key_is_unambiguous():
    rows, _ = make_learned_route_rows("cx", seed=5, t_rows=2, layers=(3, 4))
    rep = run_l3a_oracle("cx", rows, list(reversed(rows)))
    assert rep.passed


@pytest.mark.parametrize(
    "field_name",
    [
        "route_weight",
        "weight_score",
        "selection_score",
    ],
)
def test_l3a_nonfinite_active_value_fails_closed(field_name):
    """Identical NaN bytes on both sides are invalid, never a byte-exact pass."""
    rows, _ = _learned_case()
    bad = list(rows)
    bad[0] = dataclasses.replace(bad[0], **{field_name: float("nan")})
    rep = run_l3a_oracle("cx", bad, list(bad))
    assert not rep.passed
    assert rep.verdict is RouterVerdict.NON_FINITE
    assert rep.extra["nonfinite_field"] == field_name
    assert rep.first_mismatch is not None
    assert rep.first_mismatch.boundary == "L3a non-finite gate"


# --- L3b -----------------------------------------------------------------------


def _meta(router_mode="learned", layer=3):
    return {
        "case_id": "cx",
        "checkpoint_id": "c",
        "weight_id": "w",
        "absolute_layer": layer,
        "router_mode": router_mode,
        "table_fingerprint": "00" * 32,
        "bias_fingerprint": "11" * 32,
        "logit_round_point": "fp32_direct",
        "tie_break_policy": "q_desc_id_asc",
        "capacity_policy": "dropless_v1",
    }


def test_l3b_identical_engines_pass_four_stages():
    from rl_engine.moe.validation.first_mismatch import MismatchKey, TraceEvent

    events = [
        TraceEvent(
            key=MismatchKey(
                absolute_layer=3,
                site="topk",
                pass_direction="forward",
                event_index=i,
                global_token_id=i // K,
                rank=0,
            ),
            payload={"logical_expert_id": i % 256},
        )
        for i in range(12)
    ]
    w = torch.full((12,), 0.25)
    scores = torch.full((12,), 0.5)
    dz = torch.full((12,), 1.0)
    rep = run_l3b_dual_engine(
        "cx",
        _meta(),
        dict(_meta()),
        events,
        list(events),
        w,
        w.clone(),
        dz,
        dz.clone(),
        lhs_scores=scores,
        rhs_scores=scores.clone(),
    )
    assert rep.passed, rep.detail


def test_l3b_score_mismatch_is_not_silently_skipped():
    from rl_engine.moe.validation.first_mismatch import MismatchKey, TraceEvent

    events = [
        TraceEvent(
            key=MismatchKey(
                absolute_layer=3,
                site="topk",
                pass_direction="forward",
                event_index=i,
                global_token_id=0,
                rank=0,
            ),
            payload={"logical_expert_id": i},
        )
        for i in range(K)
    ]
    w = torch.full((K,), 0.25)
    lhs_scores = torch.full((K,), 0.5)
    rhs_scores = lhs_scores.clone()
    rhs_scores[2] += 1e-6
    dz = torch.zeros(K)
    rep = run_l3b_dual_engine(
        "cx",
        _meta(),
        dict(_meta()),
        events,
        list(events),
        w,
        w.clone(),
        dz,
        dz.clone(),
        lhs_scores=lhs_scores,
        rhs_scores=rhs_scores,
    )
    assert not rep.passed
    assert rep.verdict is RouterVerdict.SCORE_BYTES_MISMATCH
    assert rep.extra["walked"] == ["identity", "discrete", "score_weight"]


def test_l3b_identity_drift_halts_walk():
    events = []
    lhs_meta = _meta()
    rhs_meta = _meta()
    # generic identity drift (10); policy fields keep 14/56 and are covered
    # by their own comparison tests
    rhs_meta["checkpoint_id"] = "ckpt-other"
    w = torch.zeros(0)
    rep = run_l3b_dual_engine(
        "cx",
        lhs_meta,
        rhs_meta,
        events,
        events,
        w,
        w.clone(),
        w,
        w.clone(),
        lhs_scores=w,
        rhs_scores=w.clone(),
    )
    assert not rep.passed
    assert rep.verdict is RouterVerdict.IDENTITY_DRIFT
    assert rep.extra["walked"] == ["identity"]


# --- fingerprints ---------------------------------------------------------------


def test_semantic_hash_excludes_padding_and_envelope():
    rows, meta = _learned_case()
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    a = _wrap(rows, ident)
    b = _wrap(
        rows,
        ident,
        run_id="r9",
        engine_id="miles",
        attempt_id=9,
        placement_offset=7,
        padding_rows=3,
    )
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
        [RouteRow(0, 0, 3, "learned", 0, 1, True, None, 0.0, 0.0, 0.0)], _ident()
    )
    h2 = route_semantic_hash(
        [RouteRow(0, 0, 3, "learned", 0, 1, True, None, -0.0, 0.0, 0.0)], _ident()
    )
    assert h1 != h2


# --- WS2: rank completeness + cross-config ownership -----------------------


def _case(t_rows=4, seed=21):
    rows, meta = make_learned_route_rows("cx", seed=seed, t_rows=t_rows, layers=(3,))
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    return rows, ident


def _rank_art(rank, rows, ident, group, **kw):
    kw.setdefault("run_id", f"r{rank}")
    kw.setdefault("engine_id", "e1")
    kw.setdefault("attempt_id", 1)
    return RankArtifact(
        rank=rank,
        group=group,
        artifact=make_artifact(f"cx-r{rank}", rows, ident, rank=rank, **kw),
    )


# --- rank completeness --------------------------------------------------------


def test_rank_completeness_pass():
    rows, ident = _case()
    arts = [_rank_art(r, rows, ident, "dp2") for r in range(2)]
    rep = check_rank_completeness("cx", arts, range(2))
    assert rep.passed


def test_rank_completeness_missing_rank():
    rows, ident = _case()
    arts = [_rank_art(0, rows, ident, "dp2")]
    rep = check_rank_completeness("cx", arts, range(2))
    assert not rep.passed
    assert rep.verdict is RouterVerdict.MISSING_RANK
    assert rep.extra["missing_ranks"] == [1]


def test_rank_completeness_duplicate_rank_is_stale_mix():
    rows, ident = _case()
    arts = [_rank_art(0, rows, ident, "dp1"), _rank_art(0, rows, ident, "dp1")]
    rep = check_rank_completeness("cx", arts, range(1))
    assert not rep.passed
    assert rep.verdict is RouterVerdict.STALE_RUN_METADATA


def test_rank_completeness_rejects_unexpected_rank():
    rows, ident = _case()
    arts = [_rank_art(r, rows, ident, "dp3") for r in range(3)]
    rep = check_rank_completeness("cx", arts, range(2))
    assert not rep.passed
    assert rep.verdict is RouterVerdict.STALE_RUN_METADATA
    assert rep.extra["unexpected_ranks"] == [2]


def test_rank_completeness_stale_data_precedes_missing_rank():
    rows, ident = _case()
    arts = [_rank_art(0, rows, ident, "dp2"), _rank_art(0, rows, ident, "dp2")]
    rep = check_rank_completeness("cx", arts, range(2))
    assert not rep.passed
    assert rep.verdict is RouterVerdict.STALE_RUN_METADATA
    assert rep.extra["duplicated_ranks"] == [0]


def test_rank_completeness_rejects_mixed_groups():
    rows, ident = _case()
    arts = [_rank_art(0, rows, ident, "dp2"), _rank_art(1, rows, ident, "tp2")]
    rep = check_rank_completeness("cx", arts, range(2))
    assert not rep.passed
    assert rep.verdict is RouterVerdict.STALE_RUN_METADATA
    assert rep.extra["groups"] == ["dp2", "tp2"]


# --- cross-config: partition (CP/DP) -------------------------------------------


def _split_rows(rows, parts):
    """Partition tokens into `parts` shards by global_token_id."""
    out = [[] for _ in range(parts)]
    for r in rows:
        out[r.global_token_id % parts].append(r)
    return out


def test_cross_config_partition_pass():
    rows, ident = _case(t_rows=8)
    base = [_rank_art(0, rows, ident, "dp1")]
    shards = _split_rows(rows, 2)
    other = [_rank_art(r, shards[r], ident, "dp2-cp1") for r in range(2)]
    rep = run_ws2_cross_config(
        "cx", base, other, base_config="dp1", other_config="dp2", ownership=TOKEN_PARTITION
    )
    assert rep.passed, rep.detail


def test_cross_config_partition_dropped_token_fails():
    rows, ident = _case(t_rows=8)
    base = [_rank_art(0, rows, ident, "dp1")]
    shards = _split_rows(rows, 2)
    shards[1] = [r for r in shards[1] if r.global_token_id != 3]
    other = [_rank_art(r, shards[r], ident, "dp2") for r in range(2)]
    rep = run_ws2_cross_config(
        "cx", base, other, base_config="dp1", other_config="dp2", ownership=TOKEN_PARTITION
    )
    assert not rep.passed
    assert rep.verdict is RouterVerdict.INCOMPLETE_ARTIFACT


def test_cross_config_partition_double_ownership_fails():
    """Partition mode: a token routed by two ranks is ambiguous mapping."""
    rows, ident = _case(t_rows=8)
    base = [_rank_art(0, rows, ident, "dp1")]
    shards = _split_rows(rows, 2)
    shards[1] = shards[1] + [r for r in shards[0] if r.global_token_id == 0]
    other = [_rank_art(r, shards[r], ident, "dp2") for r in range(2)]
    rep = run_ws2_cross_config(
        "cx", base, other, base_config="dp1", other_config="dp2", ownership=TOKEN_PARTITION
    )
    assert not rep.passed
    assert rep.verdict is RouterVerdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING


def test_cross_config_semantic_drift_names_token_and_rank():
    rows, ident = _case(t_rows=8)
    base = [_rank_art(0, rows, ident, "dp1")]
    shards = _split_rows(rows, 2)
    drifted = list(shards[1])
    drifted[0] = dataclasses.replace(drifted[0], route_weight=drifted[0].route_weight + 1e-6)
    shards[1] = drifted
    other = [_rank_art(r, shards[r], ident, "dp2") for r in range(2)]
    rep = run_ws2_cross_config(
        "cx", base, other, base_config="dp1", other_config="dp2", ownership=TOKEN_PARTITION
    )
    assert not rep.passed
    assert rep.verdict is RouterVerdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH
    assert rep.extra["first_rank"] == 1


def test_cross_config_partition_ghost_token_fails():
    """A routing decision owned by no base rank (unauthorized) must fail 63."""
    rows, ident = _case(t_rows=8)
    base = [_rank_art(0, rows, ident, "dp1")]
    shards = _split_rows(rows, 2)
    ghost = dataclasses.replace(shards[1][0], global_token_id=99, input_token_id=99)
    shards[1].append(ghost)
    other = [_rank_art(r, shards[r], ident, "dp2") for r in range(2)]
    rep = run_ws2_cross_config(
        "cx", base, other, base_config="dp1", other_config="dp2", ownership=TOKEN_PARTITION
    )
    assert not rep.passed
    assert rep.verdict is RouterVerdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING


# --- cross-config: replica (TP) -------------------------------------------------


def test_cross_config_replica_pass():
    """TP replica: same tokens on every rank, hashes identical."""
    rows, ident = _case(t_rows=4)
    base = [_rank_art(0, rows, ident, "tp1")]
    other = [_rank_art(r, rows, ident, "tp4-sp0") for r in range(4)]
    rep = run_ws2_cross_config(
        "cx", base, other, base_config="tp1", other_config="tp4", ownership=TOKEN_REPLICA
    )
    assert rep.passed, rep.detail


def test_cross_config_replica_one_bad_rank_fails():
    rows, ident = _case(t_rows=4)
    base = [_rank_art(0, rows, ident, "tp1")]
    bad = list(rows)
    bad[2] = dataclasses.replace(bad[2], weight_score=bad[2].weight_score + 1e-6)
    other = [
        _rank_art(0, rows, ident, "tp4"),
        _rank_art(1, bad, ident, "tp4"),
        _rank_art(2, rows, ident, "tp4"),
        _rank_art(3, rows, ident, "tp4"),
    ]
    rep = run_ws2_cross_config(
        "cx", base, other, base_config="tp1", other_config="tp4", ownership=TOKEN_REPLICA
    )
    assert not rep.passed
    assert rep.verdict is RouterVerdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH
    assert rep.extra["first_rank"] == 1


def test_cross_config_replica_ghost_token_fails():
    """TP replica: a token no base rank routed is still unauthorized (63)."""
    rows, ident = _case(t_rows=4)
    base = [_rank_art(0, rows, ident, "tp1")]
    ghost = dataclasses.replace(rows[0], global_token_id=77, input_token_id=77)
    other = [
        _rank_art(0, rows + [ghost], ident, "tp4"),
        _rank_art(1, rows, ident, "tp4"),
    ]
    rep = run_ws2_cross_config(
        "cx", base, other, base_config="tp1", other_config="tp4", ownership=TOKEN_REPLICA
    )
    assert not rep.passed
    assert rep.verdict is RouterVerdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING


def test_cross_config_envelope_changes_are_not_judged():
    """Envelope (placement/run) differences across configs are fine."""
    rows, ident = _case(t_rows=4)
    base = [_rank_art(0, rows, ident, "tp1", engine_id="megatron")]
    other = [
        _rank_art(1, rows, ident, "tp4-sp0", engine_id="miles", placement_offset=8, padding_rows=3)
    ]
    rep = run_ws2_cross_config(
        "cx", base, other, base_config="tp1", other_config="tp4", ownership=TOKEN_REPLICA
    )
    assert rep.passed


def test_cross_config_rejects_unknown_ownership():
    rows, ident = _case(t_rows=2)
    base = [_rank_art(0, rows, ident, "tp1")]
    with pytest.raises(ValueError):
        run_ws2_cross_config("cx", base, base, base_config="a", other_config="b", ownership="ep??")
