# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS2 ladder tests: rank dimension + cross-config (§3 WS2 Gate row).

"同一 token 的 Core semantic fingerprint exact，Envelope 按配置完整。"
"""

from __future__ import annotations

import dataclasses

import pytest

from rl_engine.moe.p3_verdicts import P3Verdict
from rl_engine.moe.validation.fingerprint import RouteIdentity
from rl_engine.moe.validation.synthetic_producer import (
    make_artifact,
    make_learned_route_rows,
)
from rl_engine.moe.validation.ws2 import (
    TOKEN_PARTITION,
    TOKEN_REPLICA,
    RankArtifact,
    check_rank_completeness,
    run_ws2_cross_config,
)


def _ident(**over):
    base = dict(
        checkpoint_id="ckpt-x", weight_id="w-x",
        table_fingerprint="00" * 32, bias_fingerprint="11" * 32,
    )
    base.update(over)
    return RouteIdentity(**base)


def _case(t_rows=4, seed=21):
    rows, meta = make_learned_route_rows("cx", seed=seed, t_rows=t_rows, layers=(3,))
    ident = _ident(bias_fingerprint=meta["bias_fingerprint"])
    return rows, ident


def _rank_art(rank, rows, ident, group, **kw):
    kw.setdefault("run_id", f"r{rank}")
    kw.setdefault("engine_id", "e1")
    kw.setdefault("attempt_id", 1)
    return RankArtifact(
        rank=rank, group=group,
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
    assert rep.verdict is P3Verdict.MISSING_RANK
    assert rep.extra["missing_ranks"] == [1]


def test_rank_completeness_duplicate_rank_is_stale_mix():
    rows, ident = _case()
    arts = [_rank_art(0, rows, ident, "dp1"), _rank_art(0, rows, ident, "dp1")]
    rep = check_rank_completeness("cx", arts, range(1))
    assert not rep.passed
    assert rep.verdict is P3Verdict.STALE_RUN_METADATA


# --- cross-config: partition (CP/DP) -------------------------------------------

def _split_rows(rows, parts, k):
    """Partition tokens into `parts` shards by global_token_id."""
    out = [[] for _ in range(parts)]
    for r in rows:
        out[r.global_token_id % parts].append(r)
    return out


def test_cross_config_partition_pass():
    rows, ident = _case(t_rows=8)
    base = [_rank_art(0, rows, ident, "dp1")]
    shards = _split_rows(rows, 2, None)
    other = [_rank_art(r, shards[r], ident, "dp2-cp1") for r in range(2)]
    rep = run_ws2_cross_config("cx", base, other,
                               base_config="dp1", other_config="dp2",
                               ownership=TOKEN_PARTITION)
    assert rep.passed, rep.detail


def test_cross_config_partition_dropped_token_fails():
    rows, ident = _case(t_rows=8)
    base = [_rank_art(0, rows, ident, "dp1")]
    shards = _split_rows(rows, 2, None)
    shards[1] = [r for r in shards[1] if r.global_token_id != 3]
    other = [_rank_art(r, shards[r], ident, "dp2") for r in range(2)]
    rep = run_ws2_cross_config("cx", base, other,
                               base_config="dp1", other_config="dp2",
                               ownership=TOKEN_PARTITION)
    assert not rep.passed
    assert rep.verdict is P3Verdict.INCOMPLETE_ARTIFACT


def test_cross_config_partition_double_ownership_fails():
    """Partition mode: a token routed by two ranks is ambiguous mapping."""
    rows, ident = _case(t_rows=8)
    base = [_rank_art(0, rows, ident, "dp1")]
    shards = _split_rows(rows, 2, None)
    shards[1] = shards[1] + [r for r in shards[0] if r.global_token_id == 0]
    other = [_rank_art(r, shards[r], ident, "dp2") for r in range(2)]
    rep = run_ws2_cross_config("cx", base, other,
                               base_config="dp1", other_config="dp2",
                               ownership=TOKEN_PARTITION)
    assert not rep.passed
    assert rep.verdict is P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING


def test_cross_config_semantic_drift_names_token_and_rank():
    rows, ident = _case(t_rows=8)
    base = [_rank_art(0, rows, ident, "dp1")]
    shards = _split_rows(rows, 2, None)
    drifted = list(shards[1])
    drifted[0] = dataclasses.replace(drifted[0], route_weight=drifted[0].route_weight + 1e-6)
    shards[1] = drifted
    other = [_rank_art(r, shards[r], ident, "dp2") for r in range(2)]
    rep = run_ws2_cross_config("cx", base, other,
                               base_config="dp1", other_config="dp2",
                               ownership=TOKEN_PARTITION)
    assert not rep.passed
    assert rep.verdict is P3Verdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH
    assert rep.extra["first_rank"] == 1


# --- cross-config: replica (TP) -------------------------------------------------

def test_cross_config_replica_pass():
    """TP replica: same tokens on every rank, hashes identical."""
    rows, ident = _case(t_rows=4)
    base = [_rank_art(0, rows, ident, "tp1")]
    other = [_rank_art(r, rows, ident, "tp4-sp0") for r in range(4)]
    rep = run_ws2_cross_config("cx", base, other,
                               base_config="tp1", other_config="tp4",
                               ownership=TOKEN_REPLICA)
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
    rep = run_ws2_cross_config("cx", base, other,
                               base_config="tp1", other_config="tp4",
                               ownership=TOKEN_REPLICA)
    assert not rep.passed
    assert rep.verdict is P3Verdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH
    assert rep.extra["first_rank"] == 1


def test_cross_config_envelope_changes_are_not_judged():
    """Envelope (placement/run) differences across configs are fine (§3)."""
    rows, ident = _case(t_rows=4)
    base = [_rank_art(0, rows, ident, "tp1", engine_id="megatron")]
    other = [_rank_art(1, rows, ident, "tp4-sp0", engine_id="miles",
                       placement_offset=8, padding_rows=3)]
    rep = run_ws2_cross_config("cx", base, other,
                               base_config="tp1", other_config="tp4",
                               ownership=TOKEN_REPLICA)
    assert rep.passed


def test_cross_config_rejects_unknown_ownership():
    rows, ident = _case(t_rows=2)
    base = [_rank_art(0, rows, ident, "tp1")]
    with pytest.raises(ValueError):
        run_ws2_cross_config("cx", base, base, base_config="a",
                             other_config="b", ownership="ep??")


# --- CLI integration ------------------------------------------------------------

def test_check_p3_cli_ws2_smoke():
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "check_p3.py"),
         "--cases", "learned_basic", "--ladders", "L1"],
        capture_output=True, text=True, cwd=repo,
    )
    assert proc.returncode == 0
