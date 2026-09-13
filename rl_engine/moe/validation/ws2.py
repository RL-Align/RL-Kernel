# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS2 extensions for the T09 ladder: rank dimension + cross-config (§3-T09).

Contract §3 (WS2 Gate row): "同一 token 的 Core semantic fingerprint exact，
Envelope 按配置完整" — across TP/SP/CP/DP/PP/EP configs the per-token Core
semantics must be identical, while Envelope completeness is judged per
config. This module adds:

- :func:`check_rank_completeness` — every expected rank must contribute an
  artifact (missing -> MISSING_RANK, provider band §6).
- :func:`run_ws2_cross_config` — compare two configurations' rank sets:
  partition ownership must cover the expected token set exactly once;
  replica ownership (TP) may repeat tokens but every carrier's semantic
  hash must be identical; any token-level divergence names the first
  offending (token, rank) pair via the six-tuple.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_engine.moe.p3_verdicts import P3Verdict
from rl_engine.moe.validation.fingerprint import Artifact, per_token_semantic_hashes
from rl_engine.moe.validation.report import LadderReport, make_fail, make_pass

_fail, _pass = make_fail, make_pass  # module-local aliases for brevity

TOKEN_PARTITION = "partition"   # CP/DP: token appears on exactly one rank
TOKEN_REPLICA = "replica"       # TP: token may repeat; hashes must agree


@dataclass(frozen=True)
class RankArtifact:
    """One rank's artifact under a specific parallel config."""

    rank: int
    group: str                     # e.g. "tp2-sp0", "dp4-cp2"
    artifact: Artifact


def check_rank_completeness(
    case_id: str,
    artifacts: list[RankArtifact],
    expected_ranks: range | list[int],
) -> LadderReport:
    """All expected ranks present, no duplicates (§6 MISSING_RANK)."""
    seen: dict[int, int] = {}
    for ra in artifacts:
        seen[ra.rank] = seen.get(ra.rank, 0) + 1
    missing = sorted(set(expected_ranks) - set(seen))
    if missing:
        return _fail("WS2-rank", case_id, P3Verdict.MISSING_RANK,
                     f"missing ranks: {missing}", missing_ranks=missing)
    duplicated = sorted(r for r, n in seen.items() if n > 1)
    if duplicated:
        return _fail("WS2-rank", case_id, P3Verdict.STALE_RUN_METADATA,
                     f"duplicated ranks: {duplicated}",
                     hint="one artifact per (case, config, rank) — stale run mix-in?")
    return _pass("WS2-rank", case_id,
                 f"ranks {sorted(seen)} complete", ranks=sorted(seen))


def _tokens_by_rank(arts: list[RankArtifact]) -> dict[int, dict[int, str]]:
    """{rank: {token: semantic_hash}} over active rows."""
    out: dict[int, dict[int, str]] = {}
    for ra in arts:
        out[ra.rank] = per_token_semantic_hashes(ra.artifact.rows, ra.artifact.identity)
    return out


def run_ws2_cross_config(
    case_id: str,
    base: list[RankArtifact],
    other: list[RankArtifact],
    *,
    base_config: str,
    other_config: str,
    ownership: str = TOKEN_PARTITION,
) -> LadderReport:
    """Cross-config: same token -> identical Core semantic hash (§3 WS2).

    ``base`` defines the expected token set. ``other`` must cover it under
    its ownership mode:
    - partition: each base token present exactly once across ``other`` ranks;
      missing -> INCOMPLETE_ARTIFACT, duplicated -> AMBIGUOUS_GLOBAL_TOKEN_MAPPING.
    - replica: token may appear on several ranks; every occurrence must carry
      the base hash; a divergent carrier -> ROUTE_SEMANTIC_FINGERPRINT_MISMATCH
      with (token, rank) named.
    Envelope differences between configs are expected and NOT judged here.
    """
    if ownership not in (TOKEN_PARTITION, TOKEN_REPLICA):
        raise ValueError(f"unknown ownership mode: {ownership}")

    bt = _tokens_by_rank(base)
    ot = _tokens_by_rank(other)
    expected = {t for per in bt.values() for t in per}

    carriers: dict[int, list[int]] = {}
    for rank, per in ot.items():
        for t in per:
            carriers.setdefault(t, []).append(rank)

    missing = sorted(expected - set(carriers))
    if missing:
        return _fail("WS2-cross", case_id, P3Verdict.INCOMPLETE_ARTIFACT,
                     f"{base_config}->{other_config}: tokens uncovered: {missing[:8]}",
                     missing=missing, base_config=base_config, other_config=other_config)

    if ownership == TOKEN_PARTITION:
        dup = sorted(t for t, rs in carriers.items() if len(rs) > 1)
        if dup:
            return _fail("WS2-cross", case_id, P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING,
                         f"{base_config}->{other_config}: tokens owned by >1 rank: "
                         f"{ {t: carriers[t] for t in dup[:4]} }",
                         duplicated=dup, base_config=base_config, other_config=other_config)

    base_hash = {t: next(per[t] for per in bt.values() if t in per) for t in expected}
    for t in sorted(expected):
        for rank in sorted(carriers[t]):
            if ot[rank][t] != base_hash[t]:
                return _fail(
                    "WS2-cross", case_id, P3Verdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH,
                    f"{base_config}->{other_config}: token={t} rank={rank} semantic "
                    f"{ot[rank][t][:12]}.. vs base {base_hash[t][:12]}..",
                    first_token=t, first_rank=rank,
                    base_config=base_config, other_config=other_config,
                )
    return _pass("WS2-cross", case_id,
                 f"{base_config}->{other_config}: {len(expected)} tokens semantic-exact "
                 f"({ownership}, ranks={sorted(ot)})",
                 tokens=len(expected), ownership=ownership,
                 base_config=base_config, other_config=other_config)
