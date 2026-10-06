#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""check_router: DSV4-Flash MoE router validation driver CLI.

WS1/WS2 gate: runs the recorded/synthetic validation stages
L1 (repeat) / L2 (invariance) / L3a (oracle) / L3b (dual-engine) plus the
WS2 checks (rank completeness, cross-config ownership) and prints one
summary line per case plus a final exit code (0 = all pass; non-zero =
first failing verdict band).

Examples:
    python scripts/check_router.py --cases smoke
    python scripts/check_router.py --cases learned_basic,hash_basic
    python scripts/check_router.py --stages L1,L2 --json
    python scripts/check_router.py --stages WS2-cross --cases learned_basic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.moe.naive_topk import K  # noqa: E402
from rl_engine.moe.t01_verdicts import RouterVerdict  # noqa: E402
from rl_engine.moe.validation.t01_fingerprint import RouteIdentity, RouteRow  # noqa: E402
from rl_engine.moe.validation.runner import (  # noqa: E402
    ValidationReport,
    RankArtifact,
    check_rank_completeness,
    run_l1_repeat,
    run_l2_invariance,
    run_l3a_oracle,
    run_ws2_cross_config,
    make_fail,
)
from rl_engine.moe.validation.t01_synthetic_producer import (  # noqa: E402
    make_artifact,
    make_learned_route_rows,
    repack_rows,
    shard_rows,
)

VALIDATION_STAGES = ("L1", "L2", "L3a", "L3b", "WS2-rank", "WS2-cross")


# Exit code = the first failing report's verdict code (0 = all pass). CI can
# classify the failure category straight from $?: 1-2 device band, 10-22
# provider band, 50-72 runner band.
def _exit_code(reports: list[ValidationReport]) -> int:
    for r in reports:
        if not r.passed and r.verdict is not None:
            return int(r.verdict.value)
    return 0


# --- case registry -------------------------------------------------------------


def _identity_from_meta(meta: dict[str, Any]) -> RouteIdentity:
    """Build the routing identity from case metadata.

    Optional fingerprints must preserve absence. Converting ``None`` to the
    string ``"None"`` would fabricate a present but non-decodable fingerprint.
    """
    return RouteIdentity(
        checkpoint_id=str(meta["checkpoint_id"]),
        weight_id=str(meta["weight_id"]),
        logit_round_point=str(meta["logit_round_point"]),
        tie_break_policy=str(meta["tie_break_policy"]),
        capacity_policy=str(meta["capacity_policy"]),
        # absent fingerprint (None or "") must stay absent — str(None) would
        # fabricate the string "None" and crash hex decoding downstream
        table_fingerprint=str(meta["table_fingerprint"]) if meta.get("table_fingerprint") else None,
        bias_fingerprint=str(meta["bias_fingerprint"]) if meta.get("bias_fingerprint") else None,
    )


def _ws2_material(
    case_id: str, rows: list[RouteRow], ident: RouteIdentity
) -> dict[str, list[RankArtifact]]:
    """Rank-artifact sets for the WS2 stages (synthetic parallel configs).

    - ``ranks_base``      : single rank, unsharded (dp1)
    - ``ranks_partition`` : two ranks over whole-token shards (dp2, CP/DP-like)
    - ``ranks_replica``   : two ranks carrying the full set (tp2, TP-like)
    Same identity throughout, so cross-config semantic hashes must agree.
    """
    base = [
        RankArtifact(
            rank=0,
            group="dp1",
            artifact=make_artifact(
                case_id, rows, ident, run_id="r1", engine_id="megatron", attempt_id=1
            ),
        )
    ]
    shards = shard_rows(rows, 2)
    partition = [
        RankArtifact(
            rank=r,
            group="dp2",
            artifact=make_artifact(
                case_id,
                shards[r],
                ident,
                run_id="r2",
                engine_id="megatron",
                attempt_id=2,
                rank=r,
                placement_offset=r,
            ),
        )
        for r in range(2)
    ]
    replica = [
        RankArtifact(
            rank=r,
            group="tp2",
            artifact=make_artifact(
                case_id,
                rows,
                ident,
                run_id="r3",
                engine_id="megatron",
                attempt_id=3,
                rank=r,
                placement_offset=r,
            ),
        )
        for r in range(2)
    ]
    return {"ranks_base": base, "ranks_partition": partition, "ranks_replica": replica}


def _case_learned_basic(seed: int, t_rows: int) -> dict[str, Any]:
    """Build shared learned-router material for all validation stages."""
    rows, meta = make_learned_route_rows("learned_basic", seed=seed, t_rows=t_rows, layers=(3,))
    ident = _identity_from_meta(meta)
    art_a = make_artifact(
        "learned_basic", rows, ident, run_id="r1", engine_id="megatron", attempt_id=1
    )
    # same config re-run: same seeds -> bit-identical artifact
    art_b = make_artifact(
        "learned_basic", rows, ident, run_id="r1", engine_id="megatron", attempt_id=1
    )
    # perturbed: padding + repack + different run metadata (L2)
    art_p = make_artifact(
        "learned_basic",
        repack_rows(rows, batch_size=2),
        ident,
        run_id="r2",
        engine_id="miles",
        attempt_id=2,
        placement_offset=1,
        padding_rows=4,
    )
    return {
        "case_id": "learned_basic",
        "meta": meta,
        "rows": rows,
        "identity": ident,
        "run_a": art_a,
        "run_b": art_b,
        "run_p": art_p,
        "oracle_rows": rows,
        **_ws2_material("learned_basic", rows, ident),
    }


# DSV4-Flash router constants used by the synthetic cases.
_SCALE = 1.5


def _case_hash_basic(seed: int, t_rows: int) -> dict[str, Any]:
    """Build hash-router material using a deterministic tid2eid stand-in."""
    e = 256
    table_fp = hashlib.sha256(f"t-{seed + 1}".encode()).hexdigest()
    rows: list[RouteRow] = []
    for t in range(t_rows):
        # hash router: token -> deterministic expert assignment (stand-in
        # until the tid2eid table lands; L2/L3a logic under test is identical)
        ids = [(t * 31 + i) % e for i in range(K)]
        for i in range(K):
            rows.append(
                RouteRow(
                    global_token_id=t,
                    input_token_id=t,
                    absolute_layer=0,
                    router_mode="hash",
                    topk_index=i,
                    logical_expert_id=int(ids[i]),
                    valid=True,
                    invalid_reason=None,
                    route_weight=_SCALE / K,
                    weight_score=_SCALE / K,
                    selection_score=float(int(ids[i])),
                )
            )
    meta = {
        "case_id": "hash_basic",
        "checkpoint_id": "ckpt-hash",
        "weight_id": "w-hash",
        "absolute_layer": 0,
        "router_mode": "hash",
        "table_fingerprint": table_fp,
        # hash layers have no correction bias: the irrelevant fingerprint
        # stays absent (encoded as present=false), never a fabricated value
        "bias_fingerprint": None,
        "logit_round_point": "fp32_direct",
        "tie_break_policy": "q_desc_id_asc",
        "capacity_policy": "dropless_v1",
    }
    ident = _identity_from_meta(meta)
    art_a = make_artifact(
        "hash_basic", rows, ident, run_id="r1", engine_id="megatron", attempt_id=1
    )
    art_b = make_artifact(
        "hash_basic", rows, ident, run_id="r1", engine_id="megatron", attempt_id=1
    )
    art_p = make_artifact(
        "hash_basic",
        repack_rows(rows, batch_size=2),
        ident,
        run_id="r2",
        engine_id="miles",
        attempt_id=2,
        placement_offset=1,
        padding_rows=4,
    )
    return {
        "case_id": "hash_basic",
        "meta": meta,
        "rows": rows,
        "identity": ident,
        "run_a": art_a,
        "run_b": art_b,
        "run_p": art_p,
        "oracle_rows": rows,
        **_ws2_material("hash_basic", rows, ident),
    }


_CASES = {
    "learned_basic": _case_learned_basic,
    "hash_basic": _case_hash_basic,
}

# --- validation-stage drivers --------------------------------------------------


def _run_case(case: dict[str, Any], stages: tuple[str, ...]) -> list[ValidationReport]:
    """Run selected stages in canonical order and retain every report.

    The CLI does not stop the whole case after its first failed report. Each
    stage owns its prerequisites, and collecting all reports makes one run
    useful across independent validation surfaces.
    """
    out: list[ValidationReport] = []
    cid = case["case_id"]
    if "L1" in stages:
        out.append(run_l1_repeat(cid, case["run_a"], case["run_b"]))
    if "L2" in stages:
        out.append(run_l2_invariance(cid, case["run_a"], case["run_p"], variant="pad+repack"))
    if "L3a" in stages:
        out.append(run_l3a_oracle(cid, case["oracle_rows"], case["rows"]))
    if "L3b" in stages:
        out.append(_run_l3b(case))
    if "WS2-rank" in stages:
        out.append(check_rank_completeness(cid, case["ranks_partition"], [0, 1]))
    if "WS2-cross" in stages:
        out.append(
            run_ws2_cross_config(
                cid,
                case["ranks_base"],
                case["ranks_partition"],
                base_config="dp1",
                other_config="dp2",
            )
        )
        out.append(
            run_ws2_cross_config(
                cid,
                case["ranks_base"],
                case["ranks_replica"],
                base_config="tp1",
                other_config="tp2",
                ownership="replica",
            )
        )
    return out


def _run_l3b(case: dict[str, Any]) -> ValidationReport:
    """Fail L3b closed while recorded cross-engine anchors are unavailable.

    Comparing a synthetic object with itself cannot provide independent Miles
    provenance, so it must not satisfy the formal dual-engine gate.
    """
    return make_fail(
        "L3b",
        case["case_id"],
        RouterVerdict.MISSING_PROVENANCE,
        "Miles anchor/recorded dual-engine artifacts are not published; "
        "synthetic self-comparison cannot satisfy formal L3b",
        anchor_pending=True,
    )


def parse_args() -> argparse.Namespace:
    """Define CLI selections; ``main`` validates case and stage names."""
    parser = argparse.ArgumentParser(description="MoE router validation driver.")
    parser.add_argument(
        "--cases", default="smoke", help="Comma list or 'smoke' (= learned_basic,hash_basic)."
    )
    parser.add_argument(
        "--stages",
        default=",".join(VALIDATION_STAGES),
        help=f"Comma list from {VALIDATION_STAGES}.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rows", type=int, default=8, help="Tokens per case.")
    parser.add_argument(
        "--json", action="store_true", help="Print the full structured report as JSON."
    )
    return parser.parse_args()


def main() -> None:
    """Validate selections, run stages, print reports, and exit strictly."""
    args = parse_args()
    wanted = (
        ["learned_basic", "hash_basic"]
        if args.cases == "smoke"
        else [c.strip() for c in args.cases.split(",") if c.strip()]
    )
    unknown = [c for c in wanted if c not in _CASES]
    if unknown:
        print(f"error: unknown cases: {unknown}", file=sys.stderr)
        sys.exit(2)

    stages = tuple(stage.strip() for stage in args.stages.split(",") if stage.strip())
    bad = [stage for stage in stages if stage not in VALIDATION_STAGES]
    if bad or not stages:
        print(
            f"error: unknown or empty stages: {bad or '<empty>'} (choose from {VALIDATION_STAGES})",
            file=sys.stderr,
        )
        sys.exit(2)
    if not wanted:
        print("error: no cases selected (empty --cases)", file=sys.stderr)
        sys.exit(2)

    # Builders only prepare material; validation rules remain in the runner.
    reports: list[ValidationReport] = []
    for name in wanted:
        builder = _CASES[name]
        case = builder(args.seed, args.rows)
        reports.extend(_run_case(case, stages))

    if args.json:
        print(json.dumps([_report_to_dict(r) for r in reports], ensure_ascii=False, indent=2))
    else:
        for r in reports:
            print(r.summary_line())
    code = _exit_code(reports)
    program_name = pathlib.Path(sys.argv[0]).stem
    print(
        f"{program_name}: cases={len(wanted)} stages={','.join(stages)} "
        f"passed={sum(r.passed for r in reports)}/{len(reports)} exit={code}",
        file=sys.stderr if args.json else sys.stdout,
    )

    sys.exit(code)


def _report_to_dict(r: ValidationReport) -> dict[str, Any]:
    """Convert report objects into a stable JSON-serializable structure."""
    fm = r.first_mismatch
    return {
        "stage": r.stage,
        "case_id": r.case_id,
        "passed": r.passed,
        "verdict": r.verdict.name if r.verdict else None,
        "detail": r.detail,
        "first_mismatch": (
            {
                "key": fm.key._asdict() if fm.key else None,
                "owner": fm.owner,
                "issue": fm.issue,
                "boundary": fm.boundary,
                "phase": fm.phase,
                "artifact": fm.artifact,
                "detail": fm.detail,
            }
            if fm
            else None
        ),
        "extra": r.extra,
    }


if __name__ == "__main__":
    main()
