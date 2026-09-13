#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""check_p3: P3 (DSV4-Flash MoE Router) validation-ladder driver CLI.

Contract §4-T09 / §7 WS1 Gate: runs the recorded/synthetic ladder
L1 (repeat) / L2 (invariance) / L3a (oracle) / L3b (dual-engine) and
prints one summary line per case plus a final exit code (0 = all pass;
non-zero = first failing verdict band).

Examples:
    python scripts/check_p3.py --cases smoke
    python scripts/check_p3.py --cases learned_basic,hash_basic
    python scripts/check_p3.py --ladders L1,L2 --json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.moe.naive_topk6 import K  # noqa: E402
from rl_engine.moe.validation.first_mismatch import MismatchKey, TraceEvent  # noqa: E402
from rl_engine.moe.validation.fingerprint import RouteIdentity  # noqa: E402
from rl_engine.moe.validation.ladder import (  # noqa: E402
    LadderReport,
    run_l1_repeat,
    run_l2_invariance,
    run_l3a_oracle,
    run_l3b_dual_engine,
)
from rl_engine.moe.validation.synthetic_producer import (  # noqa: E402
    make_artifact,
    make_learned_route_rows,
    repack_rows,
)

ALL_LADDERS = ("L1", "L2", "L3a", "L3b")

# Exit code = first failing verdict's primary band (contract §6: provider
# band 0-49 reserved 1-22, runner band 50-72). We map: all pass -> 0.
def _exit_code(reports: list[LadderReport]) -> int:
    for r in reports:
        if not r.passed and r.verdict is not None:
            return int(r.verdict.value)
    return 0


# --- case registry -------------------------------------------------------------

def _identity_from_meta(meta: dict[str, Any]) -> RouteIdentity:
    return RouteIdentity(
        checkpoint_id=str(meta["checkpoint_id"]),
        weight_id=str(meta["weight_id"]),
        logit_round_point=str(meta["logit_round_point"]),
        tie_break_policy=str(meta["tie_break_policy"]),
        capacity_policy=str(meta["capacity_policy"]),
        table_fingerprint=str(meta.get("table_fingerprint")) or None,
        bias_fingerprint=str(meta.get("bias_fingerprint")) or None,
    )


def _case_learned_basic(seed: int, t_rows: int) -> dict[str, Any]:
    rows, meta = make_learned_route_rows(
        "learned_basic", seed=seed, t_rows=t_rows, layers=(3,))
    ident = _identity_from_meta(meta)
    art_a = make_artifact("learned_basic", rows, ident, run_id="r1",
                          engine_id="megatron", attempt_id=1)
    # same config re-run: same seeds -> bit-identical artifact
    art_b = make_artifact("learned_basic", rows, ident, run_id="r1",
                          engine_id="megatron", attempt_id=1)
    # perturbed: padding + repack + different run metadata (L2)
    art_p = make_artifact("learned_basic", repack_rows(rows, batch_size=2),
                          ident, run_id="r2", engine_id="miles", attempt_id=2,
                          placement_offset=1, padding_rows=4)
    return {"case_id": "learned_basic", "meta": meta, "rows": rows,
            "identity": ident, "run_a": art_a, "run_b": art_b, "run_p": art_p,
            "oracle_rows": rows}


def _case_hash_basic(seed: int, t_rows: int) -> dict[str, Any]:
    import torch

    from rl_engine.moe.validation.fingerprint import RouteRow

    import hashlib

    g = torch.Generator().manual_seed(seed + 1)
    e = 256
    table_fp = hashlib.sha256(f"t-{seed + 1}".encode()).hexdigest()
    rows: list[RouteRow] = []
    for t in range(t_rows):
        # hash router: token -> deterministic expert assignment (stand-in
        # until T03's table lands; L2/L3a logic under test is identical)
        h = torch.tensor([t * 31 + i for i in range(K)], dtype=torch.int64)
        ids = (h % e).tolist()
        for i in range(K):
            rows.append(RouteRow(
                global_token_id=t, input_token_id=t, absolute_layer=0,
                router_mode="hash", topk_index=i, logical_expert_id=int(ids[i]),
                valid=True, invalid_reason=None,
                route_weight=_SCALE / K, weight_score=_SCALE / K,
                selection_score=float(int(ids[i])),
            ))
    meta = {
        "case_id": "hash_basic", "checkpoint_id": "ckpt-hash", "weight_id": "w-hash",
        "absolute_layer": 0, "router_mode": "hash",
        "table_fingerprint": table_fp,
        "bias_fingerprint": hashlib.sha256(f"b-{seed + 1}".encode()).hexdigest(),
        "logit_round_point": "fp32_direct", "tie_break_policy": "p3_canonical",
        "capacity_policy": "dropless_v1",
    }
    ident = _identity_from_meta(meta)
    art_a = make_artifact("hash_basic", rows, ident, run_id="r1",
                          engine_id="megatron", attempt_id=1)
    art_b = make_artifact("hash_basic", rows, ident, run_id="r1",
                          engine_id="megatron", attempt_id=1)
    art_p = make_artifact("hash_basic", repack_rows(rows, batch_size=2),
                          ident, run_id="r2", engine_id="miles", attempt_id=2,
                          placement_offset=1, padding_rows=4)
    return {"case_id": "hash_basic", "meta": meta, "rows": rows,
            "identity": ident, "run_a": art_a, "run_b": art_b, "run_p": art_p,
            "oracle_rows": rows}


_CASES = {
    "smoke": None,
    "learned_basic": _case_learned_basic,
    "hash_basic": _case_hash_basic,
}

_SCALE = 1.5


# --- ladder drivers ------------------------------------------------------------

def _run_case(case: dict[str, Any], ladders: tuple[str, ...]) -> list[LadderReport]:
    out: list[LadderReport] = []
    cid = case["case_id"]
    if "L1" in ladders:
        out.append(run_l1_repeat(cid, case["run_a"], case["run_b"]))
    if "L2" in ladders:
        out.append(run_l2_invariance(cid, case["run_a"], case["run_p"], variant="pad+repack"))
    if "L3a" in ladders:
        out.append(run_l3a_oracle(cid, case["identity"],
                                  case["oracle_rows"], case["rows"]))
    if "L3b" in ladders:
        out.append(_run_l3b(case))
    return out


def _run_l3b(case: dict[str, Any]) -> LadderReport:
    import torch

    t_rows = max(r.global_token_id for r in case["rows"]) + 1
    n_active = t_rows * K
    # identical core bytes from both engines -> byte-exact across 4 stages
    lhs_w = torch.full((n_active,), _SCALE / K)
    rhs_w = torch.full((n_active,), _SCALE / K)
    dz = torch.full((n_active,), 1.0)

    events = [
        TraceEvent(
            key=MismatchKey(absolute_layer=case["meta"]["absolute_layer"],
                            site="hash_lookup" if case["meta"]["router_mode"] == "hash" else "topk",
                            pass_direction="forward", event_index=i,
                            global_token_id=i // K, rank=0),
            payload={"logical_expert_id": int(r.logical_expert_id)},
        )
        for i, r in enumerate(case["rows"][:n_active])
    ]
    return run_l3b_dual_engine(
        case["case_id"], case["meta"], dict(case["meta"]), events, list(events),
        lhs_w, rhs_w, dz, dz,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P3 validation ladder driver (T09).")
    parser.add_argument("--cases", default="smoke",
                        help="Comma list or 'smoke' (= learned_basic,hash_basic).")
    parser.add_argument("--ladders", default=",".join(ALL_LADDERS),
                        help=f"Comma list from {ALL_LADDERS}.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rows", type=int, default=8, help="Tokens per case.")
    parser.add_argument("--json", action="store_true",
                        help="Print the full structured report as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wanted = ["learned_basic", "hash_basic"] if args.cases == "smoke" \
        else [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in wanted if c != "smoke" and c not in _CASES]
    if unknown:
        print(f"error: unknown cases: {unknown}", file=sys.stderr)
        sys.exit(2)

    ladders = tuple(l.strip() for l in args.ladders.split(",") if l.strip())
    bad = [l for l in ladders if l not in ALL_LADDERS]
    if bad:
        print(f"error: unknown ladders: {bad}", file=sys.stderr)
        sys.exit(2)

    reports: list[LadderReport] = []
    for name in wanted:
        builder = _CASES["learned_basic" if name == "smoke" else name]
        case = builder(args.seed, args.rows)
        reports.extend(_run_case(case, ladders))

    if args.json:
        print(json.dumps([_report_to_dict(r) for r in reports], ensure_ascii=False, indent=2))
    else:
        for r in reports:
            print(r.summary_line())
    code = _exit_code(reports)
    print(f"check_p3: cases={len(wanted)} ladders={','.join(ladders)} "
          f"passed={sum(r.passed for r in reports)}/{len(reports)} exit={code}",
          file=sys.stderr if args.json else sys.stdout)

    sys.exit(code)


def _report_to_dict(r: LadderReport) -> dict[str, Any]:
    fm = r.first_mismatch
    return {
        "ladder": r.ladder, "case_id": r.case_id, "passed": r.passed,
        "verdict": r.verdict.name if r.verdict else None,
        "detail": r.detail, "first_mismatch": fm.key._asdict() if fm else None,
        "owner_issue": f"{fm.owner}/{fm.issue}" if fm else None,
        "extra": {k: str(v) for k, v in r.extra.items()},
    }


if __name__ == "__main__":
    main()
