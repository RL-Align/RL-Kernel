#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Compare ``qwen3_next_logprob_parity.py`` runs made under different NCCL settings.

    python tools/validation/models/qwen3_next/qwen3_next_logprob_parity_compare.py \\
        --runs <dir per NCCL setting>... --out report.json

Every point except the raw-NCCL negative control must have one digest per rank
across all settings, and the TP4 selected logprob and LSE must equal TP1. The report records how
many distinct digests each point produced, per rank.
"""

import argparse
import json
from pathlib import Path

CONTROL = "o_proj_partials.nccl_all_reduce"


def load(run: Path):
    ranks = sorted(run.glob("rank-*.json"))
    if [p.name for p in ranks] != [f"rank-{r}.json" for r in range(4)]:
        raise SystemExit(f"{run} does not hold four rank reports")
    return [json.loads(p.read_text()) for p in ranks]


def compare(runs):
    reports = {run.name: load(run) for run in runs}
    first = next(iter(reports.values()))
    commits = {r["rl_kernel_commit"] for rs in reports.values() for r in rs}
    dirty = any(r["tracked_tree_dirty"] for rs in reports.values() for r in rs)
    one_commit = len(commits) == 1
    points = sorted(k for k, v in first[0]["points"].items() if isinstance(v, str))
    table = {}
    for point in points:
        per_rank = []
        for rank in range(4):
            per_rank.append(len({rs[rank]["points"][point] for rs in reports.values()}))
        table[point] = {"distinct_digests_per_rank": per_rank}
    deterministic = [p for p in points if p != CONTROL]
    invariant = all(max(table[p]["distinct_digests_per_rank"]) == 1 for p in deterministic)
    tp1 = all(
        r["points"][f"logprob.tp4.{part}"] == r["points"][f"logprob.tp1.{part}"]
        for rs in reports.values()
        for r in rs
        for part in ("selected", "lse")
    )
    return {
        "kind": "qwen3_next_distributed_logprob_parity",
        "rfc": 428,
        "rl_kernel_commit": next(iter(commits)) if one_commit else sorted(commits),
        "tracked_tree_dirty": dirty,
        "environment": first[0]["environment"],
        "tokens": first[0]["tokens"],
        "settings": {name: rs[0]["nccl_settings"] for name, rs in reports.items()},
        "points": table,
        "deterministic_points_invariant": invariant,
        "tp4_equals_tp1": tp1,
        "nccl_all_reduce_varies": max(table[CONTROL]["distinct_digests_per_rank"]) > 1,
        "entropy_tp4_equals_tp1": all(
            r["points"]["entropy.tp4_equals_tp1"] for rs in reports.values() for r in rs
        ),
        "full_logits_match_gathered": all(
            r["points"]["full_logits.matches_gathered"] for rs in reports.values() for r in rs
        ),
        "passed": invariant and tp1 and one_commit and not dirty,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"passed={report['passed']} invariant={report['deterministic_points_invariant']} "
        f"tp4==tp1={report['tp4_equals_tp1']} nccl_varies={report['nccl_all_reduce_varies']}"
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
