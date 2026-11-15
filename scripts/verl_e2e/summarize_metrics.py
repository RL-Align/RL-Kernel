#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize a verl file-logger JSONL ({"step": n, "data": {...}} per line): bitwise verdict per
step and the step-time breakdown. Writes <out>.json and <out>.md; exit 0 only if every step is
bitwise (rollout == old-policy recompute, FP32 bits) and at least one step was logged.

    python3 summarize_metrics.py metrics.jsonl --out summary
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys

BITWISE = "training/rollout_logprob_bitwise_mismatch"
TOKENS = "training/rollout_logprob_tokens"
SHOW = [
    TOKENS,
    BITWISE,
    "training/rollout_logprob_absdiff_max",
    "training/rollout_probs_diff_max",
    "training/rollout_probs_diff_mean",
    "rollout_corr/k3_kl",
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "response_length/mean",
    "critic/score/mean",
]


def load(path):
    steps = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            data = rec.get("data", rec)
            step = rec.get("step", data.get("training/global_step"))
            steps.setdefault(step, {}).update(data)
    return [steps[k] | {"_step": k} for k in sorted(steps, key=lambda s: (s is None, s))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out", default="summary")
    a = ap.parse_args()
    rows = load(a.jsonl)
    rows = [r for r in rows if TOKENS in r or any(k.startswith("timing_s/") for k in r)]
    verdicts = []
    for r in rows:
        if TOKENS not in r:
            verdicts.append("no-metric")
        elif r[TOKENS] <= 0:
            verdicts.append("no-tokens")
        else:
            verdicts.append("BITWISE" if r.get(BITWISE) == 0 else "MISMATCH")
    timing_keys = sorted({k for r in rows for k in r if k.startswith(("timing_s/", "perf/"))})
    timing = {}
    for k in timing_keys:
        vals = [r[k] for r in rows[1:] if isinstance(r.get(k), (int, float))] or [
            r[k] for r in rows if isinstance(r.get(k), (int, float))
        ]
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        if vals:
            timing[k] = {
                "mean": statistics.mean(vals), "min": min(vals), "max": max(vals), "n": len(vals)
            }
    ok = bool(rows) and all(v == "BITWISE" for v in verdicts)
    total = sum(int(r.get(TOKENS, 0)) for r in rows)
    bad = sum(int(r.get(BITWISE, 0)) for r in rows)
    result = {
        "file": a.jsonl,
        "steps": len(rows),
        "bitwise_all_steps": ok,
        "tokens_compared": total,
        "tokens_mismatched": bad,
        "per_step": [
            {"step": r["_step"], "verdict": v, **{k: r.get(k) for k in SHOW}}
            for r, v in zip(rows, verdicts)
        ],
        "timing_excluding_first_step": timing,
    }
    with open(a.out + ".json", "w") as f:
        json.dump(result, f, indent=1)
    md = [
        f"# {a.jsonl}",
        "",
        f"**Bitwise on every step: {'YES' if ok else 'NO'}** "
        f"({total} response tokens compared, {bad} mismatched, {len(rows)} steps)",
        "",
        "| step | verdict | " + " | ".join(k.replace("training/", "") for k in SHOW) + " |",
        "|---|---|" + "---|" * len(SHOW),
    ]
    for p in result["per_step"]:
        cells = []
        for k in SHOW:
            v = p.get(k)
            cells.append("" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v)))
        md.append(f"| {p['step']} | {p['verdict']} | " + " | ".join(cells) + " |")
    md += ["", "| timing / perf (steps 2+) | mean | min | max |", "|---|---|---|---|"]
    for k, v in timing.items():
        md.append(f"| {k} | {v['mean']:.4g} | {v['min']:.4g} | {v['max']:.4g} |")
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
