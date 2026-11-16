#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize and compare runs of the 8-GPU DSv4 LoRA comparison (README.md).

Each run directory holds verl's file-logger JSONL (``metrics.jsonl``, one
``{"step": n, "data": {...}}`` per line). For one run: the per-step bitwise table and the
mean timings, written to ``<run>/summary.md`` / ``summary.json``. For several runs: also a
side-by-side table, the first run being the reference for the ratios.

    python3 compare.py OUT/rlkernel_* OUT/baseline_*

Bitwise: ``training/rollout_logprobs_mismatch_count`` is the number of response tokens whose
rollout (vLLM) and old-policy recompute (actor) FP32 log-probs differ; 0 at every step is the
claim. ``actor/old_logprobs_mismatch_count`` (verl #7483's commit) does the same for the
grad-enabled update forward against the recompute. Timings leave out the first ``--warmup``
steps (JIT compilation, allocator growth).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

MISMATCH = "training/rollout_logprobs_mismatch_count"
UPDATE_MISMATCH = "actor/old_logprobs_mismatch_count"
PER_STEP = [
    MISMATCH,
    UPDATE_MISMATCH,
    "training/rollout_logprobs_diff_max",
    "training/rollout_actor_probs_pearson_corr",
    "actor/grad_norm",
    "response_length/mean",
    "critic/score/mean",
    "timing_s/step",
]
TIMINGS = [
    "timing_s/step",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "perf/throughput",
    "actor/perf/max_memory_allocated_gb",
    "actor/perf/max_memory_reserved_gb",
    "perf/total_num_tokens",
    "response_length/mean",
]


def load(run: pathlib.Path) -> dict[int, dict]:
    path = run / "metrics.jsonl" if run.is_dir() else run
    steps = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            steps.setdefault(int(rec["step"]), {}).update(rec["data"])
    return dict(sorted(steps.items()))


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def summarize(run: pathlib.Path, warmup: int) -> dict:
    steps = load(run)
    mism = {s: d.get(MISMATCH) for s, d in steps.items()}
    upd = {s: d.get(UPDATE_MISMATCH) for s, d in steps.items() if UPDATE_MISMATCH in d}
    timed = [d for s, d in steps.items() if s > warmup] or list(steps.values())
    means = {}
    for key in TIMINGS + sorted({k for d in timed for k in d if k.startswith("timing_s/")}):
        vals = [d[key] for d in timed if isinstance(d.get(key), (int, float))]
        if vals and key not in means:
            means[key] = statistics.fmean(vals)
    known = [v for v in mism.values() if v is not None]
    out = {
        "run": str(run),
        "steps": len(steps),
        "bitwise_every_step": bool(known) and len(known) == len(steps) and max(known) == 0,
        "mismatched_tokens_total": sum(known),
        "first_mismatch_step": next((s for s, v in mism.items() if v), None),
        "update_forward_bitwise": (max(upd.values()) == 0) if upd else None,
        "steps_with_nonzero_grad_norm": sum(
            1 for d in steps.values() if (d.get("actor/grad_norm") or 0) > 0
        ),
        "score_first_last": (
            [
                steps[min(steps)].get("critic/score/mean"),
                steps[max(steps)].get("critic/score/mean"),
            ]
            if steps
            else None
        ),
        "mean_after_warmup": means,
    }
    lines = [f"# {run.name}", "", "| step | " + " | ".join(PER_STEP) + " |"]
    lines.append("|" + "---|" * (len(PER_STEP) + 1))
    for s, d in steps.items():
        lines.append(f"| {s} | " + " | ".join(fmt(d.get(k)) for k in PER_STEP) + " |")
    lines += ["", f"Bitwise on every step: **{'YES' if out['bitwise_every_step'] else 'NO'}**"]
    for k in (
        "mismatched_tokens_total",
        "first_mismatch_step",
        "update_forward_bitwise",
        "steps_with_nonzero_grad_norm",
        "score_first_last",
    ):
        lines.append(f"- {k}: {out[k]}")
    lines += ["", f"Means after {warmup} warmup steps:", ""]
    lines += [f"- {k}: {fmt(v)}" for k, v in means.items()]
    out["markdown"] = "\n".join(lines)
    if run.is_dir():
        (run / "summary.md").write_text(out["markdown"] + "\n")
        (run / "summary.json").write_text(
            json.dumps({k: v for k, v in out.items() if k != "markdown"}, indent=2) + "\n"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=pathlib.Path)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    sums = [summarize(r, args.warmup) for r in args.runs]
    for s in sums:
        print(s["markdown"], "\n")
    if len(sums) > 1:
        ref = sums[0]["mean_after_warmup"]
        names = [pathlib.Path(s["run"]).name for s in sums]
        print("# Comparison (ratio to the first run)\n")
        print("| metric | " + " | ".join(names) + " |")
        print("|" + "---|" * (len(names) + 1))
        print(
            "| bitwise every step | "
            + " | ".join("YES" if s["bitwise_every_step"] else "NO" for s in sums)
            + " |"
        )
        print(
            "| mismatched tokens | "
            + " | ".join(str(s["mismatched_tokens_total"]) for s in sums)
            + " |"
        )
        for key in TIMINGS:
            cells = []
            for s in sums:
                v = s["mean_after_warmup"].get(key)
                r = ref.get(key)
                cells.append(fmt(v) + (f" ({v / r:.2f}x)" if v and r and s is not sums[0] else ""))
            print(f"| {key} | " + " | ".join(cells) + " |")
    return 0 if all(s["bitwise_every_step"] for s in sums[:1]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
