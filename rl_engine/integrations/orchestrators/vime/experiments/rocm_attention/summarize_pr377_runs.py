"""Summarize P/P and R/R PR377-workload runs into consistency and performance tables.

Reads the ``perf N:`` dictionaries Vime prints into ``launcher.log`` for each
arm, averages them across rounds and renders the same two tables used for the
CUDA PR377 report: an exactness table (mismatch count, max |dlogp|,
``torch.equal``) and a per-phase mean-time table with the R/R-relative-to-P/P
column.

Example::

    python -m \
        rl_engine.integrations.orchestrators.vime.experiments.rocm_attention.summarize_pr377_runs \
      --pp-run /app/model/vime-runs/pp-30round --rr-run /app/model/vime-runs/rr-30round
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

PERF_LINE = re.compile(r"perf (\d+): (\{.*\})\s*$")

ROWS = [
    ("rollout time", "perf/rollout_time", "s", "lower"),
    ("effective tokens/GPU/s", "perf/effective_tokens_per_gpu_per_sec", "", "higher"),
    ("update weights", "perf/update_weights_time", "s", "lower"),
    ("reference log probs", "perf/ref_log_probs_time", "s", "lower"),
    ("log probs", "perf/log_probs_time", "s", "lower"),
    ("actor train", "perf/actor_train_time", "s", "lower"),
    ("train time", "perf/train_time", "s", "lower"),
    ("actor train tok/s", "perf/actor_train_tok_per_s", "", "higher"),
    ("end-to-end step", "perf/step_time", "s", "lower"),
]


def load_rounds(run_dir: Path) -> dict[int, dict[str, float]]:
    arms = sorted((run_dir / "arms").glob("*"))
    if not arms:
        raise FileNotFoundError(f"no arms under {run_dir}")
    log = arms[0] / "launcher.log"
    rounds: dict[int, dict[str, float]] = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PERF_LINE.search(line)
        if not match:
            continue
        payload = ast.literal_eval(match.group(2))
        bucket = rounds.setdefault(int(match.group(1)), {})
        for key, value in payload.items():
            if key.startswith("perf/") and isinstance(value, (int, float)):
                bucket[key] = float(value)
    return rounds


def load_metrics(run_dir: Path) -> dict:
    arms = sorted((run_dir / "arms").glob("*"))
    validation = json.loads((arms[0] / "validation.json").read_text(encoding="utf-8"))
    metrics = validation.get("metrics", {}) or {}
    return {
        "passed": validation.get("passed"),
        "errors": validation.get("errors", []),
        "metrics": metrics,
    }


def _metric(metrics: dict, *names, default=None):
    for name in names:
        if name in metrics:
            return metrics[name]
    aggregate = metrics.get("aggregate") or metrics.get("summary") or {}
    for name in names:
        if name in aggregate:
            return aggregate[name]
    return default


def mean_rows(rounds: dict[int, dict[str, float]], keys: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in keys:
        values = [r[key] for r in rounds.values() if key in r]
        out[key] = sum(values) / len(values) if values else None
    return out


def fmt(value, unit):
    if value is None:
        return "n/a"
    return f"{value:.6f}{(' ' + unit) if unit else ''}"


def relative(pp, rr, direction):
    if pp is None or rr is None or pp == 0:
        return "n/a"
    if direction == "lower":
        delta = (rr - pp) / pp
        return f"{-delta * 100:.2f}% faster" if delta < 0 else f"{delta * 100:.2f}% slower"
    delta = (rr - pp) / pp
    return f"{delta * 100:.2f}% higher" if delta > 0 else f"{-delta * 100:.2f}% lower"


def render(pp_dir: Path, rr_dir: Path) -> str:
    pp_rounds = load_rounds(pp_dir)
    rr_rounds = load_rounds(rr_dir)
    pp_info = load_metrics(pp_dir)
    rr_info = load_metrics(rr_dir)
    n_pp, n_rr = len(pp_rounds), len(rr_rounds)
    lines = []
    lines.append(f"## Consistency results over {min(n_pp, n_rr)} rounds\n")
    lines.append("| Configuration | Mismatch Count | Max \\|Δlogp\\| | torch.equal |")
    lines.append("|---|---:|---:|:---:|")
    for label, info in (("P/P native", pp_info), ("R/R strict", rr_info)):
        m = info["metrics"]
        mismatch = _metric(m, "mismatch_count", "mismatched_tokens", "mismatch_tokens")
        total = _metric(m, "element_count", "token_count", "compared_tokens", "total_tokens")
        max_abs = _metric(m, "max_abs_diff", "max_abs_drift", "max_abs_delta")
        equal = _metric(m, "torch_equal", "bitwise_equal", "exact")
        count = f"{mismatch} / {total}" if total is not None else f"{mismatch}"
        lines.append(
            f"| {label} | {count} | {max_abs if max_abs is None else f'{max_abs:g}'} | "
            f"{str(equal).lower()} |"
        )
    lines.append("")
    lines.append(f"## Average performance over {min(n_pp, n_rr)} rounds\n")
    lines.append(f"(P/P rounds={n_pp}, R/R rounds={n_rr})\n")
    lines.append("| Metric | P/P native | R/R strict | R/R relative to P/P |")
    lines.append("|---|---:|---:|---:|")
    keys = [row[1] for row in ROWS]
    pp_mean = mean_rows(pp_rounds, keys)
    rr_mean = mean_rows(rr_rounds, keys)
    for label, key, unit, direction in ROWS:
        pp_value, rr_value = pp_mean[key], rr_mean[key]
        if pp_value is None and rr_value is None:
            continue
        lines.append(
            f"| {label} | {fmt(pp_value, unit)} | {fmt(rr_value, unit)} | "
            f"{relative(pp_value, rr_value, direction)} |"
        )
    lines.append("")
    lines.append("### Per-round details\n")
    lines.append(
        "| round | P/P rollout | R/R rollout | P/P actor train | R/R actor train "
        "| P/P step | R/R step |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for index in sorted(set(pp_rounds) | set(rr_rounds)):
        pp = pp_rounds.get(index, {})
        rr = rr_rounds.get(index, {})

        def cell(bucket, key):
            return f"{bucket[key]:.3f}" if key in bucket else "-"

        lines.append(
            f"| {index} | {cell(pp, 'perf/rollout_time')} | {cell(rr, 'perf/rollout_time')} | "
            f"{cell(pp, 'perf/actor_train_time')} | {cell(rr, 'perf/actor_train_time')} | "
            f"{cell(pp, 'perf/step_time')} | {cell(rr, 'perf/step_time')} |"
        )
    lines.append("")
    lines.append(
        f"validation: P/P passed={pp_info['passed']} errors={len(pp_info['errors'])}; "
        f"R/R passed={rr_info['passed']} errors={len(rr_info['errors'])}"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pp-run", type=Path, required=True)
    parser.add_argument("--rr-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    text = render(args.pp_run, args.rr_run)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
