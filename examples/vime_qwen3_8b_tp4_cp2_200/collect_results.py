# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Collect sealed VIME ablation runs into reproducible CSV and JSON tables."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


RECORD_RE = re.compile(r"\b(rollout|step|perf)\s+(\d+):\s+(\{.*\})\s*$")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _records(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {
        "rollout": {},
        "step": {},
        "perf": {},
    }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RECORD_RE.search(line)
        if not match:
            continue
        try:
            value = ast.literal_eval(match.group(3))
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            result[match.group(1)].setdefault(int(match.group(2)), {}).update(value)
    return result


def _value(mapping: dict[str, Any], name: str) -> float | None:
    value = mapping.get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _run_rows(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load(run_dir / "manifest.json")
    validation = _load(run_dir / "run-validation.json")
    if not validation.get("passed"):
        raise ValueError(f"sealed run has a failed validation: {run_dir}")
    records = _records(run_dir / "run.log")
    validation_rows = {int(row["step"]): row for row in validation["train_rollout_logprob"]["rows"]}
    steps = sorted(set(records["rollout"]) | set(records["step"]) | set(records["perf"]))
    rows = []
    for index in steps:
        rollout = records["rollout"].get(index, {})
        step = records["step"].get(index, {})
        perf = records["perf"].get(index, {})
        exact = validation_rows.get(index, {})
        rows.append(
            {
                "phase": run_dir.parent.name,
                "run_id": manifest["run_id"],
                "group": manifest["arm"]["group"],
                "seed": manifest["seed"],
                "rollout_seed": manifest["rollout_seed"],
                "step": index,
                "framework_consistency": manifest["arm"]["framework_use_rollout_logprobs"],
                "operator_case": manifest["arm"]["logp_case"],
                "reward": _value(rollout, "rollout/rewards"),
                "raw_reward": _value(rollout, "rollout/raw_reward"),
                "response_length": _value(rollout, "rollout/response_lengths"),
                "truncated_ratio": _value(rollout, "rollout/truncated"),
                "rollout_kl": _value(rollout, "rollout/kl"),
                "train_loss": _value(step, "train/loss"),
                "pg_loss": _value(step, "train/pg_loss"),
                "pg_clipfrac": _value(step, "train/pg_clipfrac"),
                "entropy": _value(step, "train/entropy_loss"),
                "ppo_kl": _value(step, "train/ppo_kl"),
                "grad_norm": _value(step, "train/grad_norm"),
                "mean_abs_dlogp": _value(step, "train/train_rollout_logprob_abs_diff"),
                "max_abs_dlogp": _value(step, "train/train_current_rollout_logprob_max_abs_diff"),
                "mismatch_count": exact.get("bitwise_mismatch_count"),
                "active_token_count": exact.get("active_token_count"),
                "rollout_time": _value(perf, "perf/rollout_time"),
                "train_time": _value(perf, "perf/train_time"),
                "step_time": _value(perf, "perf/step_time"),
                "update_weights_time": _value(perf, "perf/update_weights_time"),
                "actor_tokens_per_second": _value(perf, "perf/actor_train_tok_per_s"),
                "actor_train_tflops": _value(perf, "perf/actor_train_tflops"),
            }
        )
    run = {
        "phase": run_dir.parent.name,
        "run_id": manifest["run_id"],
        "group": manifest["arm"]["group"],
        "seed": manifest["seed"],
        "rollout_seed": manifest["rollout_seed"],
        "num_rollout": manifest["num_rollout"],
        "rl_kernel_revision": manifest["revisions"]["rl_kernel"],
        "vime_revision": manifest["revisions"]["vime"],
        "prompt_data_sha256": manifest["prompt_data_sha256"],
        "cudagraph_passed": validation["cudagraph"]["passed"],
        "validation_passed": validation["passed"],
        "offline_tensor_comparison": validation["offline_tensor_comparison"]["status"],
        "rounds_observed": len(rows),
    }
    return run, rows


def _finite(rows: list[dict[str, Any]], name: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["phase"]), str(row["group"]))].append(row)
    summaries = []
    for (phase, group), items in sorted(grouped.items()):
        mismatches = _finite(items, "mismatch_count")
        tokens = _finite(items, "active_token_count")
        mean_abs = _finite(items, "mean_abs_dlogp")
        weighted_abs_numerator = sum(
            float(item["mean_abs_dlogp"]) * float(item["active_token_count"])
            for item in items
            if item.get("mean_abs_dlogp") is not None and item.get("active_token_count") is not None
        )
        token_total = sum(tokens)
        summaries.append(
            {
                "phase": phase,
                "group": group,
                "run_count": len({str(item["run_id"]) for item in items}),
                "round_count": len(items),
                "active_token_exposure": token_total,
                "bitwise_mismatch_count": sum(mismatches),
                "bitwise_mismatch_rate": (sum(mismatches) / token_total if token_total else None),
                "mean_abs_dlogp_token_weighted": (
                    weighted_abs_numerator / token_total if token_total else None
                ),
                "max_abs_dlogp": max(_finite(items, "max_abs_dlogp"), default=None),
                "reward_mean": (
                    mean(_finite(items, "reward")) if _finite(items, "reward") else None
                ),
                "raw_reward_mean": (
                    mean(_finite(items, "raw_reward")) if _finite(items, "raw_reward") else None
                ),
                "truncated_ratio_mean": (
                    mean(_finite(items, "truncated_ratio"))
                    if _finite(items, "truncated_ratio")
                    else None
                ),
                "step_time_mean": (
                    mean(_finite(items, "step_time")) if _finite(items, "step_time") else None
                ),
                "actor_tokens_per_second_mean": (
                    mean(_finite(items, "actor_tokens_per_second"))
                    if _finite(items, "actor_tokens_per_second")
                    else None
                ),
                "unweighted_mean_abs_dlogp": mean(mean_abs) if mean_abs else None,
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dirs = sorted(
        path.parent
        for path in args.runs_root.rglob("COMPLETE")
        if (path.parent / "manifest.json").is_file()
        and (path.parent / "run-validation.json").is_file()
        and (path.parent / "run.log").is_file()
    )
    if not run_dirs:
        raise ValueError(f"no sealed runs found under {args.runs_root}")
    runs: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run, run_rounds = _run_rows(run_dir)
        runs.append(run)
        rounds.extend(run_rounds)
    summaries = _summaries(rounds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "runs.csv", runs)
    _write_csv(args.output_dir / "rounds.csv", rounds)
    _write_csv(args.output_dir / "summary.csv", summaries)
    report = {
        "schema_version": "rlkernel.vime_qwen3_8b_tp4_cp2_200.results.v1",
        "sealed_run_count": len(runs),
        "round_count": len(rounds),
        "summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
