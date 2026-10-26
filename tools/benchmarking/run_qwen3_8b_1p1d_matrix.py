# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Run a reproducible three-round Qwen3-8B regular/1P1D H100 matrix.

The matrix separates operator effects from rollout-topology effects:

* native + regular: README-style baseline;
* consistency + regular: RL-Kernel operator delta;
* native + 1p1d: Prefill/Decode topology control;
* consistency + 1p1d: target configuration.

Runs are serialized because every arm owns the same eight GPUs. Each run is
validated and sealed before the next arm starts. The benchmark is deliberately
report-only: correctness remains a hard gate, while timing and throughput are
written to the report for review.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

RECORD_RE = re.compile(r"\b(rollout|step|perf)\s+(\d+):\s+(\{.*\})\s*$")
ARM_ORDER = (
    ("native", "regular"),
    ("consistency", "regular"),
    ("native", "1p1d"),
    ("consistency", "1p1d"),
)


@dataclass(frozen=True)
class Arm:
    mode: str
    layout: str

    @property
    def name(self) -> str:
        return f"{self.layout}-{self.mode}"

    @property
    def memory_fraction(self) -> float:
        return 0.25 if self.layout == "1p1d" else 0.4


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


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
            payload = ast.literal_eval(match.group(3))
        except (SyntaxError, ValueError):
            continue
        if isinstance(payload, dict):
            result[match.group(1)].setdefault(int(match.group(2)), {}).update(payload)
    return result


def _nvidia_memory() -> list[int]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if output.returncode:
        return []
    values: list[int] = []
    for line in output.stdout.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return values


def _recovered_peak_memory(path: Path, gpu_count: int) -> list[int]:
    """Recover per-GPU training peaks when reusing a sealed run."""
    pattern = re.compile(r"'gpu':\s*'(?P<gpu>\d+)'.*?'used_GB':\s*(?P<used>[0-9.]+)")
    peaks: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        gpu = int(match.group("gpu"))
        used = float(match.group("used"))
        peaks[gpu] = max(peaks.get(gpu, 0.0), used)
    if len(peaks) != gpu_count or sorted(peaks) != list(range(gpu_count)):
        return []
    return [round(peaks[index] * 1024) for index in range(gpu_count)]


def _gpu_inventory() -> list[dict[str, str]]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if output.returncode:
        return []
    inventory = []
    for line in output.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 4:
            inventory.append(
                {
                    "index": fields[0],
                    "name": fields[1],
                    "driver": fields[2],
                    "memory_mib": fields[3],
                }
            )
    return inventory


class MemoryMonitor:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[list[int]] = []

    def _sample(self) -> None:
        while not self._stop.is_set():
            values = _nvidia_memory()
            if values:
                self.samples.append(values)
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "MemoryMonitor":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def peak(self) -> list[int]:
        width = max((len(sample) for sample in self.samples), default=0)
        return [
            max((sample[index] for sample in self.samples if len(sample) > index), default=0)
            for index in range(width)
        ]


def _run_process(command: list[str], log_path: Path) -> tuple[int, list[int], float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        with MemoryMonitor() as monitor:
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            returncode = process.wait()
    return returncode, monitor.peak(), time.perf_counter() - started


def _run_checked(command: list[str], log_path: Path) -> None:
    returncode, _peak, _elapsed = _run_process(command, log_path)
    if returncode:
        raise RuntimeError(f"command failed with return code {returncode}: {' '.join(command)}")


def _command(args: argparse.Namespace, arm: Arm, run_id: str) -> list[str]:
    command = [
        args.launcher,
        "run",
        "--profile",
        str(args.profile),
        "--output-root",
        str(args.runs_root),
        "--mode",
        arm.mode,
        "--rollouts",
        str(args.rounds),
        "--seed",
        str(args.seed),
        "--rollout-seed",
        str(args.rollout_seed),
        "--tp",
        "4",
        "--cp",
        "2",
        "--rollout-tp",
        "4",
        "--rollout-cp",
        "1",
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--kl-coef",
        str(args.kl_coef),
        "--max-response-len",
        str(args.max_response_len),
        "--max-tokens-per-gpu",
        str(args.max_tokens_per_gpu),
        "--vllm-gpu-memory-utilization",
        str(arm.memory_fraction),
        "--require-updates",
        "--run-id",
        run_id,
        "--wait",
    ]
    if arm.layout == "1p1d":
        command.insert(command.index("--output-root"), "--pd")
    return command


def _validate_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    return [
        args.launcher,
        "validate",
        "--profile",
        str(args.profile),
        "--run-dir",
        str(run_dir),
        "--seal",
    ]


def _require_sealed(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(run_dir / "manifest.json")
    validation = _load_json(run_dir / "run-validation.json")
    if validation.get("passed") is not True or not (run_dir / "COMPLETE").is_file():
        raise RuntimeError(f"run did not pass validation and sealing: {run_dir}")
    return manifest, validation


def _metric(mapping: Mapping[str, Any], key: str) -> float | None:
    return _finite(mapping.get(key))


def _round_rows(arm: Arm, manifest: Mapping[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    records = _records(run_dir / "run.log")
    validation = _load_json(run_dir / "run-validation.json")
    validation_rows = {
        int(row["step"]): row for row in validation.get("train_rollout_logprob", {}).get("rows", [])
    }
    steps = sorted(set(records["rollout"]) | set(records["step"]) | set(records["perf"]))
    expected = int(manifest["num_rollout"])
    if steps != list(range(expected)):
        raise RuntimeError(
            f"{run_dir}: expected perf records for steps 0..{expected - 1}, got {steps}"
        )
    rows = []
    for step in steps:
        rollout = records["rollout"].get(step, {})
        perf = records["perf"].get(step, {})
        exact = validation_rows.get(step, {})
        response_length = _metric(rollout, "rollout/response_len/mean")
        rollout_time = _metric(perf, "perf/rollout_time")
        response_tokens = None
        if response_length is not None:
            response_tokens = response_length * int(manifest["batching"]["global_batch_size"])
        rows.append(
            {
                "arm": arm.name,
                "mode": arm.mode,
                "layout": arm.layout,
                "run_id": manifest["run_id"],
                "step": step,
                "response_length_mean": response_length,
                "active_token_count": exact.get("active_token_count"),
                "mismatch_count": exact.get("bitwise_mismatch_count"),
                "rollout_time_s": rollout_time,
                "train_wait_time_s": _metric(perf, "perf/train_wait_time"),
                "train_time_s": _metric(perf, "perf/train_time"),
                "actor_train_time_s": _metric(perf, "perf/actor_train_time"),
                "update_weights_time_s": _metric(perf, "perf/update_weights_time"),
                "step_time_s": _metric(perf, "perf/step_time"),
                "rollout_tokens_per_gpu_s": _metric(perf, "perf/tokens_per_gpu_per_sec"),
                "actor_tokens_per_s": _metric(perf, "perf/actor_train_tok_per_s"),
                "response_tokens_total": response_tokens,
            }
        )
    return rows


def _validation_summary(validation: Mapping[str, Any]) -> dict[str, Any]:
    train = validation.get("train_rollout_logprob", {})
    return {
        "passed": validation.get("passed") is True,
        "errors": validation.get("errors", []),
        "pd_disaggregation_passed": validation.get("pd_disaggregation", {}).get("passed"),
        "cudagraph_passed": validation.get("cudagraph", {}).get("passed"),
        "train_rollout_logprob": {
            key: train.get(key)
            for key in (
                "passed",
                "require_exact",
                "mismatch_count",
                "max_abs_diff",
                "train_rollout_logprob_abs_diff",
                "sample_count",
            )
        },
    }


def _describe(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
        "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
    }


def _summary(
    rows: list[dict[str, Any]], peak_memory: list[int], validation: Mapping[str, Any]
) -> dict[str, Any]:
    def values(key: str, source: list[dict[str, Any]] = rows) -> list[float]:
        return [float(row[key]) for row in source if _finite(row.get(key)) is not None]

    steady = rows[1:] if len(rows) > 1 else rows
    return {
        "arm": rows[0]["arm"],
        "mode": rows[0]["mode"],
        "layout": rows[0]["layout"],
        "run_id": rows[0]["run_id"],
        "rounds": len(rows),
        "validation_passed": True,
        "validation": validation,
        "peak_memory_mib": peak_memory,
        "all_steps": {
            "step_time_s": _describe(values("step_time_s")),
            "rollout_time_s": _describe(values("rollout_time_s")),
            "actor_tokens_per_s": _describe(values("actor_tokens_per_s")),
            "rollout_tokens_per_gpu_s": _describe(values("rollout_tokens_per_gpu_s")),
        },
        "steady_steps_1_to_n": {
            "step_time_s": _describe(values("step_time_s", steady)),
            "rollout_time_s": _describe(values("rollout_time_s", steady)),
            "actor_tokens_per_s": _describe(values("actor_tokens_per_s", steady)),
        },
    }


def _delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any], metric: str) -> float | None:
    left = candidate["steady_steps_1_to_n"][metric]["mean"]
    right = baseline["steady_steps_1_to_n"][metric]["mean"]
    if left is None or right in (None, 0):
        return None
    return (float(left) / float(right) - 1.0) * 100.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    conditions = report["conditions"]
    summaries = report["summaries"]
    lines = [
        "# Qwen3-8B H100 three-round regular/1P1D benchmark",
        "",
        "This is a short development acceptance run, not a long-run performance claim.",
        "All four arms use the same Qwen3-8B workload and run serially on one 8xH100 node.",
        "",
        "## Conditions",
        "",
        f"- Rounds: `{conditions['rounds']}`; seed: `{conditions['seed']}`; "
        f"rollout seed: `{conditions['rollout_seed']}`",
        f"- Sampling: temperature `{conditions['temperature']}`, "
        f"top-p `{conditions['top_p']}`, KL coefficient `{conditions['kl_coef']}`",
        f"- Workload: max response `{conditions['max_response_len']}`, "
        f"max tokens/GPU `{conditions['max_tokens_per_gpu']}`, "
        f"global batch `{conditions['global_batch_size']}`",
        "- Regular vLLM memory fraction: `0.40`; 1P1D vLLM memory fraction: `0.25`",
        "",
        "## Results",
        "",
        "| Arm | Validation | Steady step median (s) | Steady step mean (s) | "
        "Rollout tok/GPU/s | Actor tok/s | Peak VRAM (MiB) |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        steady = item["steady_steps_1_to_n"]
        all_steps = item["all_steps"]
        lines.append(
            (
                "| {arm} | {validation} | {median:.3f} | {mean:.3f} | "
                "{rollout:.3f} | {actor:.3f} | {memory} |"
            ).format(
                arm=item["arm"],
                validation="passed" if item["validation_passed"] else "failed",
                median=float(steady["step_time_s"]["median"] or 0.0),
                mean=float(steady["step_time_s"]["mean"] or 0.0),
                rollout=float(all_steps["rollout_tokens_per_gpu_s"]["mean"] or 0.0),
                actor=float(steady["actor_tokens_per_s"]["mean"] or 0.0),
                memory=", ".join(str(value) for value in item["peak_memory_mib"]) or "unavailable",
            )
        )
    lines.extend(
        [
            "",
            "## Correctness",
            "",
            "Native controls allow numerical drift; consistency arms require exact "
            "train/rollout logprobs.",
            "",
            "| Arm | Validator | Exact required | Mismatches | Max abs diff |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in summaries:
        train = item["validation"]["train_rollout_logprob"]
        lines.append(
            "| {arm} | {passed} | {exact} | {mismatch} | {max_diff} |".format(
                arm=item["arm"],
                passed="passed" if item["validation"]["passed"] else "failed",
                exact=str(train.get("require_exact", "n/a")).lower(),
                mismatch=train.get("mismatch_count", "n/a"),
                max_diff=(
                    "n/a"
                    if train.get("max_abs_diff") is None
                    else f"{float(train['max_abs_diff']):.6g}"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Relative comparisons",
            "",
            "Negative step-time delta is faster. Positive throughput delta is faster.",
            "",
            "| Candidate | Baseline | Step-time delta | Actor throughput delta |",
            "|---|---|---:|---:|",
        ]
    )
    for item in report["comparisons"]:
        lines.append(
            "| {candidate} | {baseline} | {step} | {actor} |".format(
                candidate=item["candidate"],
                baseline=item["baseline"],
                step=(
                    "n/a"
                    if item["step_time_delta_pct"] is None
                    else f"{item['step_time_delta_pct']:+.2f}%"
                ),
                actor=(
                    "n/a"
                    if item["actor_tokens_per_s_delta_pct"] is None
                    else f"{item['actor_tokens_per_s_delta_pct']:+.2f}%"
                ),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--launcher", default=os.environ.get("RLK_LAUNCHER", "rlk"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rollout-seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--max-response-len", type=int, default=6912)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=4096)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=[f"{layout}-{mode}" for mode, layout in ARM_ORDER],
        default=[f"{layout}-{mode}" for mode, layout in ARM_ORDER],
        help="subset of matrix arms; default runs all four serially",
    )
    parser.add_argument("--suite-id", default=None)
    parser.add_argument(
        "--reuse-sealed",
        action="store_true",
        help="reuse existing sealed runs instead of launching them again",
    )
    return parser


def _parse_arm(value: str) -> Arm:
    layout, mode = value.split("-", 1)
    return Arm(mode=mode, layout=layout)


def _run_matrix(args: argparse.Namespace, arms: list[Arm]) -> list[dict[str, Any]]:
    args.runs_root.mkdir(parents=True, exist_ok=True)
    controller_dir = args.runs_root / "controller"
    suite_id = args.suite_id or datetime.now(timezone.utc).strftime("qwen3-8b-3r-%Y%m%dT%H%M%SZ")
    results = []
    inventory = _gpu_inventory()
    previous_report: dict[str, Any] = {}
    if args.reuse_sealed and (args.report_dir / "report.json").is_file():
        previous_report = _load_json(args.report_dir / "report.json")
    previous_runs = {
        item.get("run_dir"): item
        for item in previous_report.get("runs", [])
        if isinstance(item, dict)
    }
    for arm in arms:
        run_id = f"{suite_id}-{arm.name}"
        run_dir = args.runs_root / run_id
        if run_dir.exists():
            if args.reuse_sealed:
                manifest, validation = _require_sealed(run_dir)
                rows = _round_rows(arm, manifest, run_dir)
                command = _command(args, arm, run_id)
                previous = previous_runs.get(str(run_dir), {})
                peak_memory = previous.get("peak_memory_mib")
                if not isinstance(peak_memory, list) or len(peak_memory) != len(inventory):
                    peak_memory = _recovered_peak_memory(run_dir / "run.log", len(inventory))
                results.append(
                    {
                        "arm": arm,
                        "run_dir": run_dir,
                        "manifest": manifest,
                        "rows": rows,
                        "peak_memory_mib": peak_memory,
                        "elapsed_seconds": None,
                        "inventory": inventory,
                        "command": command,
                        "validation": _validation_summary(validation),
                    }
                )
                print(
                    json.dumps(
                        {"event": "reuse", "arm": arm.name, "run_id": run_id},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            raise FileExistsError(f"refusing existing run directory: {run_dir}")
        command = _command(args, arm, run_id)
        command_log = controller_dir / f"{run_id}.launcher.log"
        validate_log = controller_dir / f"{run_id}.validate.log"
        print(json.dumps({"event": "start", "arm": arm.name, "run_id": run_id}), flush=True)
        returncode, peak_memory, elapsed = _run_process(command, command_log)
        if returncode:
            raise RuntimeError(f"launcher failed for {arm.name}; see {command_log}")
        _run_checked(_validate_command(args, run_dir), validate_log)
        manifest, validation = _require_sealed(run_dir)
        rows = _round_rows(arm, manifest, run_dir)
        results.append(
            {
                "arm": arm,
                "run_dir": run_dir,
                "manifest": manifest,
                "rows": rows,
                "peak_memory_mib": peak_memory,
                "elapsed_seconds": elapsed,
                "inventory": inventory,
                "command": command,
                "validation": _validation_summary(validation),
            }
        )
        print(
            json.dumps(
                {
                    "event": "complete",
                    "arm": arm.name,
                    "run_id": run_id,
                    "elapsed_seconds": elapsed,
                    "peak_memory_mib": peak_memory,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return results


def _analyze(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        _summary(item["rows"], item["peak_memory_mib"], item["validation"]) for item in results
    ]
    by_arm = {item["arm"]: item for item in summaries}
    pairs = (
        ("regular-consistency", "regular-native"),
        ("1p1d-native", "regular-native"),
        ("1p1d-consistency", "regular-consistency"),
        ("1p1d-consistency", "regular-native"),
    )
    comparisons = []
    for candidate, baseline in pairs:
        if candidate not in by_arm or baseline not in by_arm:
            continue
        comparisons.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "step_time_delta_pct": _delta(by_arm[candidate], by_arm[baseline], "step_time_s"),
                "actor_tokens_per_s_delta_pct": _delta(
                    by_arm[candidate], by_arm[baseline], "actor_tokens_per_s"
                ),
            }
        )
    all_rows = [row for item in results for row in item["rows"]]
    flat_summaries = []
    for item in summaries:
        flat_summaries.append(
            {
                "arm": item["arm"],
                "mode": item["mode"],
                "layout": item["layout"],
                "run_id": item["run_id"],
                "rounds": item["rounds"],
                "validation_passed": item["validation_passed"],
                "step_time_mean_s": item["steady_steps_1_to_n"]["step_time_s"]["mean"],
                "step_time_median_s": item["steady_steps_1_to_n"]["step_time_s"]["median"],
                "rollout_tokens_per_gpu_s_mean": item["all_steps"]["rollout_tokens_per_gpu_s"][
                    "mean"
                ],
                "actor_tokens_per_s_mean": item["steady_steps_1_to_n"]["actor_tokens_per_s"][
                    "mean"
                ],
                "peak_memory_mib": json.dumps(item["peak_memory_mib"]),
                "exact_required": item["validation"]["train_rollout_logprob"].get("require_exact"),
                "mismatch_count": item["validation"]["train_rollout_logprob"].get("mismatch_count"),
                "max_abs_diff": item["validation"]["train_rollout_logprob"].get("max_abs_diff"),
            }
        )
    report = {
        "schema_version": "rlkernel.qwen3_8b_h100_1p1d_matrix.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": {"gpu_inventory": results[0]["inventory"] if results else []},
        "conditions": {
            "rounds": args.rounds,
            "seed": args.seed,
            "rollout_seed": args.rollout_seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "kl_coef": args.kl_coef,
            "max_response_len": args.max_response_len,
            "max_tokens_per_gpu": args.max_tokens_per_gpu,
            "global_batch_size": 8,
            "n_samples_per_prompt": 8,
            "training_topology": "TP4/CP2",
            "rollout_topology": "TP4/CP1",
            "regular_vllm_memory_fraction": 0.4,
            "1p1d_vllm_memory_fraction": 0.25,
        },
        "runs": [
            {
                "arm": item["arm"].name,
                "run_dir": str(item["run_dir"]),
                "command": item["command"],
                "elapsed_seconds": item["elapsed_seconds"],
                "peak_memory_mib": item["peak_memory_mib"],
                "rl_kernel_revision": item["manifest"]["revisions"]["rl_kernel"],
                "vime_revision": item["manifest"]["revisions"]["vime"],
                "validation": item["validation"],
            }
            for item in results
        ],
        "summaries": summaries,
        "comparisons": comparisons,
    }
    _json_dump(report_dir / "report.json", report)
    _write_csv(report_dir / "rounds.csv", all_rows)
    _write_csv(report_dir / "summary.csv", flat_summaries)
    _write_markdown(report_dir / "REPORT.md", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rounds != 3:
        raise SystemExit("this acceptance harness is intentionally fixed to --rounds 3")
    if args.max_response_len <= 0 or args.max_tokens_per_gpu <= 0:
        raise SystemExit("response and token budgets must be positive")
    arms = [_parse_arm(value) for value in args.arms]
    results = _run_matrix(args, arms)
    _analyze(args, results)
    print(json.dumps({"status": "passed", "report_dir": str(args.report_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
