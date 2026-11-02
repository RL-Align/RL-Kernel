# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Run the native Megatron/Transformer Engine CP KV-ring comparison.

This runner deliberately delegates model execution to Megatron Bridge. RL-Kernel
does not reimplement TE's KV ring here. The delegated teacher script must set
``transformer_impl=transformer_engine`` and receives ``cp_comm_type`` from this
runner. A non-zero child exit or a missing runtime provenance is a failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-script", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--token-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cp-comm-type", choices=("p2p", "all_gather"), default="p2p")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--cp-sizes", default="1,2")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=5.0e-2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cp_sizes = _parse_cp_sizes(args.cp_sizes)
    _validate_args(args, cp_sizes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for cp_size in cp_sizes:
        result = _run_teacher(args, cp_size=cp_size)
        runs.append(result)

    comparison = None
    errors: list[str] = []
    if all(run.get("status") == "passed" for run in runs):
        try:
            comparison = _compare_runs(runs, atol=args.atol)
        except (RuntimeError, ValueError) as exc:
            comparison = {"pass": False, "identity_error": str(exc), "atol": args.atol}
            errors.append(f"CP token identity validation failed: {exc}")
        else:
            if not comparison["pass"]:
                errors.append(
                    f"CP=1 vs CP=2 native TE drift exceeds atol={args.atol}: "
                    f"max_abs={comparison['max_abs']}"
                )
    else:
        errors.extend(
            f"CP={run['cp_size']} teacher run failed: {run.get('error', 'unknown error')}"
            for run in runs
            if run.get("status") != "passed"
        )

    report = {
        "schema_version": "ws2_megatron_te_cp_compare/v1",
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "transport": (
            "native_te_kv_ring" if args.cp_comm_type == "p2p" else "native_te_kv_all_gather"
        ),
        "requested": {
            "cp_comm_type": args.cp_comm_type,
            "tensor_parallel_size": args.tensor_parallel_size,
            "context_parallel_sizes": cp_sizes,
            "dtype": "bfloat16",
            "seed": args.seed,
        },
        "runs": runs,
        "comparison": comparison,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


def _run_teacher(args: argparse.Namespace, *, cp_size: int) -> dict[str, Any]:
    stem = f"megatron_tp{args.tensor_parallel_size}_cp{cp_size}_{args.cp_comm_type}"
    output = args.output_dir / f"{stem}.json"
    log = args.output_dir / f"{stem}.log"
    command = [
        args.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={args.tensor_parallel_size * cp_size}",
        str(args.teacher_script),
        "--model",
        str(args.model),
        "--token-artifact",
        str(args.token_artifact),
        "--output",
        str(output),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--context-parallel-size",
        str(cp_size),
        "--cp-comm-type",
        args.cp_comm_type,
        "--seed",
        str(args.seed),
    ]
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=args.teacher_script.resolve().parents[2],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    row: dict[str, Any] = {
        "cp_size": cp_size,
        "world_size": args.tensor_parallel_size * cp_size,
        "command": command,
        "output": str(output),
        "log": str(log),
        "returncode": completed.returncode,
    }
    if completed.returncode != 0:
        row.update({"status": "failed", "error": f"returncode={completed.returncode}"})
        return row
    if not output.is_file():
        row.update({"status": "failed", "error": "teacher output JSON is missing"})
        return row
    try:
        artifact = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        row.update({"status": "failed", "error": f"invalid teacher output JSON: {exc}"})
        return row
    if not isinstance(artifact, dict):
        row.update({"status": "failed", "error": "teacher output JSON is not an object"})
        return row
    if artifact.get("schema") != "ws2.megatron_teacher_logprobs.v1":
        row.update({"status": "failed", "error": "teacher output schema is invalid"})
        return row
    actual = artifact.get("actual")
    provider = actual.get("provider") if isinstance(actual, Mapping) else None
    if not isinstance(provider, Mapping):
        provider = {}
    active_token_logprobs = artifact.get("active_token_logprobs")
    required = {
        "context_parallel_size": cp_size,
        "tensor_model_parallel_size": args.tensor_parallel_size,
        "cp_comm_type": args.cp_comm_type,
        "transformer_impl": "transformer_engine",
    }
    mismatches = {
        key: {"expected": expected, "actual": provider.get(key)}
        for key, expected in required.items()
        if provider.get(key) != expected
    }
    row.update(
        {
            "status": "passed" if not mismatches else "failed",
            "actual": actual,
            "mismatches": mismatches,
            "active_token_count": (
                len(active_token_logprobs) if isinstance(active_token_logprobs, list) else 0
            ),
        }
    )
    try:
        _, _, _, token_ids_sha256 = _token_identity(artifact, label=f"CP={cp_size}")
    except (RuntimeError, ValueError) as exc:
        row.update(status="failed", error=str(exc))
    else:
        row["token_ids_sha256"] = token_ids_sha256
    if mismatches:
        row["error"] = "runtime provider provenance does not match native TE request"
    return row


def _compare_runs(runs: list[dict[str, Any]], *, atol: float) -> dict[str, Any]:
    if len(runs) != 2:
        raise ValueError("CP comparison requires exactly two runs")
    if [run.get("cp_size") for run in runs] != [1, 2]:
        raise ValueError("CP comparison requires runs ordered as CP=1 then CP=2")
    artifacts = [json.loads(Path(run["output"]).read_text(encoding="utf-8")) for run in runs]
    left = artifacts[0]["active_token_logprobs"]
    right = artifacts[1]["active_token_logprobs"]
    left_positions, left_token_ids, left_all_token_ids, left_hash = _token_identity(
        artifacts[0], label="left CP run"
    )
    right_positions, right_token_ids, right_all_token_ids, right_hash = _token_identity(
        artifacts[1], label="right CP run"
    )
    if len(left) != len(right):
        raise RuntimeError(
            f"CP runs produced different active-token counts: left={len(left)}, right={len(right)}"
        )
    if left_positions != right_positions:
        raise RuntimeError("CP runs produced different active-token positions")
    if left_all_token_ids != right_all_token_ids:
        raise RuntimeError("CP runs used different complete token ID sequences")
    if left_token_ids != right_token_ids:
        mismatch = next(
            index
            for index, (left_id, right_id) in enumerate(
                zip(left_token_ids, right_token_ids, strict=True)
            )
            if left_id != right_id
        )
        raise RuntimeError(
            "CP runs produced different active-token IDs at index "
            f"{mismatch}: left={left_token_ids[mismatch]}, right={right_token_ids[mismatch]}"
        )
    if left_hash != right_hash:
        raise RuntimeError("CP runs produced different token_ids_sha256 values")
    diffs = [
        abs(float(left_row["logprob"]) - float(right_row["logprob"]))
        for left_row, right_row in zip(left, right, strict=True)
    ]
    worst_index = max(range(len(diffs)), key=diffs.__getitem__)
    return {
        "left_cp_size": runs[0]["cp_size"],
        "right_cp_size": runs[1]["cp_size"],
        "active_token_count": len(diffs),
        "token_ids_sha256": left_hash,
        "max_abs": max(diffs, default=0.0),
        "mean_abs": sum(diffs) / len(diffs) if diffs else 0.0,
        "worst": {
            "position": left[worst_index]["position"] if diffs else None,
            "token_id": left[worst_index]["token_id"] if diffs else None,
            "abs_diff": diffs[worst_index] if diffs else 0.0,
        },
        "atol": atol,
        "pass": max(diffs, default=0.0) <= atol,
    }


def _token_identity(
    artifact: dict[str, Any],
    *,
    label: str,
) -> tuple[list[int], list[int], list[int], str]:
    entries = artifact.get("active_token_logprobs")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"{label} active_token_logprobs is missing or empty")
    positions: list[int] = []
    token_ids: list[int] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError(f"{label} token entry {index} is not an object")
        position = entry.get("position")
        token_id = entry.get("token_id")
        if isinstance(position, bool) or not isinstance(position, int):
            raise RuntimeError(f"{label} token entry {index} has an invalid position")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise RuntimeError(f"{label} token entry {index} has an invalid token_id")
        logprob = entry.get("logprob")
        if (
            isinstance(logprob, bool)
            or not isinstance(logprob, (int, float))
            or not math.isfinite(logprob)
        ):
            raise RuntimeError(f"{label} token entry {index} has a non-finite logprob") from None
        positions.append(position)
        token_ids.append(token_id)
    all_token_ids = artifact.get("token_ids")
    if not (
        isinstance(all_token_ids, list)
        and all_token_ids
        and all(
            isinstance(token_id, int) and not isinstance(token_id, bool)
            for token_id in all_token_ids
        )
    ):
        raise RuntimeError(f"{label} complete token_ids is missing or invalid")
    expected_positions = list(range(1, len(entries) + 1))
    if positions != expected_positions:
        raise RuntimeError(f"{label} active-token positions are not canonical")
    if len(all_token_ids) != len(entries) + 1:
        raise RuntimeError(f"{label} complete token_ids length is inconsistent")
    if any(
        all_token_ids[position] != token_id
        for position, token_id in zip(positions, token_ids, strict=True)
    ):
        raise RuntimeError(f"{label} active token IDs do not match complete token_ids")
    digest = hashlib.sha256(
        json.dumps(all_token_ids, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    declared_digest = artifact.get("token_ids_sha256")
    if declared_digest != digest:
        raise RuntimeError(f"{label} token_ids_sha256 does not match complete token_ids")
    return positions, token_ids, all_token_ids, digest


def _parse_cp_sizes(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("cp-sizes must be a comma-separated list of positive integers") from exc
    if values != (1, 2):
        raise ValueError("cp-sizes must be exactly 1,2 for the CP consistency comparison")
    return values


def _validate_args(args: argparse.Namespace, cp_sizes: tuple[int, ...]) -> None:
    if not args.teacher_script.is_file():
        raise FileNotFoundError(f"teacher script does not exist: {args.teacher_script}")
    if not args.model.exists():
        raise FileNotFoundError(f"model path does not exist: {args.model}")
    if not args.token_artifact.is_file():
        raise FileNotFoundError(f"token artifact does not exist: {args.token_artifact}")
    if args.tensor_parallel_size < 1:
        raise ValueError("parallel sizes must be positive")
    if not math.isfinite(args.atol) or args.atol < 0:
        raise ValueError("atol must be finite and non-negative")


if __name__ == "__main__":
    raise SystemExit(main())
