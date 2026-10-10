#!/usr/bin/env python3
"""Upload the complete ROCm G10/G11 RL result set to Weights & Biases."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import wandb

RECORD_RE = re.compile(r"(?:perf|step|rollout)\s+(\d+):\s+(\{.*\})")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g10-log", type=Path, required=True)
    parser.add_argument("--g11-log", type=Path, required=True)
    parser.add_argument("--g10-validation", type=Path, required=True)
    parser.add_argument("--g11-validation", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--entity", default="RL-Kernel")
    parser.add_argument("--project", default="rocm-qwen3-8b-tp4-cp2-200")
    parser.add_argument("--url-output", type=Path)
    return parser.parse_args()


def read_log(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for raw_line in path.open(encoding="utf-8", errors="replace"):
        match = RECORD_RE.search(ANSI_RE.sub("", raw_line))
        if not match:
            continue
        try:
            payload = ast.literal_eval(match.group(2))
        except (SyntaxError, ValueError):
            continue
        if isinstance(payload, dict):
            rows.setdefault(int(match.group(1)), {}).update(payload)
    if sorted(rows) != list(range(200)):
        raise RuntimeError(f"{path} has {len(rows)} steps; expected 0..199")
    return rows


def scalar_metrics(payload: dict[str, Any]) -> dict[str, int | float]:
    return {
        key: value
        for key, value in payload.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def run_config(case_id: str) -> dict[str, Any]:
    return {
        "model": "Qwen3-8B",
        "hardware": "1 node, 8x AMD Instinct MI300X 192GB",
        "case_id": case_id,
        "actor_parallelism": "TP4/CP2/PP1",
        "rollout_engines": 2,
        "rollout_tensor_parallel_size": 4,
        "steps": 200,
        "rollout_batch_size": 1,
        "samples_per_prompt": 8,
        "global_batch_size": 8,
        "max_response_length": 7168,
        "max_tokens_per_gpu": 4096,
        "seed": 1234,
        "rollout_seed": 1234,
        "vllm_memory_utilization": 0.38,
        "use_rollout_logprobs": True,
        "reference_model": True,
        "kl_loss_coefficient": 0.001,
        "torch_profiler": False,
        "rl_kernel_revision": "7a9f3b5657ac380ae2419539975498513d18c490",
        "vime_revision": "c80200e7aef08edc918e50a3998ea981bd689934",
        "megatron_revision": "1dcf0dafa884ad52ffb243625717a3471643e087",
        "result_base_revision": "0547fb60634c49a73d55facaab2277b2c2220d17",
        "source_pr": 396,
        "frozen_inputs_equal": True,
        "frozen_sources_equal": True,
    }


def upload_run(
    *,
    args: argparse.Namespace,
    case_id: str,
    name: str,
    log_path: Path,
    validation_path: Path,
) -> str:
    rows = read_log(log_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="pr396-rocm-g10-g11-refkl001-s1234",
        job_type="rl-training",
        name=name,
        config=run_config(case_id),
        tags=["rocm", "mi300x", "qwen3-8b", "grpo", case_id.lower().replace("/", "-")],
        reinit="finish_previous",
    )
    assert run is not None
    for index in range(200):
        run.log({"log_step_index": index, **scalar_metrics(rows[index])}, step=index)

    run.summary.update(
        {
            "completed_steps": 200,
            "validation_passed": bool(validation.get("passed")),
            "frozen_sources_match": bool(validation.get("frozen_sources_match")),
            "launcher_returncode": int(validation.get("launcher_returncode", -1)),
        }
    )
    metrics = validation.get("metrics", {})
    for key in (
        "sample_count",
        "element_count",
        "mismatch_count",
        "max_abs_diff",
        "train_rollout_logprob_abs_diff",
        "torch_equal",
    ):
        if key in metrics:
            run.summary[f"validation/{key}"] = metrics[key]

    artifact = wandb.Artifact(name=f"{name}-raw", type="rl-run-data")
    artifact.add_file(str(log_path), name=f"{case_id.lower().replace('/', '-')}/launcher.log")
    artifact.add_file(
        str(validation_path), name=f"{case_id.lower().replace('/', '-')}/validation.json"
    )
    run.log_artifact(artifact)
    url = run.url
    run.finish()
    return url


def main() -> None:
    args = parse_args()
    urls = {
        "g10": upload_run(
            args=args,
            case_id="P/P",
            name="pr396-rocm-g10-pp-refkl001-s1234",
            log_path=args.g10_log,
            validation_path=args.g10_validation,
        ),
        "g11": upload_run(
            args=args,
            case_id="R/R",
            name="pr396-rocm-g11-rr-refkl001-s1234",
            log_path=args.g11_log,
            validation_path=args.g11_validation,
        ),
    }
    report = wandb.init(
        entity=args.entity,
        project=args.project,
        group="pr396-rocm-g10-g11-refkl001-s1234",
        job_type="report",
        name="pr396-rocm-g10-g11-200-step-report",
        config={"g10_run": urls["g10"], "g11_run": urls["g11"]},
        reinit="finish_previous",
    )
    assert report is not None
    bundle = wandb.Artifact(name="pr396-rocm-g10-g11-200-step-results", type="evaluation-report")
    bundle.add_dir(str(args.bundle_dir))
    report.log_artifact(bundle)
    for image_path in sorted(args.bundle_dir.glob("*.png")):
        report.log({image_path.stem: wandb.Image(str(image_path))})
    urls["report"] = report.url
    report.finish()
    output = json.dumps(urls, indent=2, sort_keys=True) + "\n"
    if args.url_output:
        args.url_output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
