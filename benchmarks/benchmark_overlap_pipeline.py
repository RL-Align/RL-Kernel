# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.executors.overlap_pipeline import (  # noqa: E402
    IterationSpec,
    ManifestWeightHandoff,
    OverlapPipeline,
    PipelineConfig,
    RolloutExecutorWorker,
    RolloutStageResult,
    TorchRLTrainingConfig,
    TorchRLTrainingWorker,
    TrainingStageResult,
    timeline_summary_to_dict,
)
from rl_engine.executors.rollout import RolloutExecutor  # noqa: E402

ISSUE_DIR = REPO_ROOT / "task-workspace" / "issues" / "issue_18"
CSV_COLUMNS = [
    "timestamp",
    "candidate",
    "mode",
    "stage",
    "num_iterations",
    "max_prefetch",
    "rollout_avg_ms",
    "training_avg_ms",
    "sequential_estimate_ms",
    "overlapped_elapsed_ms",
    "overlap_ms",
    "overlap_ratio",
    "max_queue_depth",
    "final_weight_version",
    "training_data_source",
    "environment",
    "weight_versions",
    "status",
    "notes",
]


class SyntheticRolloutWorker:
    def __init__(self, delay_seconds: float):
        self.delay_seconds = delay_seconds

    def rollout(self, spec: IterationSpec) -> RolloutStageResult:
        started_at = time.perf_counter()
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        finished_at = time.perf_counter()
        return RolloutStageResult(
            iteration=spec.iteration,
            weight_version=int(spec.weight_version),
            payload={"prompts": list(spec.prompts), "mode": "synthetic"},
            started_at=started_at,
            finished_at=finished_at,
            metrics={"backend": "synthetic", "delay_seconds": self.delay_seconds},
        )


class SyntheticTrainingWorker:
    def __init__(self, delay_seconds: float):
        self.delay_seconds = delay_seconds

    def train(self, rollout: RolloutStageResult) -> TrainingStageResult:
        started_at = time.perf_counter()
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        finished_at = time.perf_counter()
        return TrainingStageResult(
            iteration=rollout.iteration,
            consumed_weight_version=rollout.weight_version,
            published_weight_version=rollout.weight_version + 1,
            metrics={
                "backend": "synthetic",
                "delay_seconds": self.delay_seconds,
                "training_data_source": "synthetic_timing_worker",
            },
            started_at=started_at,
            finished_at=finished_at,
        )


class ManifestInstallSyntheticRolloutWorker(SyntheticRolloutWorker):
    def __init__(self, delay_seconds: float):
        super().__init__(delay_seconds)
        self.installed_versions: list[int] = []
        self.installed_transports: list[str] = []
        self.released_update_ids: list[str] = []

    def rollout(self, spec: IterationSpec) -> RolloutStageResult:
        started_at = time.perf_counter()
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        token_base = int(spec.iteration) + 1
        payload = {
            "normalized_outputs": [
                [
                    {"token_ids": [token_base, token_base + 1]},
                    {"token_ids": [token_base + 2, token_base + 3]},
                ]
            ],
            "mode": "manifest-handoff",
        }
        finished_at = time.perf_counter()
        return RolloutStageResult(
            iteration=spec.iteration,
            weight_version=int(spec.weight_version),
            payload=payload,
            started_at=started_at,
            finished_at=finished_at,
            metrics={
                "backend": "synthetic-manifest-rollout",
                "delay_seconds": self.delay_seconds,
                "installed_weight_version": (
                    self.installed_versions[-1] if self.installed_versions else None
                ),
            },
        )

    def install_weight_manifest(self, manifest) -> None:
        self.installed_versions.append(manifest.weight_version)
        self.installed_transports.append(manifest.transport)

    def release_weight_manifest(self, update_id: str) -> None:
        self.released_update_ids.append(update_id)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv_row(row: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists() and output.stat().st_size > 0
    if exists:
        first_line = output.read_text(encoding="utf-8").splitlines()[0]
        if first_line.split(",") != CSV_COLUMNS:
            exists = False
    with output.open("a" if exists else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_timeline(summary: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _row_from_summary(
    *,
    mode: str,
    status: str,
    notes: str,
    summary: Mapping[str, Any],
    num_iterations: int,
    max_prefetch: int,
) -> dict[str, Any]:
    rollouts = summary["rollout_results"]
    trainings = summary["training_results"]
    rollout_avg = (
        sum(item["duration_seconds"] for item in rollouts) / len(rollouts) * 1000.0
        if rollouts
        else 0.0
    )
    training_avg = (
        sum(item["duration_seconds"] for item in trainings) / len(trainings) * 1000.0
        if trainings
        else 0.0
    )
    weight_versions = [
        f"{item['consumed_weight_version']}->{item['published_weight_version']}"
        for item in trainings
    ]
    data_sources = sorted(
        {str(item.get("metrics", {}).get("training_data_source", "unknown")) for item in trainings}
    )
    return {
        "timestamp": _timestamp(),
        "candidate": "issue-18-candidate-1",
        "mode": mode,
        "stage": "evaluation",
        "num_iterations": num_iterations,
        "max_prefetch": max_prefetch,
        "rollout_avg_ms": f"{rollout_avg:.4f}",
        "training_avg_ms": f"{training_avg:.4f}",
        "sequential_estimate_ms": f"{summary['sequential_estimate_seconds'] * 1000.0:.4f}",
        "overlapped_elapsed_ms": f"{summary['elapsed_seconds'] * 1000.0:.4f}",
        "overlap_ms": f"{summary['overlap_seconds'] * 1000.0:.4f}",
        "overlap_ratio": f"{summary['overlap_ratio']:.6f}",
        "max_queue_depth": summary["max_queue_depth"],
        "final_weight_version": summary.get("final_published_weight_version", ""),
        "training_data_source": ";".join(data_sources),
        "environment": _environment_summary(),
        "weight_versions": ";".join(weight_versions),
        "status": status,
        "notes": notes,
    }


def _blocked_row(
    *,
    mode: str,
    notes: str,
    num_iterations: int,
    max_prefetch: int,
) -> dict[str, Any]:
    return {
        "timestamp": _timestamp(),
        "candidate": "issue-18-candidate-1",
        "mode": mode,
        "stage": "evaluation",
        "num_iterations": num_iterations,
        "max_prefetch": max_prefetch,
        "rollout_avg_ms": "",
        "training_avg_ms": "",
        "sequential_estimate_ms": "",
        "overlapped_elapsed_ms": "",
        "overlap_ms": "",
        "overlap_ratio": "",
        "max_queue_depth": "",
        "final_weight_version": "",
        "training_data_source": "",
        "environment": _environment_summary(),
        "weight_versions": "",
        "status": "blocked",
        "notes": notes,
    }


def _run_pipeline(
    *,
    rollout_worker,
    training_worker,
    num_iterations: int,
    max_prefetch: int,
    prompt_prefix: str,
    token_prompts: bool = False,
    weight_handoff: Optional[ManifestWeightHandoff] = None,
) -> tuple[list[TrainingStageResult], dict[str, Any]]:
    def prompts_for(index: int) -> list[Any]:
        if token_prompts:
            base = [4, 5, 6, 7, 8, 9, 2, index + 10]
            return [
                {"prompt_token_ids": list(base)},
                {"prompt_token_ids": list(base) + [12]},
            ]
        return [f"{prompt_prefix}-{index}-a", f"{prompt_prefix}-{index}-b"]

    specs = [
        IterationSpec(
            iteration=index,
            weight_version=None,
            prompts=prompts_for(index),
        )
        for index in range(num_iterations)
    ]
    pipeline = OverlapPipeline(
        rollout_worker,
        training_worker,
        PipelineConfig(max_prefetch=max_prefetch),
        weight_handoff=weight_handoff,
    )
    results = pipeline.run(specs)
    summary = timeline_summary_to_dict(pipeline.timeline_summary())
    return results, summary


def run_synthetic(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    _, summary = _run_pipeline(
        rollout_worker=SyntheticRolloutWorker(args.synthetic_rollout_delay_ms / 1000.0),
        training_worker=SyntheticTrainingWorker(args.synthetic_training_delay_ms / 1000.0),
        num_iterations=args.num_iterations,
        max_prefetch=args.max_prefetch,
        prompt_prefix="synthetic",
    )
    row = _row_from_summary(
        mode="synthetic",
        status="pass",
        notes="deterministic synthetic fallback; validates scheduler timing only",
        summary=summary,
        num_iterations=args.num_iterations,
        max_prefetch=args.max_prefetch,
    )
    return row, summary


def run_manifest_handoff(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    rollout_worker = ManifestInstallSyntheticRolloutWorker(args.synthetic_rollout_delay_ms / 1000.0)
    training_device = args.training_device
    if training_device == "cuda" and not torch.cuda.is_available():
        training_device = "cpu"
    training_worker = TorchRLTrainingWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=args.num_generations,
            prompt_len=2,
            completion_len=min(args.max_tokens, 4),
            vocab_size=64,
            hidden_dim=32,
            valid_density=1.0,
            device=training_device,
            seed=args.seed,
        )
    )
    _, summary = _run_pipeline(
        rollout_worker=rollout_worker,
        training_worker=training_worker,
        num_iterations=args.num_iterations,
        max_prefetch=args.max_prefetch,
        prompt_prefix="manifest-handoff",
        weight_handoff=ManifestWeightHandoff(),
    )
    row = _row_from_summary(
        mode="manifest-handoff",
        status="pass",
        notes=(
            "local production-semantics manifest handoff; "
            f"installed_versions={rollout_worker.installed_versions}; "
            f"transports={rollout_worker.installed_transports}; "
            f"released={len(rollout_worker.released_update_ids)}"
        ),
        summary=summary,
        num_iterations=args.num_iterations,
        max_prefetch=args.max_prefetch,
    )
    return row, summary


def run_realistic(args: argparse.Namespace) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    training_device = args.training_device
    if training_device == "cuda" and not torch.cuda.is_available():
        training_device = "cpu"

    training_worker = TorchRLTrainingWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=args.num_generations,
            prompt_len=4,
            completion_len=args.max_tokens,
            vocab_size=64,
            hidden_dim=32,
            valid_density=0.75,
            device=training_device,
            seed=args.seed,
        )
    )
    rollout_worker = _build_realistic_rollout_worker(args)
    warmup_seconds = _warm_realistic_rollout_worker(rollout_worker, args)
    _, summary = _run_pipeline(
        rollout_worker=rollout_worker,
        training_worker=training_worker,
        num_iterations=args.num_iterations,
        max_prefetch=args.max_prefetch,
        prompt_prefix="realistic",
        token_prompts=args.skip_tokenizer_init,
    )
    notes = (
        "production-facing adapters: RolloutExecutor.generate_candidates plus "
        f"TorchRLTrainingWorker(device={training_device}); "
        f"token_prompts={args.skip_tokenizer_init}; "
        f"warmup_rollouts={args.realistic_warmup_rollouts}; "
        f"warmup_seconds={warmup_seconds:.4f}"
    )
    row = _row_from_summary(
        mode="realistic",
        status="pass",
        notes=notes,
        summary=summary,
        num_iterations=args.num_iterations,
        max_prefetch=args.max_prefetch,
    )
    return row, summary


def _build_realistic_rollout_worker(args: argparse.Namespace) -> RolloutExecutorWorker:
    if not args.model:
        raise RuntimeError(
            "realistic rollout requires --model for vLLM; pass a local model path/name "
            "or record this as the environment blocker"
        )
    config = {
        "model": args.model,
        "num_generations": args.num_generations,
        "sampling_params": {"max_tokens": args.max_tokens},
        "engine_kwargs": {
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "trust_remote_code": False,
            "enforce_eager": args.enforce_eager,
        },
    }
    if args.skip_tokenizer_init:
        config["engine_kwargs"]["skip_tokenizer_init"] = True
    executor = RolloutExecutor(config)
    return RolloutExecutorWorker(
        executor,
        num_generations=args.num_generations,
        sampling_params={"max_tokens": args.max_tokens},
    )


def _warm_realistic_rollout_worker(
    worker: RolloutExecutorWorker,
    args: argparse.Namespace,
) -> float:
    if args.realistic_warmup_rollouts <= 0:
        return 0.0
    start = time.perf_counter()
    for index in range(args.realistic_warmup_rollouts):
        prompts = (
            [{"prompt_token_ids": [4, 5, 6, 7, 8, 9, 2, 10 + index]}]
            if args.skip_tokenizer_init
            else [f"warmup-{index}"]
        )
        worker.rollout(IterationSpec(iteration=-1 - index, weight_version=0, prompts=prompts))
    return time.perf_counter() - start


def _environment_summary() -> str:
    cuda = torch.version.cuda or ""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"torch={torch.__version__};cuda={cuda};gpu={device};gpu_mem_gb={memory_gb:.2f}"
    return f"torch={torch.__version__};cuda={cuda};gpu=none"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlapping rollout/training pipeline benchmark")
    parser.add_argument(
        "--mode",
        choices=["realistic", "synthetic", "manifest-handoff"],
        default="synthetic",
    )
    parser.add_argument("--smoke", action="store_true", help="Use small issue #18 smoke settings")
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--max-prefetch", type=int, default=1)
    parser.add_argument("--model", default=None, help="vLLM model path/name for realistic mode")
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--training-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-tokenizer-init", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--realistic-warmup-rollouts", type=int, default=1)
    parser.add_argument("--synthetic-rollout-delay-ms", type=float, default=30.0)
    parser.add_argument("--synthetic-training-delay-ms", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=ISSUE_DIR / "benchmark.csv")
    parser.add_argument("--timeline-output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.smoke:
        args.num_iterations = min(args.num_iterations, 3)
        args.max_prefetch = max(1, args.max_prefetch)
        args.num_generations = min(args.num_generations, 2)
        args.max_tokens = min(args.max_tokens, 8)
        args.max_model_len = min(args.max_model_len, 64)

    timeline_output = args.timeline_output or (
        ISSUE_DIR / "outputs" / f"overlap-{args.mode}-{int(time.time())}.json"
    )

    summary: Optional[dict[str, Any]] = None
    try:
        if args.mode == "synthetic":
            row, summary = run_synthetic(args)
        elif args.mode == "manifest-handoff":
            row, summary = run_manifest_handoff(args)
        else:
            row, summary = run_realistic(args)
    except Exception as exc:
        row = _blocked_row(
            mode=args.mode,
            notes=f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            num_iterations=args.num_iterations,
            max_prefetch=args.max_prefetch,
        )

    _write_csv_row(row, args.output)
    if summary is not None:
        _write_timeline(summary, timeline_output)

    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerow(row)


if __name__ == "__main__":
    main()
