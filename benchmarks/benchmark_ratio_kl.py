# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.ops.pytorch.loss.ratio_kl import NativeRatioKLOp  # noqa: E402
from rl_engine.testing import make_synthetic_rl_kernel_batch  # noqa: E402

CSV_COLUMNS = [
    "timestamp",
    "case",
    "candidate",
    "device",
    "dtype",
    "num_prompts",
    "samples_per_prompt",
    "prompt_len",
    "completion_len",
    "vocab_size",
    "mask_density",
    "valid_tokens",
    "reference_ms",
    "candidate_ms",
    "speedup",
    "reference_mem_gb",
    "candidate_mem_gb",
    "ratio_drift",
    "kl_drift",
    "status",
    "notes",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    case: str
    candidate: str
    device: torch.device
    dtype: torch.dtype
    num_prompts: int
    samples_per_prompt: int
    prompt_len: int
    completion_len: int
    vocab_size: int
    mask_density: float
    seed: int
    warmup: int
    repeat: int


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(fn, device: torch.device, *, warmup: int = 3, repeat: int = 10) -> tuple[Any, float]:
    result = None
    for _ in range(max(0, warmup)):
        result = fn()
    _sync(device)

    elapsed: list[float] = []
    for _ in range(max(1, repeat)):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            end.synchronize()
            elapsed.append(start.elapsed_time(end))
        else:
            _sync(device)
            start_time = time.perf_counter()
            result = fn()
            _sync(device)
            elapsed.append((time.perf_counter() - start_time) * 1000.0)

    _sync(device)
    return result, statistics.median(elapsed)


def _peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _ratio_kl_row(config: BenchmarkConfig) -> dict[str, Any]:
    candidate_name = "TritonRatioKLOp"

    batch = make_synthetic_rl_kernel_batch(
        num_prompts=config.num_prompts,
        samples_per_prompt=config.samples_per_prompt,
        prompt_len=config.prompt_len,
        completion_len=config.completion_len,
        vocab_size=config.vocab_size,
        valid_density=config.mask_density,
        dtype=config.dtype,
        device=config.device,
        seed=config.seed,
    )

    logit_shape = (batch.batch_size, batch.completion_len, config.vocab_size)
    policy_logits = torch.randn(logit_shape, device=config.device, dtype=config.dtype)
    ref_logits = torch.randn(logit_shape, device=config.device, dtype=config.dtype)
    action_ids = batch.token_ids
    mask = batch.completion_mask
    old_logps = batch.old_logps

    native = NativeRatioKLOp()

    def run(op):
        with torch.no_grad():
            return op(policy_logits, ref_logits, action_ids, mask, old_logps)

    _reset_peak(config.device)
    (ref_ratio, ref_kl), reference_ms = _time_ms(
        lambda: run(native), config.device, warmup=config.warmup, repeat=config.repeat
    )
    reference_mem_gb = _peak_memory_gb(config.device)

    notes = ""
    status = "pass"
    candidate_ms: float | str = ""
    speedup: float | str = ""
    candidate_mem_gb: float | str = ""
    ratio_drift: float | str = ""
    kl_drift: float | str = ""

    if config.device.type != "cuda":
        status = "blocked"
        notes = "candidate requires CUDA"
    else:
        try:
            from rl_engine.kernels.registry import kernel_registry

            candidate_op = kernel_registry.get_op("ratio_kl")
            if candidate_op.__class__.__name__ != candidate_name:
                raise RuntimeError(f"{candidate_name} backend is unavailable")

            _reset_peak(config.device)
            (cand_ratio, cand_kl), candidate_ms = _time_ms(
                lambda: run(candidate_op),
                config.device,
                warmup=config.warmup,
                repeat=config.repeat,
            )
            candidate_mem_gb = _peak_memory_gb(config.device)
            speedup = reference_ms / candidate_ms if candidate_ms else float("inf")
            ratio_drift = (cand_ratio.float() - ref_ratio.float()).abs().max().item()
            kl_drift = (cand_kl.float() - ref_kl.float()).abs().max().item()
        except Exception as exc:
            status = "blocked"
            notes = f"candidate unavailable: {str(exc).splitlines()[0]}"

    metadata = batch.benchmark_metadata()
    timing_mode = "cuda_event_median_ms" if config.device.type == "cuda" else "wall_median_ms"
    timing_notes = f"warmup={config.warmup}; repeat={config.repeat}; {timing_mode}; forward-only"
    notes = f"{notes}; {timing_notes}" if notes else timing_notes

    def _fmt(value: Any, spec: str) -> Any:
        return format(value, spec) if isinstance(value, float) else value

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case": config.case,
        "candidate": candidate_name,
        "device": str(config.device),
        "dtype": str(config.dtype),
        "num_prompts": config.num_prompts,
        "samples_per_prompt": config.samples_per_prompt,
        "prompt_len": config.prompt_len,
        "completion_len": config.completion_len,
        "vocab_size": config.vocab_size,
        "mask_density": config.mask_density,
        "valid_tokens": metadata["valid_tokens"],
        "reference_ms": f"{reference_ms:.4f}",
        "candidate_ms": _fmt(candidate_ms, ".4f"),
        "speedup": _fmt(speedup, ".2f"),
        "reference_mem_gb": f"{reference_mem_gb:.6f}",
        "candidate_mem_gb": _fmt(candidate_mem_gb, ".6f"),
        "ratio_drift": _fmt(ratio_drift, ".3e"),
        "kl_drift": _fmt(kl_drift, ".3e"),
        "status": status,
        "notes": notes,
    }


def _write_rows(rows: list[dict[str, Any]], output: Path | None) -> None:
    if output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists() and output.stat().st_size > 0
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fused ratio/KL RL-Kernel benchmark runner")
    parser.add_argument("--case", default="ratio_kl", choices=["ratio_kl"])
    parser.add_argument("--candidate", default="triton", choices=["triton"])
    parser.add_argument("--smoke", action="store_true", help="Run a small local-development shape")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--g-sizes", default="4", help="Comma-separated samples-per-prompt values")
    parser.add_argument("--prompt-len", type=int, default=0)
    parser.add_argument("--completion-lens", default="512")
    parser.add_argument("--vocab-sizes", default="32768,131072")
    parser.add_argument("--mask-densities", default="1.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)

    if args.smoke:
        num_prompts = 1
        g_sizes = [2]
        prompt_len = 0
        completion_lens = [8]
        vocab_sizes = [128]
        mask_densities = [0.5, 1.0]
    else:
        num_prompts = args.num_prompts
        g_sizes = _parse_int_list(args.g_sizes)
        prompt_len = args.prompt_len
        completion_lens = _parse_int_list(args.completion_lens)
        vocab_sizes = _parse_int_list(args.vocab_sizes)
        mask_densities = _parse_float_list(args.mask_densities)

    rows: list[dict[str, Any]] = []
    for samples_per_prompt in g_sizes:
        for completion_len in completion_lens:
            for vocab_size in vocab_sizes:
                for mask_density in mask_densities:
                    config = BenchmarkConfig(
                        case=args.case,
                        candidate=args.candidate,
                        device=device,
                        dtype=dtype,
                        num_prompts=num_prompts,
                        samples_per_prompt=samples_per_prompt,
                        prompt_len=prompt_len,
                        completion_len=completion_len,
                        vocab_size=vocab_size,
                        mask_density=mask_density,
                        seed=args.seed,
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )
                    try:
                        rows.append(_ratio_kl_row(config))
                    except torch.cuda.OutOfMemoryError as exc:
                        rows.append(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "case": args.case,
                                "candidate": "TritonRatioKLOp",
                                "device": str(device),
                                "dtype": str(dtype),
                                "num_prompts": num_prompts,
                                "samples_per_prompt": samples_per_prompt,
                                "prompt_len": prompt_len,
                                "completion_len": completion_len,
                                "vocab_size": vocab_size,
                                "mask_density": mask_density,
                                "valid_tokens": "",
                                "reference_ms": "",
                                "candidate_ms": "",
                                "speedup": "",
                                "reference_mem_gb": "",
                                "candidate_mem_gb": "",
                                "ratio_drift": "",
                                "kl_drift": "",
                                "status": "oom",
                                "notes": str(exc),
                            }
                        )

    _write_rows(rows, args.output)


if __name__ == "__main__":
    main()
