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
from functools import partial
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.testing import (  # noqa: E402
    compute_policy_ratio,
    compute_reference_kl,
    make_synthetic_rl_kernel_batch,
    selected_logprobs_reference,
    summarize_kernel_drift,
)

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
    "peak_memory_gb",
    "max_error",
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


def _memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def _selected_logprob_row(config: BenchmarkConfig) -> dict[str, Any]:
    candidate_names = {
        "apply": "FusedLogp.apply",
        "out": "FusedLogp.out",
        "fp32": "FusedLogp.apply_fp32",
        "indexed_out": "FusedLogp.indexed_out",
        "indexed_fp32": "FusedLogp.indexed_fp32",
        "online_out": "FusedLogp.online_out",
        "online_fp32": "FusedLogp.online_fp32",
        "online_indexed_out": "FusedLogp.online_indexed_out",
        "online_indexed_fp32": "FusedLogp.online_indexed_fp32",
        "deterministic_fp32": "DeterministicLogp.apply_fp32",
        "deterministic_indexed_fp32": "DeterministicLogp.indexed_fp32",
    }
    candidate_name = candidate_names[config.candidate]

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

    logits = torch.randn(
        batch.batch_size,
        batch.completion_len,
        config.vocab_size,
        device=config.device,
        dtype=config.dtype,
    )

    if config.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(config.device)

    reference, reference_ms = _time_ms(
        lambda: selected_logprobs_reference(
            logits,
            batch.token_ids,
            mask=batch.completion_mask,
            output_dtype=torch.float32,
        ),
        config.device,
        warmup=config.warmup,
        repeat=config.repeat,
    )
    old_logps = reference - 0.01
    ref_logps = reference - 0.02

    notes = ""
    status = "pass"
    candidate_ms: float | str
    max_error: float | str
    ratio_drift: float | str
    kl_drift: float | str

    if config.device.type != "cuda":
        candidate_ms = ""
        max_error = ""
        ratio_drift = ""
        kl_drift = ""
        status = "blocked"
        notes = "candidate requires CUDA"
    else:
        try:
            from rl_engine.kernels.registry import kernel_registry

            dense_op = kernel_registry.get_op("logp")
            indexed_op = kernel_registry.get_op("logp_indexed")
            online_op = kernel_registry.get_op("logp_online")
            online_indexed_op = kernel_registry.get_op("logp_online_indexed")
            deterministic_op = kernel_registry.get_op("logp_deterministic")
            deterministic_indexed_op = kernel_registry.get_op("logp_deterministic_indexed")
            candidate_backend_names = {
                "apply": "FusedLogpGenericOp",
                "out": "FusedLogpGenericOp",
                "fp32": "FusedLogpGenericOp",
                "indexed_out": "FusedLogpGenericOp",
                "indexed_fp32": "FusedLogpGenericOp",
                "online_out": "FusedLogpGenericOp",
                "online_fp32": "FusedLogpGenericOp",
                "online_indexed_out": "FusedLogpGenericOp",
                "online_indexed_fp32": "FusedLogpGenericOp",
                "deterministic_fp32": "DeterministicLogpCUDAOp",
                "deterministic_indexed_fp32": "DeterministicLogpCUDAOp",
            }
            required_backends = {
                "apply": dense_op,
                "out": dense_op,
                "fp32": dense_op,
                "indexed_out": indexed_op,
                "indexed_fp32": indexed_op,
                "online_out": online_op,
                "online_fp32": online_op,
                "online_indexed_out": online_indexed_op,
                "online_indexed_fp32": online_indexed_op,
                "deterministic_fp32": deterministic_op,
                "deterministic_indexed_fp32": deterministic_indexed_op,
            }
            selected_backend = required_backends[config.candidate]
            candidate_backend_name = candidate_backend_names[config.candidate]
            if selected_backend.__class__.__name__ != candidate_backend_name:
                raise RuntimeError(f"{candidate_backend_name} backend is unavailable")

            if config.candidate == "apply":
                run_candidate = partial(dense_op, logits, batch.token_ids)
            elif config.candidate == "out":
                output = torch.empty(
                    batch.token_ids.shape, device=config.device, dtype=config.dtype
                )
                run_candidate = partial(dense_op.out, logits, batch.token_ids, output)
                notes = "preallocated output"
            elif config.candidate == "fp32":
                run_candidate = partial(dense_op.apply_fp32, logits, batch.token_ids)
                notes = "float32 output"
            elif config.candidate == "indexed_out":
                output = torch.zeros(
                    batch.token_ids.shape, device=config.device, dtype=config.dtype
                )
                run_candidate = partial(
                    indexed_op.indexed_out,
                    logits,
                    batch.token_ids,
                    batch.valid_indices,
                    output,
                )
                notes = "valid-index preallocated output"
            elif config.candidate == "indexed_fp32":
                run_candidate = partial(
                    indexed_op.indexed_fp32,
                    logits,
                    batch.token_ids,
                    batch.valid_indices,
                )
                notes = "valid-index float32 output"
            elif config.candidate == "online_out":
                output = torch.empty(
                    batch.token_ids.shape, device=config.device, dtype=config.dtype
                )
                run_candidate = partial(online_op.online_out, logits, batch.token_ids, output)
                notes = "online log-sum-exp preallocated output"
            elif config.candidate == "online_fp32":
                run_candidate = partial(online_op.online_fp32, logits, batch.token_ids)
                notes = "online log-sum-exp float32 output"
            elif config.candidate == "online_indexed_out":
                output = torch.zeros(
                    batch.token_ids.shape, device=config.device, dtype=config.dtype
                )
                run_candidate = partial(
                    online_indexed_op.online_indexed_out,
                    logits,
                    batch.token_ids,
                    batch.valid_indices,
                    output,
                )
                notes = "online log-sum-exp valid-index preallocated output"
            elif config.candidate == "online_indexed_fp32":
                run_candidate = partial(
                    online_indexed_op.online_indexed_fp32,
                    logits,
                    batch.token_ids,
                    batch.valid_indices,
                )
                notes = "online log-sum-exp valid-index float32 output"
            elif config.candidate == "deterministic_fp32":
                run_candidate = partial(deterministic_op.apply_fp32, logits, batch.token_ids)
                notes = "batch-invariant deterministic float32 output"
            elif config.candidate == "deterministic_indexed_fp32":
                run_candidate = partial(
                    deterministic_indexed_op.indexed_fp32,
                    logits,
                    batch.token_ids,
                    batch.valid_indices,
                )
                notes = "batch-invariant deterministic valid-index float32 output"
            else:
                raise ValueError(f"unsupported candidate: {config.candidate}")

            candidate, candidate_ms = _time_ms(
                run_candidate,
                config.device,
                warmup=config.warmup,
                repeat=config.repeat,
            )
            candidate = candidate.float().masked_fill(~batch.completion_mask, 0.0)
            drift = summarize_kernel_drift(candidate, reference, batch.completion_mask)
            max_error = drift["max_abs_error"]
            reference_ratio = compute_policy_ratio(reference, old_logps, batch.completion_mask)
            candidate_ratio = compute_policy_ratio(candidate, old_logps, batch.completion_mask)
            ratio_drift = summarize_kernel_drift(
                candidate_ratio, reference_ratio, batch.completion_mask
            )["max_abs_error"]
            reference_kl = compute_reference_kl(
                reference,
                ref_logps,
                batch.completion_mask,
            )
            candidate_kl = compute_reference_kl(
                candidate,
                ref_logps,
                batch.completion_mask,
            )
            kl_drift = summarize_kernel_drift(
                candidate_kl,
                reference_kl,
                batch.completion_mask,
            )["max_abs_error"]
        except Exception as exc:
            candidate_ms = ""
            max_error = ""
            ratio_drift = ""
            kl_drift = ""
            status = "blocked"
            notes = f"candidate unavailable: {str(exc).splitlines()[0]}"

    peak_memory_gb = _memory_gb(config.device)
    metadata = batch.benchmark_metadata()
    timing_mode = "cuda_event_median_ms" if config.device.type == "cuda" else "wall_median_ms"
    timing_notes = f"warmup={config.warmup}; repeat={config.repeat}; {timing_mode}"
    notes = f"{notes}; {timing_notes}" if notes else timing_notes

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
        "candidate_ms": f"{candidate_ms:.4f}" if isinstance(candidate_ms, float) else candidate_ms,
        "peak_memory_gb": f"{peak_memory_gb:.6f}",
        "max_error": max_error,
        "ratio_drift": ratio_drift,
        "kl_drift": kl_drift,
        "status": status,
        "notes": notes,
    }


def _write_rows(rows: list[dict[str, Any]], output: Path | None) -> None:
    if output is None:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=CSV_COLUMNS)
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
    parser = argparse.ArgumentParser(description="RL-shaped RL-Kernel benchmark runner")
    parser.add_argument("--case", default="selected_logprob", choices=["selected_logprob"])
    parser.add_argument(
        "--candidate",
        default="apply",
        choices=[
            "apply",
            "out",
            "fp32",
            "indexed_out",
            "indexed_fp32",
            "online_out",
            "online_fp32",
            "online_indexed_out",
            "online_indexed_fp32",
            "deterministic_fp32",
            "deterministic_indexed_fp32",
        ],
    )
    parser.add_argument("--smoke", action="store_true", help="Run a small local-development shape")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--g-sizes", default="4", help="Comma-separated samples-per-prompt values")
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--completion-lens", default="16")
    parser.add_argument("--vocab-sizes", default="1024")
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
        prompt_len = 4
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
                        rows.append(_selected_logprob_row(config))
                    except torch.cuda.OutOfMemoryError as exc:
                        rows.append(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "case": args.case,
                                "candidate": args.candidate,
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
                                "peak_memory_gb": "",
                                "max_error": "",
                                "ratio_drift": "",
                                "kl_drift": "",
                                "status": "oom",
                                "notes": str(exc),
                            }
                        )

    _write_rows(rows, args.output)


if __name__ == "__main__":
    main()
