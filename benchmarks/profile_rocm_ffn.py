# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Collect torch.profiler data for the strict ROCm Triton Qwen3 FFN.

This script profiles the public ``qwen3_ffn`` API without changing or wrapping
its internal arithmetic. Profiler timings are intended for attribution only;
uninstrumented GPU-event samples are saved separately for latency comparisons.

Example:

    python benchmarks/profile_rocm_ffn.py \
      --direction both \
      --tokens 32 \
      --warmup 3 \
      --active-steps 5 \
      --latency-samples 20 \
      --output-dir /tmp/rlk_torch_profiler_m32
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from rl_engine.kernels.ops.triton.ffn import (
    Qwen3FFNForwardWeights,
    pack_qwen3_ffn_forward_weights,
    qwen3_ffn,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_INPUT_SEEDS = {1: 3010, 8: 3012, 32: 3014}
_KEY_AVERAGE_FIELDS = (
    "name",
    "count",
    "device_type",
    "self_cpu_time_us",
    "cpu_time_us",
    "self_device_time_us",
    "device_time_us",
    "self_cpu_memory_bytes",
    "cpu_memory_bytes",
    "self_device_memory_bytes",
    "device_memory_bytes",
    "input_shapes",
    "stack",
)


@dataclass
class FFNCase:
    direction: str
    hidden: torch.Tensor
    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    down_weight: torch.Tensor
    grad_output: torch.Tensor
    forward_weights: Qwen3FFNForwardWeights | None

    @property
    def slug(self) -> str:
        return self.direction.replace("-", "_")

    @property
    def training(self) -> bool:
        return self.direction == "forward-backward"

    def clear_gradients(self) -> None:
        for tensor in (
            self.hidden,
            self.gate_weight,
            self.up_weight,
            self.down_weight,
        ):
            tensor.grad = None

    def run(self, *, use_forward_weights: bool = True) -> torch.Tensor:
        forward_weights = self.forward_weights if use_forward_weights else None
        if not self.training:
            with torch.no_grad():
                return qwen3_ffn(
                    self.hidden,
                    self.gate_weight,
                    self.up_weight,
                    self.down_weight,
                    forward_weights=forward_weights,
                )

        self.clear_gradients()
        output = qwen3_ffn(
            self.hidden,
            self.gate_weight,
            self.up_weight,
            self.down_weight,
            forward_weights=forward_weights,
        )
        output.backward(self.grad_output)
        return output

    def result_tensors(self, output: torch.Tensor) -> dict[str, torch.Tensor]:
        result = {"output": output}
        if not self.training:
            return result

        for name, tensor in (
            ("dhidden", self.hidden),
            ("dgate_weight", self.gate_weight),
            ("dup_weight", self.up_weight),
            ("ddown_weight", self.down_weight),
        ):
            if tensor.grad is None:
                raise RuntimeError(f"{name} was not produced by backward")
            result[name] = tensor.grad
        return result


def _randn(
    shape: tuple[int, ...],
    *,
    seed: int,
    device: torch.device,
    requires_grad: bool,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.02
    return value.to(device=device, dtype=torch.bfloat16).requires_grad_(requires_grad)


def _build_case(args: argparse.Namespace, direction: str) -> FFNCase:
    device = torch.device("cuda", args.device)
    training = direction == "forward-backward"
    input_seed = (
        args.input_seed if args.input_seed is not None else _INPUT_SEEDS.get(args.tokens, 3010)
    )
    hidden = _randn(
        (args.tokens, args.hidden_size),
        seed=input_seed,
        device=device,
        requires_grad=training,
    )
    gate_weight = _randn(
        (args.intermediate_size, args.hidden_size),
        seed=args.weight_seed,
        device=device,
        requires_grad=training,
    )
    up_weight = _randn(
        (args.intermediate_size, args.hidden_size),
        seed=args.weight_seed + 1,
        device=device,
        requires_grad=training,
    )
    down_weight = _randn(
        (args.hidden_size, args.intermediate_size),
        seed=args.weight_seed + 2,
        device=device,
        requires_grad=training,
    )
    forward_weights = (
        pack_qwen3_ffn_forward_weights(gate_weight, up_weight, down_weight)
        if args.weight_layout == "packed"
        else None
    )
    return FFNCase(
        direction=direction,
        hidden=hidden,
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
        grad_output=_randn(
            (args.tokens, args.hidden_size),
            seed=input_seed + 1,
            device=device,
            requires_grad=False,
        ),
        forward_weights=forward_weights,
    )


def _weight_layout_metadata(case: FFNCase) -> dict[str, Any]:
    metadata: dict[str, Any] = {"additional_forward_weight_bytes": 0}
    for name, weight in (
        ("gate_weight", case.gate_weight),
        ("up_weight", case.up_weight),
        ("down_weight", case.down_weight),
    ):
        metadata[name] = {
            "shape": list(weight.shape),
            "stride": list(weight.stride()),
            "is_contiguous": weight.is_contiguous(),
        }
    if case.forward_weights is not None:
        packed_weights = (
            ("gate_weight_t", case.forward_weights.gate_weight_t),
            ("up_weight_t", case.forward_weights.up_weight_t),
            ("down_weight_t", case.forward_weights.down_weight_t),
        )
        packed_tensors = {
            name: {
                "shape": list(weight.shape),
                "stride": list(weight.stride()),
                "is_contiguous": weight.is_contiguous(),
                "requires_grad": weight.requires_grad,
                "nbytes": weight.numel() * weight.element_size(),
            }
            for name, weight in packed_weights
        }
        additional_bytes = sum(
            weight.numel() * weight.element_size() for _, weight in packed_weights
        )
        metadata["additional_forward_weight_bytes"] = additional_bytes
        metadata["packed_forward_weights"] = {
            "additional_bytes": additional_bytes,
            "tensors": packed_tensors,
        }
    return metadata


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(samples_ms: list[float]) -> dict[str, Any]:
    if not samples_ms:
        return {"samples_ms": []}
    return {
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "p95_ms": _percentile(samples_ms, 0.95),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _gpu_event_samples(case: FFNCase, samples: int) -> list[float]:
    if samples == 0:
        return []
    events = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(samples)
    ]
    for start, end in events:
        start.record()
        output = case.run()
        end.record()
        del output
    torch.cuda.synchronize()
    case.clear_gradients()
    return [float(start.elapsed_time(end)) for start, end in events]


def _tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().contiguous()
    raw = detached.view(torch.uint8).cpu().numpy()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "nbytes": detached.numel() * detached.element_size(),
        "sha256_raw_bytes": hashlib.sha256(memoryview(raw)).hexdigest(),
    }


def _run_and_fingerprint(
    case: FFNCase,
    *,
    use_forward_weights: bool = True,
) -> dict[str, Any]:
    output = case.run(use_forward_weights=use_forward_weights)
    torch.cuda.synchronize()
    fingerprints = {
        name: _tensor_fingerprint(tensor) for name, tensor in case.result_tensors(output).items()
    }
    del output
    case.clear_gradients()
    return fingerprints


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _key_average_rows(profiler: profile, *, group_by_input_shape: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in profiler.key_averages(group_by_input_shape=group_by_input_shape):
        rows.append(
            {
                "name": event.key,
                "count": event.count,
                "device_type": str(event.device_type),
                "self_cpu_time_us": event.self_cpu_time_total,
                "cpu_time_us": event.cpu_time_total,
                "self_device_time_us": event.self_device_time_total,
                "device_time_us": event.device_time_total,
                "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                "cpu_memory_bytes": event.cpu_memory_usage,
                "self_device_memory_bytes": event.self_device_memory_usage,
                "device_memory_bytes": event.device_memory_usage,
                "input_shapes": _json_safe(event.input_shapes),
                "stack": _json_safe(event.stack),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            float(row["self_device_time_us"]),
            float(row["self_cpu_time_us"]),
        ),
        reverse=True,
    )


def _write_key_averages(output_dir: Path, slug: str, rows: list[dict[str, Any]]) -> None:
    _write_json(output_dir / f"{slug}.key_averages.json", rows)
    with (output_dir / f"{slug}.key_averages.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_KEY_AVERAGE_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["input_shapes"] = json.dumps(row["input_shapes"])
            csv_row["stack"] = json.dumps(row["stack"])
            writer.writerow(csv_row)


def _kernel_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    device_rows = [
        row
        for row in rows
        if str(row["device_type"]).endswith(("CUDA", "HIP"))
        # Kineto also emits a synthetic device-side aggregate for a user
        # annotation. It overlaps all child kernels and must not be summed.
        and not str(row["name"]).startswith("rl_kernel::")
    ]
    categories: dict[str, dict[str, float | int]] = {
        "layout_copy": {"count": 0, "self_device_time_us": 0.0},
        "gemm_leaf": {"count": 0, "self_device_time_us": 0.0},
        "gemm_reduce": {"count": 0, "self_device_time_us": 0.0},
        "gemm_root_copy": {"count": 0, "self_device_time_us": 0.0},
        "swiglu": {"count": 0, "self_device_time_us": 0.0},
        "elementwise_add": {"count": 0, "self_device_time_us": 0.0},
        "other_device_kernels": {"count": 0, "self_device_time_us": 0.0},
    }
    for row in device_rows:
        name = str(row["name"]).lower()
        if "direct_copy_kernel" in name:
            category = "layout_copy"
        elif "det_gemm_tree_leaf" in name:
            category = "gemm_leaf"
        elif "det_gemm_tree_reduce" in name:
            category = "gemm_reduce"
        elif "copy_tree_root" in name:
            category = "gemm_root_copy"
        elif "swiglu" in name:
            category = "swiglu"
        elif "functor_add" in name:
            category = "elementwise_add"
        else:
            category = "other_device_kernels"
        categories[category]["count"] += int(row["count"])
        categories[category]["self_device_time_us"] += float(row["self_device_time_us"])

    total_device_us = sum(
        float(category["self_device_time_us"]) for category in categories.values()
    )
    for category in categories.values():
        category["percent_of_device_kernel_time"] = (
            100.0 * float(category["self_device_time_us"]) / total_device_us
            if total_device_us
            else 0.0
        )
    return {
        "accounting": (
            "sum of device kernel events; synthetic rl_kernel:: annotation "
            "device aggregates are excluded to avoid double counting"
        ),
        "total_device_kernel_time_us": total_device_us,
        "categories": categories,
        "top_device_kernels": device_rows[:25],
    }


def _environment(args: argparse.Namespace) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(args.device)
    try:
        import triton

        triton_version = triton.__version__
    except (ImportError, AttributeError):
        triton_version = ""
    status = _git_output("status", "--short")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton_version,
        "gpu_index": args.device,
        "gpu": torch.cuda.get_device_name(args.device),
        "architecture": getattr(properties, "gcnArchName", ""),
        "total_memory_bytes": properties.total_memory,
        "gpu_count": torch.cuda.device_count(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_dirty": bool(status),
        "git_status": status.splitlines(),
        "environment": {
            name: os.environ.get(name, "")
            for name in (
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "NCCL_IB_DISABLE",
            )
        },
    }


def _profile_case(
    case: FFNCase,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    for _ in range(args.warmup):
        output = case.run()
        del output
    torch.cuda.synchronize()
    case.clear_gradients()

    latency = _latency_summary(_gpu_event_samples(case, args.latency_samples))
    _write_json(output_dir / f"{case.slug}.latency.json", latency)

    # The reference deliberately reuses the exact canonical tensor objects but
    # passes no forward-weight cache. In packed mode this exercises the original
    # per-call transpose path independently of the profiled candidate.
    standard_reference_fingerprints = (
        _run_and_fingerprint(case, use_forward_weights=False) if not args.skip_bitwise_hash else {}
    )
    before_profile_fingerprints = _run_and_fingerprint(case) if not args.skip_bitwise_hash else {}
    torch.cuda.synchronize()

    device = torch.device("cuda", args.device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)

    effective_record_shapes = args.record_shapes or args.profile_memory
    effective_with_stack = args.with_stack or args.profile_memory
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=effective_record_shapes,
        profile_memory=args.profile_memory,
        with_stack=effective_with_stack,
        acc_events=True,
    ) as profiler:
        for _ in range(args.active_steps):
            with record_function(f"rl_kernel::{case.direction}"):
                output = case.run()
                del output
            profiler.step()
        torch.cuda.synchronize()

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    memory = {
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": max(0, peak_allocated - baseline_allocated),
        "peak_reserved_delta_bytes": max(0, peak_reserved - baseline_reserved),
    }

    trace_path = output_dir / f"{case.slug}.trace.json.gz"
    profiler.export_chrome_trace(str(trace_path))
    rows = _key_average_rows(
        profiler,
        group_by_input_shape=effective_record_shapes,
    )
    _write_key_averages(output_dir, case.slug, rows)
    kernel_breakdown = _kernel_breakdown(rows)
    _write_json(output_dir / f"{case.slug}.kernel_breakdown.json", kernel_breakdown)

    key_averages = profiler.key_averages(group_by_input_shape=effective_record_shapes)
    summary = "\n\n".join(
        (
            "Sorted by self device time\n"
            + key_averages.table(
                sort_by="self_device_time_total",
                row_limit=args.row_limit,
            ),
            "Sorted by self CPU time\n"
            + key_averages.table(
                sort_by="self_cpu_time_total",
                row_limit=args.row_limit,
            ),
        )
    )
    (output_dir / f"{case.slug}.summary.txt").write_text(summary)

    memory_timeline_error = ""
    if args.profile_memory:
        try:
            profiler.export_memory_timeline(
                str(output_dir / f"{case.slug}.memory.raw.json.gz"),
                device=str(device),
            )
        except (AssertionError, RuntimeError, ValueError) as error:
            memory_timeline_error = f"{type(error).__name__}: {error}"

    after_profile_fingerprints = _run_and_fingerprint(case) if not args.skip_bitwise_hash else {}
    before_matches_reference = {
        name: before_profile_fingerprints.get(name) == fingerprint
        for name, fingerprint in standard_reference_fingerprints.items()
    }
    after_matches_reference = {
        name: after_profile_fingerprints.get(name) == fingerprint
        for name, fingerprint in standard_reference_fingerprints.items()
    }
    after_matches_before = {
        name: after_profile_fingerprints.get(name) == fingerprint
        for name, fingerprint in before_profile_fingerprints.items()
    }
    all_match = (
        None
        if args.skip_bitwise_hash
        else (
            all(before_matches_reference.values())
            and all(after_matches_reference.values())
            and all(after_matches_before.values())
        )
    )
    correctness = {
        "comparison": "SHA256 over the contiguous raw tensor bytes",
        "reference_mode": (
            "standard qwen3_ffn call with forward_weights=None using the same "
            "canonical input and weight tensors"
        ),
        "candidate_mode": args.weight_layout,
        "skipped": args.skip_bitwise_hash,
        "all_match": all_match,
        "matches_standard_reference_before_profile": before_matches_reference,
        "matches_standard_reference_after_profile": after_matches_reference,
        "matches_before_profile_after_profile": after_matches_before,
        "standard_reference": standard_reference_fingerprints,
        "before_profile": before_profile_fingerprints,
        "after_profile": after_profile_fingerprints,
    }
    _write_json(output_dir / f"{case.slug}.correctness.json", correctness)
    case.clear_gradients()

    return {
        "direction": case.direction,
        "weight_layout": args.weight_layout,
        "weight_layout_metadata": _weight_layout_metadata(case),
        "files": {
            "trace": trace_path.name,
            "key_averages_json": f"{case.slug}.key_averages.json",
            "key_averages_csv": f"{case.slug}.key_averages.csv",
            "summary": f"{case.slug}.summary.txt",
            "kernel_breakdown": f"{case.slug}.kernel_breakdown.json",
            "latency": f"{case.slug}.latency.json",
            "correctness": f"{case.slug}.correctness.json",
            "memory_timeline": (f"{case.slug}.memory.raw.json.gz" if args.profile_memory else ""),
        },
        "profiler": {
            "active_steps": args.active_steps,
            "record_shapes": effective_record_shapes,
            "profile_memory": args.profile_memory,
            "with_stack": effective_with_stack,
            "trace_size_bytes": trace_path.stat().st_size,
            "memory_timeline_error": memory_timeline_error,
        },
        "latency": latency,
        "memory": memory,
        "correctness_all_match": correctness["all_match"],
        "kernel_breakdown": kernel_breakdown,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if getattr(torch.version, "hip", None) is None:
        raise RuntimeError("this profiler requires a ROCm PyTorch build")
    if not torch.cuda.is_available():
        raise RuntimeError("no ROCm GPU is available")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(
            f"--device must be in [0, {torch.cuda.device_count() - 1}], got {args.device}"
        )
    for name in (
        "tokens",
        "hidden_size",
        "intermediate_size",
        "active_steps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0 or args.latency_samples < 0:
        raise ValueError("--warmup and --latency-samples must be non-negative")
    if args.profile_memory and args.active_steps > 1:
        print(
            "warning: memory/shape/stack profiling adds overhead; "
            "prefer --active-steps 1 for a memory run",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/rl_kernel_torch_profiler"),
    )
    parser.add_argument(
        "--direction",
        choices=("forward", "forward-backward", "both"),
        default="both",
    )
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=12288)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--weight-layout",
        choices=("standard", "packed"),
        default="standard",
        help=(
            "standard materializes forward transposes in every FFN call; packed "
            "materializes detached forward-only weights once before measurement"
        ),
    )
    parser.add_argument("--weight-seed", type=int, default=3000)
    parser.add_argument("--input-seed", type=int)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--active-steps", type=int, default=5)
    parser.add_argument("--latency-samples", type=int, default=20)
    parser.add_argument("--row-limit", type=int, default=100)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--skip-bitwise-hash", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=True)

    directions = ("forward", "forward-backward") if args.direction == "both" else (args.direction,)
    manifest: dict[str, Any] = {
        "environment": _environment(args),
        "workload": {
            "operator": "rl_engine.kernels.ops.triton.ffn.qwen3_ffn",
            "tokens": args.tokens,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "dtype": "torch.bfloat16",
            "weight_layout": args.weight_layout,
            "weight_seeds": [args.weight_seed + offset for offset in range(3)],
            "input_seed": (
                args.input_seed
                if args.input_seed is not None
                else _INPUT_SEEDS.get(args.tokens, 3010)
            ),
            "grad_output_seed": (
                args.input_seed + 1
                if args.input_seed is not None
                else _INPUT_SEEDS.get(args.tokens, 3010) + 1
            ),
            "warmup": args.warmup,
            "latency_samples": args.latency_samples,
            "profiler_active_steps": args.active_steps,
        },
        "methodology": {
            "profiler_use": "attribution and launch analysis only",
            "latency_use": "uninstrumented GPU events",
            "jit_and_tree_plan": "warmed before profiler starts",
            "weight_packing": ("performed once during case construction outside all timing"),
            "bitwise_fingerprint": "SHA256 over raw BF16 bytes",
            "packed_bitwise_reference": (
                "uncached standard qwen3_ffn path using the same canonical tensors"
            ),
        },
        "profiles": [],
    }
    _write_json(args.output_dir / "manifest.json", manifest)

    for direction in directions:
        print(f"profiling {direction} on cuda:{args.device} ...", flush=True)
        case = _build_case(args, direction)
        result = _profile_case(case, args, args.output_dir)
        manifest["profiles"].append(result)
        _write_json(args.output_dir / "manifest.json", manifest)
        latency = result["latency"]
        if latency.get("samples_ms"):
            print(
                f"  uninstrumented median={latency['median_ms']:.4f} ms, "
                f"p95={latency['p95_ms']:.4f} ms",
                flush=True,
            )
        print(
            f"  raw-bit fingerprints match: {result['correctness_all_match']}",
            flush=True,
        )
        del case
        torch.cuda.empty_cache()

    print(f"profiler artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
