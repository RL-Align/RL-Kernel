# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.platforms.device import device_ctx  # noqa: E402
from rl_engine.utils.logger import logger  # noqa: E402
from rl_engine.validation.common import selected_logprobs_reference  # noqa: E402


@dataclass(frozen=True)
class GPUTargetInfo:
    """Hardware target identification for cross-GPU benchmarking."""

    name: str
    architecture: str
    total_memory_gb: float
    driver_version: str
    backend: str
    compute_capability: str | None = None
    device_index: int = 0


@dataclass
class BenchmarkMetrics:
    """End-to-end performance metrics for a single benchmark run."""

    # Identification
    timestamp: str
    benchmark_name: str
    gpu_target: GPUTargetInfo

    # Workload shape
    batch_size: int
    seq_len: int
    vocab_size: int
    total_tokens: int

    # Timing
    latency_ms: float
    latency_std_ms: float | None = None
    warmup_iterations: int = 0
    repeat_iterations: int = 1

    # Throughput & Compute
    tokens_per_sec: float = 0.0
    tflops: float = 0.0

    # Memory
    peak_vram_gb: float = 0.0
    peak_vram_reduction_gb: float | None = None

    # Status & Metadata
    status: str = "pass"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        base = asdict(self)
        base["gpu_target"] = asdict(self.gpu_target)
        return base


class GPUProfiler:
    """Automatically detect and profile GPU hardware targets."""

    @staticmethod
    def get_target_info(device_index: int = 0) -> GPUTargetInfo:
        if not torch.cuda.is_available():
            return GPUTargetInfo(
                name="CPU",
                architecture="unknown",
                total_memory_gb=0.0,
                driver_version="N/A",
                backend="cpu",
                compute_capability=None,
                device_index=-1,
            )

        if device_index is None or device_index < 0:
            device_index = torch.cuda.current_device()
        device = torch.device(f"cuda:{device_index}")
        name = torch.cuda.get_device_name(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)

        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            backend = "rocm"
            driver = torch.version.hip or "unknown"
            # ROCm architecture detection from device name
            arch = GPUProfiler._infer_amd_arch(name)
            cc = None
        else:
            backend = "cuda"
            driver = torch.version.cuda or "unknown"
            major, minor = torch.cuda.get_device_capability(device)
            arch = GPUProfiler._infer_nvidia_arch(name, major, minor)
            cc = f"{major}.{minor}"

        return GPUTargetInfo(
            name=name,
            architecture=arch,
            total_memory_gb=round(total_mem, 2),
            driver_version=driver,
            backend=backend,
            compute_capability=cc,
            device_index=device_index,
        )

    @staticmethod
    def _infer_nvidia_arch(name: str, major: int, minor: int) -> str:
        mapping = {
            (7, 0): "Volta",
            (7, 5): "Turing",
            (8, 0): "Ampere",
            (8, 6): "Ampere",
            (8, 7): "Ampere",
            (8, 9): "Ada Lovelace",
            (9, 0): "Hopper",
            (10, 0): "Blackwell",
        }
        return mapping.get((major, minor), f"SM{major}{minor}")

    @staticmethod
    def _infer_amd_arch(name: str) -> str:
        name_lower = name.lower()
        if "mi300" in name_lower:
            return "CDNA3"
        if "mi250" in name_lower:
            return "CDNA2"
        if "mi100" in name_lower:
            return "CDNA1"
        if "rx 7900" in name_lower or "gfx1100" in name_lower:
            return "RDNA3"
        if "gfx90a" in name_lower:
            return "CDNA2"
        return "AMD-GCN/RDNA"


class PerformanceProfiler:
    """
    Automated end-to-end performance profiling suite.

    Measures Tokens/sec, TFLOPS, and peak VRAM across different GPU targets
    for RL-Kernel operators and sampling pipelines.
    """

    def __init__(self, device: torch.device | None = None, warmup: int = 3, repeat: int = 10):
        self.device = device if device is not None else device_ctx.device
        self.warmup = warmup
        self.repeat = repeat
        self.gpu_info = GPUProfiler.get_target_info(
            device_index=self.device.index if self.device.type == "cuda" else -1
        )
        self._metrics: list[BenchmarkMetrics] = []

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _reset_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def _peak_memory_gb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024**3)

    def _time_kernel(self, fn) -> tuple[Any, float, float | None]:
        """Run fn with warmup + repeat, return (last_result, median_ms, std_ms)."""
        result = None
        for _ in range(max(0, self.warmup)):
            result = fn()
        self._sync()

        elapsed: list[float] = []
        self._reset_memory()
        for _ in range(max(1, self.repeat)):
            if self.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = fn()
                end.record()
                end.synchronize()
                elapsed.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                result = fn()
                self._sync()
                elapsed.append((time.perf_counter() - t0) * 1000.0)

        self._sync()
        median_ms = statistics.median(elapsed)
        std_ms = statistics.stdev(elapsed) if len(elapsed) > 1 else None
        return result, median_ms, std_ms

    def _measure_peak_memory_once(self, fn) -> float:
        """Measure peak allocation for one invocation after the configured warmup."""
        for _ in range(max(0, self.warmup)):
            _ = fn()
        self._sync()
        self._reset_memory()
        _ = fn()
        self._sync()
        return self._peak_memory_gb()

    @staticmethod
    def estimate_logp_tflops(batch_size: int, seq_len: int, vocab_size: int) -> float:
        """
        Estimate floating-point operations for a single logp forward pass.

        log_softmax ~= 5 * B * S * V  FLOPs
        (subtract max, exp, sum, divide, log)
        gather is negligible.
        """
        return (5.0 * batch_size * seq_len * vocab_size) / 1e12

    @staticmethod
    def estimate_sampling_tflops(batch_size: int, vocab_size: int) -> float:
        """
        Estimate FLOPs for a single sampling step.

        softmax ~= 5 * B * V FLOPs
        top-k and multinomial are comparatively negligible.
        """
        return (5.0 * batch_size * vocab_size) / 1e12

    def profile_logp(
        self,
        candidate_fn,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        benchmark_name: str = "logp_forward",
        native_fn=None,
    ) -> BenchmarkMetrics:
        """
        Profile a log-probability operator and compute end-to-end metrics.

        Args:
            candidate_fn: Callable that performs the fused/candidate logp computation.
            native_fn: Optional baseline callable for VRAM reduction comparison.
        """
        logger.info(f"Profiling '{benchmark_name}' | shape=({batch_size},{seq_len},{vocab_size})")

        total_tokens = batch_size * seq_len
        tflops_per_call = self.estimate_logp_tflops(batch_size, seq_len, vocab_size)

        try:
            _, latency_ms, latency_std = self._time_kernel(candidate_fn)
            peak_vram = (
                self._measure_peak_memory_once(candidate_fn)
                if native_fn is not None and self.device.type == "cuda"
                else self._peak_memory_gb()
            )
        except torch.cuda.OutOfMemoryError as exc:
            metrics = BenchmarkMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                benchmark_name=benchmark_name,
                gpu_target=self.gpu_info,
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                total_tokens=total_tokens,
                latency_ms=float("inf"),
                latency_std_ms=None,
                status="oom",
                notes=str(exc),
            )
            self._metrics.append(metrics)
            return metrics

        latency_s = latency_ms / 1000.0
        tokens_per_sec = total_tokens / latency_s if latency_s > 0 else 0.0
        tflops = tflops_per_call / latency_s if latency_s > 0 else 0.0

        vram_reduction = None
        if native_fn is not None and self.device.type == "cuda":
            try:
                native_peak = self._measure_peak_memory_once(native_fn)
                vram_reduction = max(0.0, native_peak - peak_vram)
            except Exception as exc:
                logger.warning(f"Native baseline failed for VRAM comparison: {exc}")

        metrics = BenchmarkMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            benchmark_name=benchmark_name,
            gpu_target=self.gpu_info,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            latency_std_ms=latency_std,
            warmup_iterations=self.warmup,
            repeat_iterations=self.repeat,
            tokens_per_sec=tokens_per_sec,
            tflops=tflops,
            peak_vram_gb=peak_vram,
            peak_vram_reduction_gb=vram_reduction,
            status="pass",
        )
        self._metrics.append(metrics)
        return metrics

    def profile_sampling(
        self,
        candidate_fn,
        batch_size: int,
        vocab_size: int,
        seq_len: int = 1,
        benchmark_name: str = "sampling_step",
    ) -> BenchmarkMetrics:
        """
        Profile a sampling operator and compute end-to-end metrics.
        """
        logger.info(f"Profiling '{benchmark_name}' | shape=({batch_size},{vocab_size})")

        total_tokens = batch_size * seq_len
        tflops_per_call = self.estimate_sampling_tflops(batch_size, vocab_size)

        try:
            _, latency_ms, latency_std = self._time_kernel(candidate_fn)
            peak_vram = self._peak_memory_gb()
        except torch.cuda.OutOfMemoryError as exc:
            metrics = BenchmarkMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                benchmark_name=benchmark_name,
                gpu_target=self.gpu_info,
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                total_tokens=total_tokens,
                latency_ms=float("inf"),
                latency_std_ms=None,
                status="oom",
                notes=str(exc),
            )
            self._metrics.append(metrics)
            return metrics

        latency_s = latency_ms / 1000.0
        tokens_per_sec = total_tokens / latency_s if latency_s > 0 else 0.0
        tflops = tflops_per_call / latency_s if latency_s > 0 else 0.0

        metrics = BenchmarkMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            benchmark_name=benchmark_name,
            gpu_target=self.gpu_info,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            latency_std_ms=latency_std,
            warmup_iterations=self.warmup,
            repeat_iterations=self.repeat,
            tokens_per_sec=tokens_per_sec,
            tflops=tflops,
            peak_vram_gb=peak_vram,
            status="pass",
        )
        self._metrics.append(metrics)
        return metrics

    def get_metrics(self) -> list[BenchmarkMetrics]:
        return list(self._metrics)

    def clear_metrics(self) -> None:
        self._metrics.clear()

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "report_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "gpu_target": asdict(self.gpu_info),
            "metrics": [m.to_dict() for m in self._metrics],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Performance report saved to {path}")

    def save_csv(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._metrics:
            logger.warning("No metrics to write to CSV.")
            return

        fieldnames = list(self._metrics[0].to_dict().keys())
        # Flatten gpu_target nested dict into prefixed keys for CSV
        flat_rows: list[dict[str, Any]] = []
        for m in self._metrics:
            row = m.to_dict()
            gpu = row.pop("gpu_target")
            for k, v in gpu.items():
                row[f"gpu_{k}"] = v
            # Flatten extra dict
            extra = row.pop("extra", {})
            for k, v in extra.items():
                row[f"extra_{k}"] = v
            flat_rows.append(row)

        # Recompute fieldnames from flattened rows to ensure completeness
        all_keys = set()
        for r in flat_rows:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)

        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerows(flat_rows)
        logger.info(f"CSV metrics appended to {path}")

    def print_summary(self) -> None:
        from tabulate import tabulate

        if not self._metrics:
            print("No metrics collected.")
            return

        rows = []
        for m in self._metrics:
            rows.append(
                [
                    m.benchmark_name,
                    m.batch_size,
                    m.seq_len,
                    m.vocab_size,
                    f"{m.latency_ms:.2f}",
                    f"{m.tokens_per_sec:,.0f}",
                    f"{m.tflops:.2f}",
                    f"{m.peak_vram_gb:.2f}",
                    m.status,
                ]
            )

        headers = [
            "Benchmark",
            "Batch",
            "SeqLen",
            "Vocab",
            "Latency(ms)",
            "Tokens/sec",
            "TFLOPS",
            "Peak VRAM(GB)",
            "Status",
        ]
        print("\n" + "=" * 120)
        print(f"{'RL-KERNEL AUTOMATED PERFORMANCE PROFILING SUITE':^120}")
        print(
            f"GPU: {self.gpu_info.name} | Arch: {self.gpu_info.architecture} "
            f"| Backend: {self.gpu_info.backend}"
        )
        print("=" * 120)
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
        print("=" * 120 + "\n")


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def _native_logp_fn(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    logits = torch.randn(
        batch_size,
        seq_len,
        vocab_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    token_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        device=device,
        generator=generator,
    )
    return lambda: selected_logprobs_reference(logits, token_ids)


def _fused_logp_fn(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
):
    from rl_engine.runtime.registry import kernel_registry

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    logits = torch.randn(
        batch_size,
        seq_len,
        vocab_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    token_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        device=device,
        generator=generator,
    )
    op = kernel_registry.get_op("logp")
    return lambda: op.apply_fp32(logits, token_ids)


def _sampling_fn(
    *,
    batch_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    top_k: int,
    top_p: float,
):
    from benchmarks.operators.sampling.benchmark_sampling import native_sampling

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype, generator=generator)
    return lambda: native_sampling(logits.clone(), top_k=top_k, top_p=top_p)


def _blocked_metric(
    *,
    profiler: PerformanceProfiler,
    benchmark_name: str,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    exc: Exception,
) -> BenchmarkMetrics:
    metric = BenchmarkMetrics(
        timestamp=datetime.now(timezone.utc).isoformat(),
        benchmark_name=benchmark_name,
        gpu_target=profiler.gpu_info,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        total_tokens=batch_size * seq_len,
        latency_ms=float("inf"),
        warmup_iterations=profiler.warmup,
        repeat_iterations=profiler.repeat,
        status="blocked",
        notes=str(exc).splitlines()[0],
    )
    profiler._metrics.append(metric)
    return metric


WorkloadRunner = Callable[
    [PerformanceProfiler, argparse.Namespace, torch.device, torch.dtype, int, int, int],
    BenchmarkMetrics,
]


def _run_logp_native_workload(
    profiler: PerformanceProfiler,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> BenchmarkMetrics:
    fn = _native_logp_fn(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )
    return profiler.profile_logp(
        fn,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        benchmark_name="logp_native",
    )


def _run_logp_fused_workload(
    profiler: PerformanceProfiler,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> BenchmarkMetrics:
    try:
        if device.type != "cuda":
            raise RuntimeError("logp-fused requires CUDA")
        fused = _fused_logp_fn(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=dtype,
            device=device,
            seed=args.seed + 1,
        )
        native = _native_logp_fn(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=dtype,
            device=device,
            seed=args.seed + 1,
        )
        return profiler.profile_logp(
            fused,
            native_fn=native,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            benchmark_name="logp_fused",
        )
    except Exception as exc:
        return _blocked_metric(
            profiler=profiler,
            benchmark_name="logp_fused",
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            exc=exc,
        )


def _run_sampling_native_workload(
    profiler: PerformanceProfiler,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> BenchmarkMetrics:
    try:
        fn = _sampling_fn(
            batch_size=batch_size,
            vocab_size=vocab_size,
            dtype=torch.float32 if device.type == "cpu" else dtype,
            device=device,
            seed=args.seed + 2,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        return profiler.profile_sampling(
            fn,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            benchmark_name="sampling_native",
        )
    except Exception as exc:
        return _blocked_metric(
            profiler=profiler,
            benchmark_name="sampling_native",
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            exc=exc,
        )


WORKLOAD_REGISTRY: dict[str, WorkloadRunner] = {
    "logp-native": _run_logp_native_workload,
    "logp-fused": _run_logp_fused_workload,
    "sampling-native": _run_sampling_native_workload,
}


def available_workloads() -> tuple[str, ...]:
    return tuple(WORKLOAD_REGISTRY)


def _parse_workloads(value: str) -> list[str]:
    workloads = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(workloads) - set(WORKLOAD_REGISTRY))
    if unknown:
        supported = ", ".join(available_workloads())
        raise ValueError(f"unsupported workloads: {', '.join(unknown)}; supported: {supported}")
    return workloads


def run_automated_suite(args: argparse.Namespace) -> PerformanceProfiler:
    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)
    profiler = PerformanceProfiler(device=device, warmup=args.warmup, repeat=args.repeat)

    if args.smoke:
        batch_sizes = [2]
        seq_lens = [4]
        vocab_sizes = [32]
    else:
        batch_sizes = _parse_int_list(args.batch_sizes)
        seq_lens = _parse_int_list(args.seq_lens)
        vocab_sizes = _parse_int_list(args.vocab_sizes)

    workloads = _parse_workloads(args.workloads)
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for vocab_size in vocab_sizes:
                for workload in workloads:
                    runner = WORKLOAD_REGISTRY[workload]
                    runner(profiler, args, device, dtype, batch_size, seq_len, vocab_size)
    return profiler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL-Kernel automated performance profiler")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-sizes", default="8,16,32")
    parser.add_argument("--seq-lens", default="128,512")
    parser.add_argument("--vocab-sizes", default="4096,128256")
    parser.add_argument(
        "--workloads",
        default=",".join(available_workloads()),
        help=f"Comma-separated workloads: {', '.join(available_workloads())}",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.smoke:
        args.dtype = "float32"
        args.warmup = 0
        args.repeat = 1
    profiler = run_automated_suite(args)
    if args.output is not None:
        if args.format == "json":
            profiler.save_json(args.output)
        else:
            profiler.save_csv(args.output)
    if not args.no_summary:
        profiler.print_summary()


if __name__ == "__main__":
    main()
