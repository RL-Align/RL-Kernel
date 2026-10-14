# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark deterministic ROCm collectives against native RCCL.

Example:

    torchrun --standalone --nproc-per-node=8 \
      benchmarks/distributed/benchmark_rocm_collectives.py \
      --size-bytes 4096 65536 1048576 16777216 \
      --output benchmarks/results/rocm_collectives_mi300x.json

The native RCCL rows are performance references only.  They are not used as a
bitwise correctness oracle because their floating-point reduction order is not
part of the strict deterministic contract.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.distributed as dist

from rl_engine.distributed import RCCLDeterministicCollective

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
_OPERATIONS = ("all_reduce", "all_gather", "reduce_scatter")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size-bytes",
        type=int,
        nargs="+",
        default=[4 * 1024, 64 * 1024, 1024 * 1024, 16 * 1024 * 1024],
    )
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bf16")
    parser.add_argument("--operations", nargs="+", choices=_OPERATIONS, default=_OPERATIONS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if any(size <= 0 for size in args.size_bytes):
        raise ValueError("every --size-bytes value must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iterations <= 0 or args.samples <= 0:
        raise ValueError("--iterations and --samples must be positive")


def _timed_sample(operation: Callable[[], None], *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iterations

    # Report the slowest rank, which is the end-to-end collective latency.
    elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64, device="cuda")
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    return float(elapsed_tensor.item())


def _benchmark(
    operation: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    samples: int,
) -> dict[str, object]:
    timings = [
        _timed_sample(operation, warmup=warmup if index == 0 else 0, iterations=iterations)
        for index in range(samples)
    ]
    median = statistics.median(timings)
    return {
        "median_us": median * 1.0e6,
        "min_us": min(timings) * 1.0e6,
        "max_us": max(timings) * 1.0e6,
        "samples_us": [value * 1.0e6 for value in timings],
    }


def _make_inputs(
    *,
    size_bytes: int,
    dtype: torch.dtype,
    world_size: int,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    element_size = torch.empty((), dtype=dtype).element_size()
    elements = max(world_size, size_bytes // element_size)
    elements -= elements % world_size
    generator = torch.Generator(device="cpu").manual_seed(942 + rank)
    tensor = torch.randn(elements, generator=generator, dtype=torch.float32).to(
        device=device,
        dtype=dtype,
    )
    return tensor.contiguous(), elements * element_size


def _operation_pair(
    name: str,
    input_tensor: torch.Tensor,
    collective: RCCLDeterministicCollective,
    world_size: int,
) -> tuple[Callable[[], None], Callable[[], None], torch.Tensor, torch.Tensor]:
    if name == "all_reduce":
        deterministic_out = torch.empty_like(input_tensor)
        native_out = torch.empty_like(input_tensor)

        def deterministic() -> None:
            collective.all_reduce(input_tensor, out=deterministic_out)

        def native() -> None:
            native_out.copy_(input_tensor)
            dist.all_reduce(native_out)

    elif name == "all_gather":
        output_shape = (input_tensor.numel() * world_size,)
        deterministic_out = torch.empty(output_shape, dtype=input_tensor.dtype, device="cuda")
        native_out = torch.empty_like(deterministic_out)

        def deterministic() -> None:
            collective.all_gather(input_tensor, out=deterministic_out)

        def native() -> None:
            dist.all_gather_into_tensor(native_out, input_tensor)

    elif name == "reduce_scatter":
        output_shape = (input_tensor.numel() // world_size,)
        deterministic_out = torch.empty(output_shape, dtype=input_tensor.dtype, device="cuda")
        native_out = torch.empty_like(deterministic_out)

        def deterministic() -> None:
            collective.reduce_scatter(input_tensor, out=deterministic_out)

        def native() -> None:
            dist.reduce_scatter_tensor(native_out, input_tensor)

    else:  # pragma: no cover - argparse constrains this value
        raise ValueError(f"unsupported operation: {name}")

    return deterministic, native, deterministic_out, native_out


def run(args: argparse.Namespace) -> dict[str, object] | None:
    _validate_args(args)
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("the ROCm collective benchmark requires an available AMD GPU")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (2, 4, 8):
        raise RuntimeError(f"the benchmark requires 2, 4, or 8 ranks, got {world_size}")
    device = torch.device("cuda", local_rank)
    dtype = _DTYPES[args.dtype]
    max_size_bytes = max(args.size_bytes) + dtype.itemsize * world_size

    rows: list[dict[str, object]] = []
    try:
        with RCCLDeterministicCollective(
            device=device,
            max_size_bytes=max_size_bytes,
        ) as collective:
            for requested_size in args.size_bytes:
                input_tensor, actual_size = _make_inputs(
                    size_bytes=requested_size,
                    dtype=dtype,
                    world_size=world_size,
                    rank=rank,
                    device=device,
                )
                for name in args.operations:
                    deterministic, native, deterministic_out, native_out = _operation_pair(
                        name,
                        input_tensor,
                        collective,
                        world_size,
                    )
                    deterministic()
                    deterministic_repeat = deterministic_out.clone()
                    deterministic()
                    repeat_bitwise = bool(torch.equal(deterministic_out, deterministic_repeat))
                    native()
                    max_abs_vs_native = float(
                        (deterministic_out.float() - native_out.float()).abs().max().item()
                    )

                    torch.cuda.reset_peak_memory_stats(device)
                    deterministic_timing = _benchmark(
                        deterministic,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        samples=args.samples,
                    )
                    deterministic_peak = int(torch.cuda.max_memory_allocated(device))
                    torch.cuda.reset_peak_memory_stats(device)
                    native_timing = _benchmark(
                        native,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        samples=args.samples,
                    )
                    native_peak = int(torch.cuda.max_memory_allocated(device))
                    deterministic_us = float(deterministic_timing["median_us"])
                    native_us = float(native_timing["median_us"])
                    rows.append(
                        {
                            "operation": name,
                            "requested_size_bytes": requested_size,
                            "actual_input_bytes": actual_size,
                            "dtype": args.dtype,
                            "deterministic": deterministic_timing,
                            "native_rccl": native_timing,
                            "latency_ratio_vs_native": deterministic_us / native_us,
                            "deterministic_input_gbps": actual_size / (deterministic_us * 1.0e3),
                            "native_input_gbps": actual_size / (native_us * 1.0e3),
                            "repeat_bitwise": repeat_bitwise,
                            "max_abs_vs_native": max_abs_vs_native,
                            "deterministic_workspace_bytes": collective.workspace_size_bytes,
                            "deterministic_peak_allocated_bytes": deterministic_peak,
                            "native_peak_allocated_bytes": native_peak,
                        }
                    )

        reports: list[list[dict[str, object]] | None] = [None] * world_size
        dist.all_gather_object(reports, rows)
        if rank != 0:
            return None
        payload = {
            "schema_version": "rlkernel.rocm_collective_benchmark.v1",
            "world_size": world_size,
            "device": torch.cuda.get_device_name(device),
            "hip_version": torch.version.hip,
            "collective_backend": RCCLDeterministicCollective.backend_id,
            "reduction_order": RCCLDeterministicCollective.reduction_order,
            "supports_compute_communication_fusion": False,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "samples": args.samples,
            "rows": rows,
            "all_rank_repeat_bitwise": all(
                bool(row["repeat_bitwise"])
                for rank_rows in reports
                if rank_rows is not None
                for row in rank_rows
            ),
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    finally:
        dist.destroy_process_group()


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(parse_args(argv))
    if payload is not None:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
