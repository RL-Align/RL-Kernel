# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark the CUDA deterministic collectives across message sizes.

Example:

    torchrun --standalone --nproc-per-node=8 \
      benchmarks/benchmark_deterministic_collectives.py \
      --output benchmarks/results/deterministic_collectives_b200.json

``--size-bytes`` is the per-rank input. The default sweep straddles the
single-block fast paths (64 KiB gathered output for ``all_gather``, 256 KiB
for the ``reduce_scatter`` input and ``all_reduce``). NCCL rows are
performance references only; they are not a correctness oracle.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.distributed as dist

from rl_engine.distributed.collectives import DeterministicCollective

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_OPERATIONS = ("all_gather", "all_gather_many", "reduce_scatter", "all_reduce")
_DEFAULT_SIZES = [
    1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    96 * 1024,
    128 * 1024,
    192 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the message-size sweep and distributed benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-bytes", type=int, nargs="+", default=_DEFAULT_SIZES)
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bf16")
    parser.add_argument("--operations", nargs="+", choices=_OPERATIONS, default=_OPERATIONS)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _median_us(
    fn: Callable[[], object],
    warmup: int,
    iterations: int,
    *,
    prepare: Callable[[], object] | None = None,
) -> float:
    """Return median CUDA-event time, excluding preparation before each call."""
    for _ in range(warmup):
        if prepare is not None:
            prepare()
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if prepare is not None:
            prepare()
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return statistics.median(samples)


def _slowest(value: float) -> float:
    """Return the largest per-rank timing across the process group."""
    tensor = torch.tensor([value], device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def main(argv: Sequence[str] | None = None) -> None:
    """Measure selected collectives on every rank and save the rank-zero report."""
    args = parse_args(argv)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    dtype = _DTYPES[args.dtype]
    element = torch.empty((), dtype=dtype).element_size()
    capacity = max(args.size_bytes) * world
    collective = DeterministicCollective(device=rank, max_size_bytes=capacity)
    rows = []
    try:
        for size in args.size_bytes:
            numel = size // element
            x = torch.randn(numel, device="cuda").to(dtype)
            gathered = torch.empty(numel * world, device="cuda", dtype=dtype)
            reduce_in = torch.randn(numel * world, device="cuda").to(dtype)
            nccl_scattered = torch.empty_like(x)
            nccl_reduced = torch.empty_like(x)
            cases: dict[str, tuple[Callable[[], object], Callable[[], object]]] = {
                "all_gather": (
                    partial(collective.all_gather, x, out=gathered),
                    partial(dist.all_gather_into_tensor, gathered, x),
                ),
                "all_gather_many": (
                    partial(collective.all_gather_many, (x,)),
                    partial(dist.all_gather_into_tensor, gathered, x),
                ),
                "reduce_scatter": (
                    partial(collective.reduce_scatter, reduce_in),
                    partial(dist.reduce_scatter_tensor, nccl_scattered, reduce_in),
                ),
                "all_reduce": (
                    partial(collective.all_reduce, x),
                    partial(dist.all_reduce, nccl_reduced),
                ),
            }
            for op in args.operations:
                deterministic, nccl = cases[op]
                prepare = partial(nccl_reduced.copy_, x) if op == "all_reduce" else None
                row = {
                    "operation": op,
                    "input_bytes_per_rank": numel * element,
                    "deterministic_us": _slowest(
                        _median_us(deterministic, args.warmup, args.iterations)
                    ),
                    "nccl_us": _slowest(
                        _median_us(nccl, args.warmup, args.iterations, prepare=prepare)
                    ),
                }
                rows.append(row)
                if rank == 0:
                    print(
                        f"{op:16s} {row['input_bytes_per_rank']:>9d} B/rank  "
                        f"deterministic {row['deterministic_us']:8.1f} us  "
                        f"nccl {row['nccl_us']:8.1f} us",
                        flush=True,
                    )
    finally:
        collective.close()
    if rank == 0 and args.output is not None:
        props = torch.cuda.get_device_properties(rank)
        report = {
            "kind": "deterministic_collectives_benchmark",
            "world_size": world,
            "dtype": args.dtype,
            "gpu": props.name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
