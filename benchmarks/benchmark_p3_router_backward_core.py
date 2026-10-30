# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Standalone T06 microbenchmark; not official P3 or model-level acceptance.

Run from the repository root (no full RL-Kernel installation required):
  python -m benchmarks.benchmark_p3_router_backward_core --json /tmp/t06.json

Graph CUDA events measure amortized device time on repeated, resident inputs.
Eager synchronized wall time includes host dispatch; the public harness also
includes allocation and value checks. The Torch baseline is a deliberately
fixed-order eager decomposition, not Megatron/vLLM's native Router backward.
"""

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from examples.p3_router_backward.prototype import (
    _load_extension,
    diagnostic_reference,
    fixed_tree6,
    route_backward_core,
)


def make_case(tokens, duplicate, padding_fraction, seed):
    rng = torch.Generator().manual_seed(seed)
    scores = torch.rand((tokens, 256), generator=rng) + 0.05
    # Unique slots use all expert IDs across rows; duplicate cases additionally
    # exercise accumulation rather than simply timing six distinct writes.
    if duplicate:
        ids = torch.randint(0, 256, (tokens, 3), generator=rng, dtype=torch.int32)
        ids = ids[:, [0, 0, 1, 2, 0, 1]].contiguous()
    else:
        ids = torch.argsort(torch.rand((tokens, 256), generator=rng), dim=1)[:, :6]
        ids = ids.to(torch.int32).contiguous()
    a = scores.gather(1, ids.long())
    z = fixed_tree6(a) + 1e-20
    p = a / z[:, None]
    g = torch.randn((tokens, 6), generator=rng)
    active = torch.ones(tokens, dtype=torch.bool)
    inactive_count = int(tokens * padding_fraction)
    active[torch.randperm(tokens, generator=rng)[:inactive_count]] = False
    # Poison padding to verify that both implementations actually mask it.
    g[~active] = float("nan")
    p[~active] = float("nan")
    z[~active] = 0
    ids[~active] = -1
    return g, ids, p, z, active


@torch.no_grad()
def torch_fixed_order(dweights, ids, p, z, active):
    """GPU eager comparator with slot-ordered duplicate sums and no atomics."""
    mask = active[:, None]
    g = torch.where(mask, dweights, 0.0)
    p = torch.where(mask, p, 0.0)
    z = torch.where(active, z, 1.0)
    ids = torch.where(mask, ids, 0).long()
    c = fixed_tree6(g * p)
    factor = torch.div(torch.full_like(z, 1.5), z)
    da = factor[:, None] * (g - c[:, None])
    out = torch.zeros((dweights.shape[0], 256), device=dweights.device, dtype=dweights.dtype)
    for slot in range(6):
        index = ids[:, slot : slot + 1]
        # One index per row per launch; different slots execute sequentially.
        out.scatter_(1, index, out.gather(1, index) + da[:, slot : slot + 1])
    return torch.where(out == 0, 0.0, out)


def assert_bits_equal(actual, expected):
    if not torch.equal(actual.cpu().view(torch.int32), expected.view(torch.int32)):
        raise RuntimeError("Benchmark correctness gate failed: FP32 output bits differ")


def summarize(samples):
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples),
        "p95_us": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_us": ordered[0],
        "samples_us": samples,
    }


def wall_time(fn, expected, warmup, samples):
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        output = fn()
        torch.cuda.synchronize()
        durations.append((time.perf_counter_ns() - start) / 1000)
    assert_bits_equal(output, expected)
    return summarize(durations)


def graph_time(fn, expected, warmup, samples, graph_batch):
    # A batch of graph nodes amortizes Python/launch gaps. This is NOT isolated
    # eager call latency or a claim that the synchronous public ABI is capturable.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(graph_batch):
            output = fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    assert_bits_equal(output, expected)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    durations = []
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end) * 1000 / graph_batch)
    assert_bits_equal(output, expected)
    return summarize(durations)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = (
        "csrc/cuda/moe/router_backward_core.cu",
        "examples/p3_router_backward/bindings.cpp",
        "examples/p3_router_backward/prototype.py",
        "tests/test_p3_router_backward_core.py",
        "benchmarks/benchmark_p3_router_backward_core.py",
    )
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=positive_int, default=[1, 16, 128, 512, 4096])
    parser.add_argument("--threads", nargs="+", type=int, choices=[128, 256], default=[128, 256])
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--samples", type=positive_int, default=25)
    parser.add_argument("--graph-batch", type=positive_int, default=32)
    parser.add_argument("--padding-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=106)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not 0 <= args.padding_fraction < 1:
        parser.error("--padding-fraction must be in [0, 1)")
    if torch.version.cuda is None or not torch.cuda.is_available():
        parser.error("an NVIDIA CUDA device is required; no CPU timing fallback")
    torch.set_num_threads(1)
    extension = _load_extension()  # Compile before timing.
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = {
        "schema": "p3-t06-core-benchmark.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic standalone arithmetic; not native Router or official P3 acceptance",
        "environment": {
            "device": props.name,
            "compute_capability": [props.major, props.minor],
            "total_memory_bytes": props.total_memory,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "settings": {key: value for key, value in vars(args).items() if key != "json"},
        "source_sha256": source_hashes(),
        "timing_notes": {
            "graph_device": "CUDA events / graph_batch; resident inputs; CPU allocation excluded",
            "eager_wall": "One call plus completion sync; includes CPU dispatch and any allocation",
            "core_out": "Preallocated output, metadata checks only; values prevalidated",
            "harness": "Public experiment entry, output allocation and synchronous value checks",
            "torch_fixed_order": "Eager GPU decomposition, ordered duplicates; not a native Router",
            "p95": "Nearest-rank sample percentile; graph samples are per-replay averages",
        },
        "results": [],
    }
    print("tokens duplicate implementation graph_device_us eager_wall_us", flush=True)
    for tokens in args.tokens:
        for duplicate in (False, True):
            cpu_case = make_case(tokens, duplicate, args.padding_fraction, args.seed)
            expected = diagnostic_reference(*cpu_case)
            case = tuple(t.cuda() for t in cpu_case)
            for threads in args.threads:
                output = torch.empty((tokens, 256), device="cuda", dtype=torch.float32)

                def core():
                    extension.route_backward_core_out(*case, output, threads)
                    return output

                def harness():
                    return route_backward_core(*case, threads=threads)

                # Validate actual values before bypassing checks for core timing.
                assert_bits_equal(harness(), expected)
                core_result = {
                    "tokens": tokens,
                    "duplicate": duplicate,
                    "active_tokens": int(cpu_case[-1].sum()),
                    "implementation": f"cuda_core_out_{threads}",
                    "byte_equal": True,
                    "graph_device": graph_time(
                        core, expected, args.warmup, args.samples, args.graph_batch
                    ),
                    "eager_wall": wall_time(core, expected, args.warmup, args.samples),
                    "harness_wall": wall_time(harness, expected, args.warmup, args.samples),
                }
                report["results"].append(core_result)
                print(
                    f"{tokens} {duplicate} cuda_core_out_{threads} "
                    f"{core_result['graph_device']['median_us']:.3f} "
                    f"{core_result['eager_wall']['median_us']:.3f}",
                    flush=True,
                )

            def eager():
                return torch_fixed_order(*case)

            assert_bits_equal(eager(), expected)
            torch_result = {
                "tokens": tokens,
                "duplicate": duplicate,
                "active_tokens": int(cpu_case[-1].sum()),
                "implementation": "torch_fixed_order",
                "byte_equal": True,
                "graph_device": graph_time(
                    eager, expected, args.warmup, args.samples, args.graph_batch
                ),
                "eager_wall": wall_time(eager, expected, args.warmup, args.samples),
            }
            report["results"].append(torch_result)
            print(
                f"{tokens} {duplicate} torch_fixed_order "
                f"{torch_result['graph_device']['median_us']:.3f} "
                f"{torch_result['eager_wall']['median_us']:.3f}",
                flush=True,
            )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
