#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark P1-2 fp32_gemm_rms: torch-native vs strict Triton vs strict CUDA.

torch-native is the unconstrained reference (cuBLAS matmul + fused torch RMS,
free to use FMA, split-K and library reduction trees): a different arithmetic
contract, reported for context, never an acceptance gate for the strict
backends. Timings are CUDA-event medians over the full autograd entry
(forward, and forward + backward with fixed upstream gradients).

Example:
    python benchmarks/benchmark_fp32_gemm_rms.py --tokens 1,16,128,512 \
        --k 16384 --warmup 5 --iterations 20 --json out.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from typing import Any, Callable

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.mhc.fp32_gemm_rms import fp32_gemm_rms  # noqa: E402

N_DIM = 24
EPS = 1e-6


def torch_native(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p = x @ w.t()
    r = 1.0 / (torch.sqrt(x.square().mean(dim=1)) + EPS)
    return p, r


def _runner(backend: str) -> Callable[[torch.Tensor, torch.Tensor], tuple]:
    if backend == "torch-native":
        return torch_native
    return lambda x, w: fp32_gemm_rms(x, w, EPS, backend=backend)


def _time_ms(fn: Callable[[], None], warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples": samples,
    }


def bench_case(backend: str, tokens: int, k: int, warmup: int, iterations: int) -> dict[str, Any]:
    gen = torch.Generator(device="cpu").manual_seed(tokens)
    x0 = torch.randn(tokens, k, generator=gen, dtype=torch.float32).cuda()
    w0 = torch.randn(N_DIM, k, generator=gen, dtype=torch.float32).cuda()
    dp = torch.randn(tokens, N_DIM, generator=gen, dtype=torch.float32).cuda()
    dr = torch.randn(tokens, generator=gen, dtype=torch.float32).cuda()
    run = _runner(backend)

    def forward_only() -> None:
        with torch.no_grad():
            run(x0, w0)

    def forward_backward() -> None:
        x = x0.detach().requires_grad_(True)
        w = w0.detach().requires_grad_(True)
        p, r = run(x, w)
        torch.autograd.backward([p, r], [dp, dr])

    return {
        "forward": _time_ms(forward_only, warmup, iterations),
        "forward_backward": _time_ms(forward_backward, warmup, iterations),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tokens", default="1,16,128,512")
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--backends", default="torch-native,triton,cuda")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark", file=sys.stderr)
        return 1

    token_list = [int(v) for v in args.tokens.split(",")]
    backends = args.backends.split(",")
    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "k": args.k,
        "n": N_DIM,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": {},
    }

    header = f"{'T':>6} {'mode':<18}" + "".join(f"{b:>16}" for b in backends)
    print(header)
    print("-" * len(header))
    for tokens in token_list:
        rows: dict[str, Any] = {}
        for backend in backends:
            try:
                rows[backend] = bench_case(backend, tokens, args.k, args.warmup, args.iterations)
            except (RuntimeError, ValueError, TypeError) as exc:
                rows[backend] = {"error": str(exc)}
        report["results"][str(tokens)] = rows
        for mode in ("forward", "forward_backward"):
            cells = []
            for backend in backends:
                entry = rows[backend]
                if "error" in entry:
                    cells.append(f"{'ERROR':>16}")
                else:
                    cells.append(f"{entry[mode]['median_ms']:>15.6f} ")
            print(f"{tokens:>6} {mode:<18}" + "".join(cells))

    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nreport written to {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
