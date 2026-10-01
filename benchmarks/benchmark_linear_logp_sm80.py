# SPDX-License-Identifier: Apache-2.0
"""A100 Stage-5 production-dispatch benchmark (5 warmups, 20 iterations)."""

from __future__ import annotations

import argparse
import csv
import math

import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.cuda.loss.linear_logp_sm80 import FusedLinearLogpSM80Op
from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

D, V = 4096, 128256


def time_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def measure(fn):
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    return out, peak / 2**20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="stage5_benchmark.csv")
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (8, 0)
    torch.manual_seed(1234)
    weight = torch.randn(V, D, device="cuda", dtype=torch.bfloat16)
    triton = TritonLinearLogpOp()
    production = FusedLinearLogpSM80Op()
    rows = []
    for n in (128, 256, 512, 1024, 2048, 4096):
        hidden = torch.randn(n, D, device="cuda", dtype=torch.bfloat16) / math.sqrt(D)
        target = torch.randint(V, (n,), device="cuda")
        oracle = F.linear(hidden.float(), weight.float()).log_softmax(-1).gather(
            -1, target[:, None]
        )[:, 0]
        funcs = {
            "cublas_materialized": lambda: F.log_softmax(F.linear(hidden, weight).float(), -1).gather(-1, target[:, None])[:, 0],
            "triton": lambda: triton(hidden, weight, target),
            "production": lambda: production(hidden, weight, target),
        }
        backend = production.selected_backend(hidden, weight, target)
        for name, fn in funcs.items():
            out, peak_mb = measure(fn)
            latency = time_ms(fn)
            err = (out - oracle).abs()
            rows.append(
                dict(
                    n=n,
                    implementation=name,
                    selected_backend=backend if name == "production" else name,
                    latency_ms=latency,
                    tokens_per_s=n * 1000.0 / latency,
                    peak_activation_mb=peak_mb,
                    max_abs_error=err.max().item(),
                    mean_abs_error=err.mean().item(),
                    finite=bool(torch.isfinite(out).all()),
                )
            )
            print(rows[-1], flush=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
