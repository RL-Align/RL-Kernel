# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Time lm_head forward (median latency and peak memory) for one backend on a CUDA/ROCm GPU.

Example (Qwen3-8B dims, SM90 CUDA kernel)::

    python benchmarks/operators/embedding/benchmark_lm_head.py --backend sm90 \
        --tokens 1 128 1024 4096
"""

import argparse
import statistics

import torch


def _make_op(backend: str):
    if backend == "sm90":
        from rl_engine.backends.cuda.embedding.lm_head import SM90LMHeadOp

        return SM90LMHeadOp()

    if backend == "triton":
        from rl_engine.backends.shared.triton.embedding.lm_head import TritonLMHeadOp

        return TritonLMHeadOp()

    from rl_engine.reference.embedding.lm_head import NativeLMHeadOp

    return NativeLMHeadOp()


def _time_forward(op, hidden, weight, warmup: int, iters: int) -> list[float]:
    """Return per-iteration forward latency in milliseconds, measured with CUDA events."""
    with torch.no_grad():
        for _ in range(warmup):
            op.forward(hidden, weight)
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            op.forward(hidden, weight)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["sm90", "triton", "native"], default="sm90")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 128, 1024, 4096])
    parser.add_argument("--vocab", type=int, default=151936)  # Qwen3-8B
    parser.add_argument("--hidden", type=int, default=4096)  # Qwen3-8B
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    op = _make_op(args.backend)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    weight = torch.randn(
        args.vocab, args.hidden, device=device, dtype=torch.bfloat16, generator=generator
    )

    print(f"device={torch.cuda.get_device_name(device)} backend={args.backend}")
    print(f"vocab={args.vocab} hidden={args.hidden} warmup={args.warmup} iters={args.iters}")
    print(f"{'tokens':>6}  {'median_ms':>10}  {'min_ms':>10}  {'max_ms':>10}  {'peak_mem_gb':>11}")
    for tokens in args.tokens:
        hidden = torch.randn(
            tokens, args.hidden, device=device, dtype=torch.bfloat16, generator=generator
        )
        torch.cuda.reset_peak_memory_stats(device)
        times = _time_forward(op, hidden, weight, args.warmup, args.iters)
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        print(
            f"{tokens:6d}  {statistics.median(times):10.3f}  {min(times):10.3f}  "
            f"{max(times):10.3f}  {peak_gb:11.2f}"
        )


if __name__ == "__main__":
    main()
