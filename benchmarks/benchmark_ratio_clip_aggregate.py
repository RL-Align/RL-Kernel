# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark fused ratio clipping and aggregation against the PyTorch reference.

Usage:
    python benchmarks/benchmark_ratio_clip_aggregate.py
    python benchmarks/benchmark_ratio_clip_aggregate.py \
        --configs "32,256;128,1024;256,4096"
"""

import argparse

import torch
from tabulate import tabulate

from rl_engine.kernels.ops.pytorch.loss.ratio_clip_aggregate import NativeRatioClipAggregateOp
from rl_engine.kernels.ops.triton.loss.ratio_clip_aggregate import TritonRatioClipAggregateOp

DEFAULT_CONFIGS = [
    (32, 256),
    (128, 1024),
    (256, 4096),
    (256, 16384),
]


def _make_inputs(batch, tokens, *, density, device):
    generator = torch.Generator(device=device).manual_seed(batch * 100000 + tokens)
    ratio = torch.exp(torch.randn(batch, tokens, generator=generator, device=device) * 0.3)
    advantages = torch.randn(batch, generator=generator, device=device)
    mask = torch.rand(batch, tokens, generator=generator, device=device) < density
    penalty = torch.rand(batch, tokens, generator=generator, device=device) * 0.1
    return ratio, advantages, mask, penalty


def _time_ms(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def _peak_vram_mb(fn, warmup=3, iterations=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / (1024**2)


def run_benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("ratio-clip-aggregate benchmark requires a CUDA GPU.")

    native = NativeRatioClipAggregateOp()
    fused = TritonRatioClipAggregateOp()
    kwargs = dict(
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        penalty_coef=args.penalty_coef,
    )
    rows = []

    for batch, tokens in args.configs:
        ratio, advantages, mask, penalty = _make_inputs(
            batch,
            tokens,
            density=args.mask_density,
            device="cuda",
        )

        def native_forward(
            ratio=ratio,
            advantages=advantages,
            mask=mask,
            penalty=penalty,
        ):
            with torch.no_grad():
                native(
                    ratio,
                    advantages,
                    mask,
                    penalty_terms=penalty,
                    **kwargs,
                )

        def fused_forward(
            ratio=ratio,
            advantages=advantages,
            mask=mask,
            penalty=penalty,
        ):
            with torch.no_grad():
                fused(
                    ratio,
                    advantages,
                    mask,
                    penalty_terms=penalty,
                    **kwargs,
                )

        ratio_grad = ratio.clone().requires_grad_(True)
        penalty_grad = penalty.clone().requires_grad_(True)

        def native_forward_backward(
            ratio_grad=ratio_grad,
            advantages=advantages,
            mask=mask,
            penalty_grad=penalty_grad,
        ):
            loss = native(
                ratio_grad,
                advantages,
                mask,
                penalty_terms=penalty_grad,
                **kwargs,
            )[0]
            torch.autograd.grad(loss, (ratio_grad, penalty_grad))

        def fused_forward_backward(
            ratio_grad=ratio_grad,
            advantages=advantages,
            mask=mask,
            penalty_grad=penalty_grad,
        ):
            loss = fused(
                ratio_grad,
                advantages,
                mask,
                penalty_terms=penalty_grad,
                **kwargs,
            )[0]
            torch.autograd.grad(loss, (ratio_grad, penalty_grad))

        native_forward_ms = _time_ms(native_forward, args.warmup, args.iterations)
        fused_forward_ms = _time_ms(fused_forward, args.warmup, args.iterations)
        native_fb_ms = _time_ms(native_forward_backward, args.warmup, args.iterations)
        fused_fb_ms = _time_ms(fused_forward_backward, args.warmup, args.iterations)
        native_vram = _peak_vram_mb(native_forward)
        fused_vram = _peak_vram_mb(fused_forward)
        rows.append(
            [
                f"{batch}x{tokens}",
                f"{batch * tokens / 1e6:.2f}M",
                f"{native_forward_ms:.3f}",
                f"{fused_forward_ms:.3f}",
                f"{native_forward_ms / fused_forward_ms:.2f}x",
                f"{native_fb_ms:.3f}",
                f"{fused_fb_ms:.3f}",
                f"{native_fb_ms / fused_fb_ms:.2f}x",
                f"{native_vram:.1f}",
                f"{fused_vram:.1f}",
                f"{native_vram / max(fused_vram, 1e-9):.2f}x",
            ]
        )

    print(
        tabulate(
            rows,
            headers=[
                "shape (B x T)",
                "tokens",
                "PyTorch fwd ms",
                "Triton fwd ms",
                "fwd speedup",
                "PyTorch f+b ms",
                "Triton f+b ms",
                "f+b speedup",
                "PyTorch MB",
                "Triton MB",
                "memory reduction",
            ],
            tablefmt="github",
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--clip-low", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=0.2)
    parser.add_argument("--penalty-coef", type=float, default=0.04)
    parser.add_argument("--mask-density", type=float, default=0.9)
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Semicolon-separated batch,tokens pairs.",
    )
    args = parser.parse_args()
    if args.configs:
        args.configs = [
            tuple(int(value) for value in config.split(",")) for config in args.configs.split(";")
        ]
    else:
        args.configs = DEFAULT_CONFIGS
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
