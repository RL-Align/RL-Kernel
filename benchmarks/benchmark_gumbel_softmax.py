# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark NativeGumbelSoftmaxOp vs TritonGumbelSoftmaxOp.

Usage:
    python benchmarks/benchmark_gumbel_softmax.py
    python benchmarks/benchmark_gumbel_softmax.py --configs "1,512,32000;4,512,50257"
"""

import argparse

import torch
from tabulate import tabulate

from rl_engine.kernels.ops.pytorch.sampling.gumbel_softmax import NativeGumbelSoftmaxOp
from rl_engine.kernels.ops.triton.sampling.gumbel_softmax import TritonGumbelSoftmaxOp
from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger

DEFAULT_CONFIGS = [
    (1, 512, 32000),
    (4, 512, 32000),
    (4, 1024, 50257),
]


def _make_inputs(batch, seq, vocab, device, dtype, internal_gumbels):
    logits = torch.randn(batch, seq, vocab, device=device, dtype=dtype)
    gumbels = None
    if not internal_gumbels:
        gumbels = (
            -torch.empty(batch, seq, vocab, device=device, dtype=torch.float32).exponential_().log()
        )
    return logits, gumbels


def _time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _peak_vram_mb(fn, warmup=3, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / (1024**2)


def run_benchmark(args):
    if device_ctx.device_type not in ["cuda", "xpu", "hip"]:
        raise RuntimeError("gumbel_softmax benchmark requires a compatible GPU device.")

    device = device_ctx.device
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    native = NativeGumbelSoftmaxOp()
    triton_op = TritonGumbelSoftmaxOp()

    logger.info(
        f"gumbel_softmax benchmark on {device} (dtype={dtype}, hard={args.hard}, "
        f"tau={args.tau}, internal_gumbels={args.internal_gumbels})"
    )

    rows = []
    for batch, seq, vocab in args.configs:
        logits, gumbels = _make_inputs(batch, seq, vocab, device, dtype, args.internal_gumbels)
        upstream = torch.randn_like(logits)
        native_leaf = logits.detach().clone().requires_grad_(True)
        triton_leaf = logits.detach().clone().requires_grad_(True)

        def fwd(op, x=logits, g=gumbels):
            with torch.no_grad():
                op(x, tau=args.tau, hard=args.hard, gumbels=g)

        def fwd_bwd(op, x, g=gumbels, grad=upstream):
            x.grad = None
            op(x, tau=args.tau, hard=args.hard, gumbels=g).backward(grad)

        n_fwd = _time_ms(lambda: fwd(native), args.warmup, args.iters)
        t_fwd = _time_ms(lambda: fwd(triton_op), args.warmup, args.iters)
        n_fb = _time_ms(lambda: fwd_bwd(native, native_leaf), args.warmup, args.iters)
        t_fb = _time_ms(lambda: fwd_bwd(triton_op, triton_leaf), args.warmup, args.iters)
        n_vram = _peak_vram_mb(lambda: fwd(native))
        t_vram = _peak_vram_mb(lambda: fwd(triton_op))

        rows.append(
            [
                f"{batch}x{seq}x{vocab}",
                str(dtype).replace("torch.", ""),
                f"{n_fwd:.3f}",
                f"{t_fwd:.3f}",
                f"{n_fwd / t_fwd:.2f}x",
                f"{n_fb:.3f}",
                f"{t_fb:.3f}",
                f"{n_fb / t_fb:.2f}x",
                f"{n_vram:.0f}",
                f"{t_vram:.0f}",
            ]
        )

    headers = [
        "shape (B x S x V)",
        "dtype",
        "native fwd ms",
        "triton fwd ms",
        "fwd speedup",
        "native f+b ms",
        "triton f+b ms",
        "f+b speedup",
        "native fwd MB",
        "triton fwd MB",
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--internal-gumbels",
        action="store_true",
        help="Let each backend sample Gumbel noise internally. By default the benchmark "
        "passes a materialized Gumbel tensor for deterministic apples-to-apples timing.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Semicolon-separated 'batch,seq,vocab' tuples, e.g. '1,512,32000;4,512,50257'.",
    )
    args = parser.parse_args()
    if args.configs:
        args.configs = [tuple(int(x) for x in tup.split(",")) for tup in args.configs.split(";")]
    else:
        args.configs = DEFAULT_CONFIGS
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
