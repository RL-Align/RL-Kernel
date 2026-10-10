# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark NativeRoPEOp vs TritonRoPEOp vs RoPESM90Op.

RoPE is an elementwise per-position rotation. The native path builds broadcast
cos/sin caches and does the rotate-half in PyTorch; the Triton and CUDA (SM90)
kernels fuse the rotation and round back to the input dtype on store, so they
touch less memory and are faster. Latency (forward and forward+backward) and peak
forward VRAM are reported, swept over (batch, seq) on the Qwen3-8B rotary config
(n_heads=32, head_dim=128, theta=1e6).

Usage:
    python benchmarks/operators/rope/benchmark_rope.py
    python benchmarks/operators/rope/benchmark_rope.py --configs "8,512;16,4096"
"""

import argparse

import torch
from tabulate import tabulate

from rl_engine.backends.shared.triton.rope.rope import TritonRoPEOp
from rl_engine.platforms.device import device_ctx
from rl_engine.reference.rope.rope import NativeRoPEOp
from rl_engine.utils.logger import logger

# Qwen3-8B rotary config.
N_HEADS = 32
HEAD_DIM = 128
THETA = 1.0e6


def _maybe_sm90_op():
    """The Hopper (SM90) RoPE op, or None when unavailable (non-Hopper / not built)."""
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
        and _EXT_AVAILABLE
        and hasattr(_C, "rope_apply_sm90")
    ):
        return None
    from rl_engine.backends.cuda.rope.rope import RoPESM90Op

    return RoPESM90Op()


# (batch, seq)
DEFAULT_CONFIGS = [
    (8, 512),
    (8, 2048),
    (16, 4096),
    (8, 8192),
]


def _make_inputs(batch, seq, device, dtype):
    x = torch.randn(batch, N_HEADS, seq, HEAD_DIM, device=device, dtype=dtype)
    positions = torch.arange(seq, device=device, dtype=torch.long)
    return x, positions


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


def _peak_vram_gb(fn, warmup=3, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / (1024**3)


def run_benchmark(args):
    if device_ctx.device_type not in ["cuda", "xpu", "hip"]:
        raise RuntimeError("rope benchmark requires a compatible GPU device.")

    device = device_ctx.device
    dtype = torch.bfloat16
    native = NativeRoPEOp()
    triton_op = TritonRoPEOp()
    sm90_op = _maybe_sm90_op()

    logger.info(
        f"rope benchmark on {device} (dtype={dtype}); "
        f"SM90 backend {'enabled' if sm90_op is not None else 'unavailable'}"
    )

    rows = []
    for batch, seq in args.configs:
        x, positions = _make_inputs(batch, seq, device, dtype)

        def fwd(op, x=x, pos=positions):
            with torch.no_grad():
                op(x, pos, theta=THETA)

        def fwd_bwd(op, x_src=x, pos=positions):
            x_in = x_src.clone().requires_grad_(True)
            op(x_in, pos, theta=THETA).sum().backward()

        n_fwd = _time_ms(lambda: fwd(native), args.warmup, args.iters)
        t_fwd = _time_ms(lambda: fwd(triton_op), args.warmup, args.iters)
        n_fb = _time_ms(lambda: fwd_bwd(native), args.warmup, args.iters)
        t_fb = _time_ms(lambda: fwd_bwd(triton_op), args.warmup, args.iters)
        n_vram = _peak_vram_gb(lambda: fwd(native))
        t_vram = _peak_vram_gb(lambda: fwd(triton_op))

        row = [
            f"{batch}x{seq}",
            f"{n_fwd:.3f}",
            f"{t_fwd:.3f}",
            f"{n_fwd/t_fwd:.2f}x",
            f"{n_fb:.3f}",
            f"{t_fb:.3f}",
            f"{n_fb/t_fb:.2f}x",
            f"{n_vram*1024:.0f}",
            f"{t_vram*1024:.0f}",
        ]
        if sm90_op is not None:
            s_fwd = _time_ms(lambda: fwd(sm90_op), args.warmup, args.iters)
            s_fb = _time_ms(lambda: fwd_bwd(sm90_op), args.warmup, args.iters)
            s_vram = _peak_vram_gb(lambda: fwd(sm90_op))
            row += [
                f"{s_fwd:.3f}",
                f"{n_fwd/s_fwd:.2f}x",
                f"{t_fwd/s_fwd:.2f}x",
                f"{s_fb:.3f}",
                f"{s_vram*1024:.0f}",
            ]
        rows.append(row)

    headers = [
        "shape (B x S)",
        "native fwd ms",
        "triton fwd ms",
        "fwd speedup",
        "native f+b ms",
        "triton f+b ms",
        "f+b speedup",
        "native fwd MB",
        "triton fwd MB",
    ]
    if sm90_op is not None:
        headers += [
            "sm90 fwd ms",
            "sm90 vs native",
            "sm90 vs triton",
            "sm90 f+b ms",
            "sm90 fwd MB",
        ]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Semicolon-separated 'batch,seq' tuples, e.g. '8,512;16,4096'.",
    )
    args = parser.parse_args()
    if args.configs:
        args.configs = [tuple(int(x) for x in tup.split(",")) for tup in args.configs.split(";")]
    else:
        args.configs = DEFAULT_CONFIGS
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
