# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark for the Qwen-Image attn-out bias GEMM (contract
``attn-out-bias-gemm-tree-v1``).

Compares the FP32 tree reference against the Triton per-tile tree backend and
the CUDA per-thread-tree backend (forward and forward+backward) across the
selected token counts. The reference runs on the same GPU. The benchmark
reports synchronized elapsed time, peak allocated memory above live inputs,
and speedups between the available backends.
"""

import argparse
import gc
import time
import warnings

import torch

from rl_engine.reference.gemm.attn_out_bias_gemm import NativeAttnOutBiasGemmOp


def _maybe_triton_op():
    try:
        from rl_engine.backends.shared.triton.gemm.attn_out_bias_gemm import TritonAttnOutBiasGemmOp

        return TritonAttnOutBiasGemmOp()
    except Exception as exc:
        warnings.warn(
            f"Skipping Triton backend: {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _maybe_cuda_op():
    try:
        from rl_engine.backends.cuda.gemm.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

        return CudaAttnOutBiasGemmOp()
    except Exception as exc:
        warnings.warn(
            f"Skipping CUDA backend: {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _measure(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / iters * 1000.0

    # Measure memory separately so collection and synchronization do not affect latency.
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak_mib = (torch.cuda.max_memory_allocated() - baseline) / 2**20
    return elapsed_ms, peak_mib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=3072, help="K = N (Qwen-Image inner dim)")
    parser.add_argument(
        "--rows",
        type=str,
        default="512,4096,6032,6889",
        help="comma-separated row counts (6032 covers the 1664 x 928 image tier)",
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--backward", action="store_true", help="time forward+backward")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--skip-reference", action="store_true", help="time only Triton and CUDA")
    args = parser.parse_args()
    if args.iters < 1 or args.warmup < 0:
        parser.error("--iters must be positive and --warmup must be nonnegative")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device("cuda")

    reference = None if args.skip_reference else NativeAttnOutBiasGemmOp()
    triton_op = _maybe_triton_op()
    # The constructor validates all forward/backward symbols together.
    cuda_op = _maybe_cuda_op()

    backends = {"ref": reference, "triton": triton_op, "cuda": cuda_op}
    if not any(op is not None for op in backends.values()):
        parser.error("no requested backend is available")

    def closures(op, x, w, b):
        if args.backward:
            xd = x.clone().requires_grad_(True)
            wd = w.clone().requires_grad_(True)
            bd = b.clone().requires_grad_(True)
            dy = x.new_ones((x.shape[0], w.shape[0]))

            def fwd_bwd():
                out = op(xd, wd, bias=bd)
                out.backward(dy)
                xd.grad = None
                wd.grad = None
                bd.grad = None

            return fwd_bwd
        return lambda: op(x, w, bias=b)

    print(
        f"attn_out_bias_gemm  dim={args.dim}  dtype={args.dtype}  "
        f"mode={'fwd+bwd' if args.backward else 'fwd'}"
    )
    for rows in [int(v) for v in args.rows.split(",")]:
        gen = torch.Generator().manual_seed(rows)
        x = torch.randn(rows, args.dim, generator=gen).to(dtype).to(device)
        w = torch.randn(args.dim, args.dim, generator=gen).to(dtype).to(device)
        b = torch.randn(args.dim, generator=gen).to(dtype).to(device)

        timings = {}
        fields = [f"S={rows:>5}"]
        for name, op in backends.items():
            if op is None:
                continue
            ms, peak_mib = _measure(closures(op, x, w, b), args.warmup, args.iters)
            timings[name] = ms
            field = f"{name} {ms:.3f} ms, peak {peak_mib:.2f} MiB"
            if name != "ref" and "ref" in timings:
                field += f", speedup vs ref {timings['ref'] / ms:.2f}x"
            fields.append(field)
        if "triton" in timings and "cuda" in timings:
            fields.append(f"Triton speedup vs CUDA {timings['cuda'] / timings['triton']:.2f}x")
        print(" | ".join(fields), flush=True)


if __name__ == "__main__":
    main()
