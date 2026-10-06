# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark for the Qwen-Image txt_in RMSNorm->Linear op (contract
``txt-in-rmsnorm-linear-v1``).

Compares the FP32 tree reference against the Triton leaf-chunk backend and
the CUDA per-thread-tree backend (forward and forward+backward) across the
issue #386 acceptance tiers. Hidden/out dims are contract-frozen (3584/3072);
see docs/operators/txt-in-rmsnorm-linear.md.
"""

import argparse
import time

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.norm.txt_in_rmsnorm_linear import (
    TXT_IN_HIDDEN,
    TXT_IN_OUT,
    NativeTxtInRMSNormLinearOp,
)


def _maybe_triton_op():
    try:
        from rl_engine.kernels.ops.triton.norm.txt_in_rmsnorm_linear import (
            TritonTxtInRMSNormLinearOp,
        )

        return TritonTxtInRMSNormLinearOp()
    except Exception:
        return None


def _maybe_cuda_op():
    try:
        from rl_engine.kernels.ops.cuda.norm.txt_in_rmsnorm_linear import CudaTxtInRMSNormLinearOp

        return CudaTxtInRMSNormLinearOp()
    except Exception:
        return None


def _bench_ms(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=str,
        default="512,1024,4096",
        help="comma-separated token tiers (single-row text patches in WS1; 4096 stress tier)",
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--backward", action="store_true", help="time forward+backward")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device("cuda")

    reference = NativeTxtInRMSNormLinearOp()
    triton_op = _maybe_triton_op()
    cuda_op = (
        _maybe_cuda_op() if (_EXT_AVAILABLE and hasattr(_C, "txt_in_norm_stats_cuda")) else None
    )

    def closures(op, x, gamma, w, b):
        if args.backward:
            xd = x.clone().requires_grad_(True)
            gd = gamma.clone().requires_grad_(True)
            wd = w.clone().requires_grad_(True)
            bd = b.clone().requires_grad_(True)

            def fwd_bwd():
                out = op(xd, gd, wd, bias=bd)
                out.backward(torch.ones_like(out))
                xd.grad = None
                gd.grad = None
                wd.grad = None
                bd.grad = None

            return fwd_bwd
        return lambda: op(x, gamma, w, bias=b)

    print(
        f"txt_in_rmsnorm_linear  H={TXT_IN_HIDDEN} O={TXT_IN_OUT}  dtype={args.dtype}  "
        f"mode={'fwd+bwd' if args.backward else 'fwd'}"
    )
    for rows in [int(v) for v in args.rows.split(",")]:
        gen = torch.Generator().manual_seed(rows)
        x = torch.randn(rows, TXT_IN_HIDDEN, generator=gen).to(dtype).to(device)
        gamma = torch.randn(TXT_IN_HIDDEN, generator=gen).to(dtype).to(device)
        w = torch.randn(TXT_IN_OUT, TXT_IN_HIDDEN, generator=gen).to(dtype).to(device)
        b = torch.randn(TXT_IN_OUT, generator=gen).to(dtype).to(device)

        ref_ms = _bench_ms(closures(reference, x, gamma, w, b), args.warmup, args.iters)
        line = f"S={rows:>5}  ref {ref_ms:8.2f} ms"
        if triton_op is not None:
            t = _bench_ms(closures(triton_op, x, gamma, w, b), args.warmup, args.iters)
            line += f"  | triton {t:8.2f} ms  speedup {ref_ms / t:6.1f}x"
        if cuda_op is not None:
            c = _bench_ms(closures(cuda_op, x, gamma, w, b), args.warmup, args.iters)
            line += f"  | cuda {c:8.2f} ms  speedup {ref_ms / c:6.1f}x"
        print(line, flush=True)


if __name__ == "__main__":
    main()
