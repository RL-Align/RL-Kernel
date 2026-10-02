# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark for the Qwen-Image attn-out bias GEMM (contract
``attn-out-bias-gemm-tree-v1``).

Compares the FP32 tree reference against the Triton leaf-chunk backend and
the CUDA per-thread-tree backend (forward and forward+backward) across the
issue #386 acceptance tiers. All backends share the frozen reduction tree;
see docs/operators/attn-out-bias-gemm.md.
"""

import argparse
import time

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.linear.attn_out_bias_gemm import NativeAttnOutBiasGemmOp


def _maybe_triton_op():
    try:
        from rl_engine.kernels.ops.triton.linear.attn_out_bias_gemm import TritonAttnOutBiasGemmOp

        return TritonAttnOutBiasGemmOp()
    except Exception:
        return None


def _maybe_cuda_op():
    try:
        from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

        return CudaAttnOutBiasGemmOp()
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
    parser.add_argument("--dim", type=int, default=3072, help="K = N (Qwen-Image inner dim)")
    parser.add_argument(
        "--rows",
        type=str,
        default="512,4096,6889",
        help="comma-separated token tiers (issue acceptance: 4096/6889 for 1024^2/1328^2)",
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--backward", action="store_true", help="time forward+backward")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device("cuda")

    reference = NativeAttnOutBiasGemmOp()
    triton_op = _maybe_triton_op()
    cuda_op = (
        _maybe_cuda_op() if (_EXT_AVAILABLE and hasattr(_C, "attn_out_tree_gemm_cuda")) else None
    )

    def closures(op, x, w, b):
        if args.backward:
            xd = x.clone().requires_grad_(True)
            wd = w.clone().requires_grad_(True)
            bd = b.clone().requires_grad_(True)

            def fwd_bwd():
                out = op(xd, wd, bias=bd)
                out.backward(torch.ones_like(out))
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

        ref_ms = _bench_ms(closures(reference, x, w, b), args.warmup, args.iters)
        line = f"S={rows:>5}  ref {ref_ms:8.2f} ms"
        if triton_op is not None:
            t = _bench_ms(closures(triton_op, x, w, b), args.warmup, args.iters)
            line += f"  | triton {t:8.2f} ms  speedup {ref_ms / t:6.1f}x"
        if cuda_op is not None:
            c = _bench_ms(closures(cuda_op, x, w, b), args.warmup, args.iters)
            line += f"  | cuda {c:8.2f} ms  speedup {ref_ms / c:6.1f}x"
        print(line, flush=True)


if __name__ == "__main__":
    main()
