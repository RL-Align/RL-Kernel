# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark the Qwen-Image joint attention QK GEMM (issue #386).

Compares the CUDA kernel against the fixed-order PyTorch reference at the joint
sequence lengths implied by the three reference resolutions. The reference is an
explicit elementwise tree, so it is orders of magnitude slower than the kernel
and is skipped above ``--reference-limit`` tokens by default.

    python benchmarks/operators/attention/benchmark_joint_attn_qk_gemm.py
    python benchmarks/operators/attention/benchmark_joint_attn_qk_gemm.py --backward
    python benchmarks/operators/attention/benchmark_joint_attn_qk_gemm.py \
        --seqs 512,4160 --heads 24 --dtype bf16
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Joint sequence lengths: 64 padded text tokens prepended to the image tokens of
# the 1024², 1664x928 and 1328² reference resolutions.
DEFAULT_SEQS = (512, 4160, 6096, 6953)
HEAD_DIM = 128
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seqs",
        default=",".join(str(value) for value in DEFAULT_SEQS),
        help="Comma-separated joint sequence lengths.",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24, help="Qwen-Image uses 24 heads.")
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--backward", action="store_true", help="Benchmark dQ/dK as well.")
    parser.add_argument(
        "--reference-limit",
        type=int,
        default=1024,
        help="Skip the PyTorch reference above this joint sequence length.",
    )
    return parser.parse_args()


def timed(fn: Callable[[], object], warmup: int, iters: int) -> tuple[float, float]:
    """Median wall time in milliseconds and peak CUDA memory in GiB."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return samples[len(samples) // 2], peak


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("No CUDA device available.")
        return 1

    from rl_engine.backends.cuda.attention.joint_attn_qk_gemm import JointAttnQkGemmCudaOp
    from rl_engine.reference.attention.joint_attn_qk_gemm import NativeJointAttnQkGemmOp

    try:
        cuda_op: Optional[JointAttnQkGemmCudaOp] = JointAttnQkGemmCudaOp()
    except RuntimeError as exc:
        print(f"CUDA backend unavailable: {exc}")
        return 1
    reference = NativeJointAttnQkGemmOp()

    dtype = DTYPES[args.dtype]
    scale = 1.0 / math.sqrt(args.head_dim)
    seqs = [int(value) for value in args.seqs.split(",") if value]

    print(f"device       : {torch.cuda.get_device_name(0)}")
    print(f"torch / cuda : {torch.__version__} / {torch.version.cuda}")
    print(
        f"config       : batch={args.batch} heads={args.heads} head_dim={args.head_dim} "
        f"dtype={args.dtype} iters={args.iters}"
    )
    print(f"contract     : {cuda_op.contract_version}")
    header = f"{'joint S':>9} {'cuda fwd ms':>12} {'peak GiB':>9} {'ref fwd ms':>12} {'speedup':>8}"
    if args.backward:
        header += f" {'cuda bwd ms':>12} {'bwd peak GiB':>13}"
    print(header)

    for seq in seqs:
        shape = (args.batch, args.heads, seq, args.head_dim)
        q = torch.randn(*shape, dtype=dtype, device="cuda")
        k = torch.randn(*shape, dtype=dtype, device="cuda")

        fwd_ms, fwd_peak = timed(
            lambda: cuda_op.forward(q, k, scale=scale), args.warmup, args.iters
        )

        ref_text, speedup_text = "skipped", "-"
        if seq <= args.reference_limit:
            ref_ms, _ = timed(lambda: reference.forward(q, k, scale=scale), 1, 1)
            ref_text = f"{ref_ms:.2f}"
            speedup_text = f"{ref_ms / fwd_ms:.1f}x"

        row = f"{seq:>9} {fwd_ms:>12.3f} {fwd_peak:>9.2f} {ref_text:>12} {speedup_text:>8}"

        if args.backward:
            qg = q.clone().requires_grad_()
            kg = k.clone().requires_grad_()
            grad = torch.randn(args.batch, args.heads, seq, seq, dtype=dtype, device="cuda")

            def run_backward() -> None:
                qg.grad = None
                kg.grad = None
                cuda_op.forward(qg, kg, scale=scale).backward(grad)

            bwd_ms, bwd_peak = timed(run_backward, 1, max(1, args.iters // 2))
            row += f" {bwd_ms:>12.1f} {bwd_peak:>13.2f}"
            qg = kg = grad = None  # release before the next shape

        print(row)
        q = k = None  # release before the next shape
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
