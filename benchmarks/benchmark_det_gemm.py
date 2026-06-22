# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Overhead of batch-invariant det_gemm vs cuBLAS + Triton (WS1 #146).

det_gemm (CUDA, naive first milestone) and the Triton path are batch-invariant
and SLOWER than cuBLAS by design (no split-K/stream-K, fixed accumulation, FP32,
no TF32). Reports overhead vs the fair baseline (cuBLAS, TF32 disabled), not a
speedup. The naive CUDA kernel is correctness-first; a tensor-core pass follows.
"""
import argparse
import torch

from rl_engine.kernels.ops.cuda.matmul import deterministic_gemm
from rl_engine.kernels.ops.pytorch.matmul import native_gemm

try:
    from rl_engine.kernels.ops.triton.matmul import deterministic_gemm_triton
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

DEV = "cuda"
WARMUP, ITERS = 10, 50

SHAPES = [
    ("qkv",     4096, 4096, 12288),
    ("o_proj",  4096, 4096, 4096),
    ("mlp_up",  4096, 4096, 14336),
    ("mlp_dn",  4096, 14336, 4096),
    ("lm_head", 4096, 4096, 32000),
]


def _time(fn, a, b):
    for _ in range(WARMUP):
        fn(a, b)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn(a, b)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def run():
    rows = []
    for name, M, K, N in SHAPES:
        a = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
        b = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
        torch.backends.cuda.matmul.allow_tf32 = True
        t_tf32 = _time(lambda x, y: torch.matmul(x, y), a, b)
        torch.backends.cuda.matmul.allow_tf32 = False
        t_fp32 = _time(native_gemm, a, b)
        t_cuda = _time(deterministic_gemm, a, b)
        t_tri = _time(deterministic_gemm_triton, a, b) if _HAS_TRITON else float("nan")
        rows.append((name, M, K, N, t_tf32, t_fp32, t_cuda, t_tri, t_cuda / t_fp32))
    return rows


def to_markdown(rows, dev, cap):
    out = [f"## det_gemm overhead — {dev} (SM{cap[0]}{cap[1]})", "",
           "| shape | M | K | N | cuBLAS tf32 | cuBLAS fp32 | det CUDA | det Triton | overhead |",
           "|---|---|---|---|---|---|---|---|---|"]
    for n, M, K, N, t1, t2, t3, t4, ov in rows:
        out.append(f"| {n} | {M} | {K} | {N} | {t1:.3f} | {t2:.3f} | {t3:.3f} | {t4:.3f} | {ov:.1f}x |")
    out += ["",
            "_Overhead = det CUDA vs cuBLAS (TF32 disabled). Naive CUDA kernel is "
            "correctness-first; both det paths trade speed for bitwise "
            "batch-invariance. Tensor-core pass is a follow-up (#146)._"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    name, cap = torch.cuda.get_device_name(), torch.cuda.get_device_capability()
    print(name, cap)
    md = to_markdown(run(), name, cap)
    print("\n" + md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()