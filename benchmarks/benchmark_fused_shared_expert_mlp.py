#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Fused shared-expert fc1 + SwiGLU vs the unfused det_gemm composition.

Both paths are byte-identical (tests/test_fused_shared_expert_mlp.py asserts
it), so this measures only what fusing buys. The BF16 cuBLAS column is the
non-deterministic speed ceiling, not an alternative: it is neither
batch-invariant nor TP-equivalent.

    python benchmarks/benchmark_fused_shared_expert_mlp.py [--tokens 512,2048,8192]
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def time_ms(fn, warmup: int = 10, iters: int = 50, repeats: int = 5) -> float:
    """Median of `repeats` timed runs -- the first samples are unstable."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return statistics.median(samples)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--ffn", type=int, default=2048)
    p.add_argument("--tokens", default="128,512,2048,8192,32768")
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("CUDA device required")
        return 1
    try:
        from rl_engine import _C
    except ImportError as exc:
        print(f"[skip] rl_engine._C is not built: {exc}")
        return 1
    for sym in ("fused_shared_expert_fc1_swiglu", "det_gemm_fwd_rhs_transposed"):
        if not hasattr(_C, sym):
            print(f"[skip] rl_engine._C lacks {sym}; rebuild the extension")
            return 1

    H, F = args.hidden, args.ffn
    print(f"H={H} F={F} ({torch.cuda.get_device_name(0)})")
    print(f"{'tokens':>7} {'unfused':>10} {'fused':>9} {'speedup':>8} {'bf16 cuBLAS':>13}")
    for T in [int(v) for v in args.tokens.split(",")]:
        gen = torch.Generator().manual_seed(1)
        x = (torch.randn(T, H, generator=gen) * 0.7).to(torch.bfloat16).cuda()
        w1 = (torch.randn(2 * F, H, generator=gen) / H**0.5).to(torch.bfloat16).cuda()

        def unfused():
            z = _C.det_gemm_fwd_rhs_transposed(x, w1).float()
            gate, up = z[:, :F], z[:, F:]
            return ((gate * torch.sigmoid(gate)) * up).to(torch.bfloat16)

        def fused():
            return _C.fused_shared_expert_fc1_swiglu(x, w1)

        def cublas():
            z = x @ w1.t()
            return (torch.nn.functional.silu(z[:, :F]) * z[:, F:]).to(torch.bfloat16)

        t_un, t_fu, t_cb = time_ms(unfused), time_ms(fused), time_ms(cublas)
        print(f"{T:>7} {t_un:10.3f} {t_fu:9.3f} {t_un / t_fu:7.2f}x {t_cb:13.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
