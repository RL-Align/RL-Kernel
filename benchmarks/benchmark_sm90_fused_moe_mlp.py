#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""SM90 fused routed-expert MLP benchmark: two-launch and fused paths vs a
BF16 cuBLAS loop over experts (non-deterministic speed ceiling).

    python benchmarks/benchmark_sm90_fused_moe_mlp.py [--rows 2048,8192,32768]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.moe.contract import ExpertBatch  # noqa: E402
from rl_engine.moe.mx_format import mx_dequantize, mx_quantize  # noqa: E402


def time_ms(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--ffn", type=int, default=2048)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--rows", default="2048,8192,32768")
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("CUDA device required")
        return 1
    from rl_engine.moe.backends.sm90_fused_mlp import PROFILE, Sm90FusedMoeMlp

    try:
        be = Sm90FusedMoeMlp()
    except NotImplementedError as exc:
        print(f"[skip] {exc}")
        return 1
    H, F, E = args.hidden, args.ffn, args.experts
    g = torch.Generator().manual_seed(2026)
    print(f"H={H} F={F} E={E} ({torch.cuda.get_device_name(0)})")
    print(f"{'rows':>7} {'two-launch ms':>14} {'fused ms':>10} {'bf16 loop ms':>13} {'two-launch TFLOPS':>18}")
    for M in [int(v) for v in args.rows.split(",")]:
        x = (torch.randn(M, H, generator=g) * 0.5).bfloat16().cuda()
        w1 = (torch.randn(E, 2 * F, H, generator=g) / H**0.5).bfloat16().cuda()
        w2 = (torch.randn(E, H, F, generator=g) / F**0.5).bfloat16().cuda()
        p_s = torch.rand(M, generator=g).cuda()
        offsets = torch.tensor([0] + [M * (i + 1) // E for i in range(E)], dtype=torch.int32, device="cuda")
        batch = ExpertBatch(x=x, expert_offsets=offsets, p_s=p_s, w1=mx_quantize(w1, "e2m1"),
                            w2=mx_quantize(w2, "e2m1"), lora=None,
                            output_slot=torch.arange(M, dtype=torch.int32, device="cuda"),
                            numeric_profile=PROFILE)
        x_q = mx_quantize(x, "e4m3")
        xd, w1d, w2d = mx_dequantize(x_q).bfloat16(), mx_dequantize(batch.w1).bfloat16(), mx_dequantize(batch.w2).bfloat16()
        offs = offsets.tolist()

        def bf16_loop():
            y = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
            for e in range(E):
                lo, hi = offs[e], offs[e + 1]
                z = xd[lo:hi] @ w1d[e].t()
                h = (torch.nn.functional.silu(z[:, :F]) * z[:, F:]) * p_s[lo:hi, None]
                y[lo:hi] = h.bfloat16() @ w2d[e].t()
            return y

        t2 = time_ms(lambda: be.forward(batch, x_q, "two_launch"))
        tf = time_ms(lambda: be.forward(batch, x_q, "fused"))
        tb = time_ms(bf16_loop)
        flops = 6.0 * M * H * F
        print(f"{M:>7} {t2:14.3f} {tf:10.3f} {tb:13.3f} {flops / (t2 * 1e-3) / 1e12:18.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
