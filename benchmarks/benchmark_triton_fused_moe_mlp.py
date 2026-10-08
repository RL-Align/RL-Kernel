#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton fused MoE MLP vs the CUDA path, and both vs an FP64 reference.

The Triton kernels are the ROCm-facing lane
(``rl_engine/kernels/ops/triton/moe/fused_mlp.py``); this is the script to run
on a CDNA host when tuning them. On an H100 it reports the gap against the
hand-written CUDA kernels, which the Triton lane deliberately does not chase.

Two things are measured, because only one of them is allowed to regress:

* time per launch, split fc1 / fc3, where the Triton lane is expected to lose;
* max relative error against an FP64 reference, where it must not.

The shared expert (BF16, fc1+SwiGLU and the TP-invariant fc3) is timed after
the routed half, next to an unfused ``torch`` BF16 chain (rocBLAS/hipBLASLt or
cuBLAS) for scale. ``--shared-tokens ''`` skips it.

The CUDA columns are printed only when ``rl_engine._C`` carries the SM90
symbols; otherwise the Triton lane is timed on its own.

    python benchmarks/benchmark_triton_fused_moe_mlp.py [--tokens 512,2048,8192]
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


def fp64_reference(x_q, w1, w2, offsets, p_s, ffn: int) -> torch.Tensor:
    """The same chain in FP64: neither lane's rounding, so both can be scored."""
    from rl_engine.moe.mx_format import mx_dequantize

    a, w1d, w2d = mx_dequantize(x_q).double(), mx_dequantize(w1).double(), mx_dequantize(w2).double()
    out = torch.zeros(a.shape[0], w2d.shape[1], dtype=torch.float64, device=a.device)
    offs = offsets.tolist()
    for e in range(len(offs) - 1):
        lo, hi = offs[e], offs[e + 1]
        if lo == hi:
            continue
        z = a[lo:hi] @ w1d[e].t()
        gate = z[:, :ffn].clamp(max=10.0)
        up = z[:, ffn:].clamp(-10.0, 10.0)
        h = ((gate * torch.sigmoid(gate)) * up) * p_s[lo:hi, None].double()
        out[lo:hi] = h @ w2d[e].t()
    return out


def rel(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.double(), want.double()
    return ((got - want).abs().max() / (want.abs().max() + 1e-30)).item()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--ffn", type=int, default=2048)
    p.add_argument("--tokens", default="512,2048,8192")
    p.add_argument("--accuracy-tokens", default="512,2048", help="subset scored against FP64")
    p.add_argument("--shared-tokens", default="512,2048,8192", help="shared-expert rows; '' skips")
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("GPU required")
        return 1
    try:
        from rl_engine.kernels.ops.triton.moe import fused_mlp as tf
    except ImportError as exc:
        print(f"[skip] triton is not installed: {exc}")
        return 1
    from rl_engine.moe.mx_format import mx_quantize

    cuda = None
    try:
        from rl_engine.moe.backends import Sm90FusedMoeMlp

        cuda = Sm90FusedMoeMlp()
    except NotImplementedError as exc:
        print(f"[note] CUDA columns omitted: {exc}\n")

    E, H, F = args.experts, args.hidden, args.ffn
    tokens = [int(v) for v in args.tokens.split(",")]
    scored = {int(v) for v in args.accuracy_tokens.split(",")}
    print(f"E={E} H={H} F={F} ({torch.cuda.get_device_name(0)})")

    rows_time, rows_acc = [], []
    for M in tokens:
        gen = torch.Generator().manual_seed(1)
        x = (torch.randn(M, H, generator=gen) * 0.5).to(torch.bfloat16).cuda()
        w1 = mx_quantize(
            (torch.randn(E, 2 * F, H, generator=gen) / H**0.5).to(torch.bfloat16).cuda(), "e2m1"
        )
        w2 = mx_quantize(
            (torch.randn(E, H, F, generator=gen) / F**0.5).to(torch.bfloat16).cuda(), "e2m1"
        )
        p_s = torch.rand(M, generator=gen).cuda()
        offsets = torch.tensor(
            [0] + [M * (i + 1) // E for i in range(E)], dtype=torch.int32
        ).cuda()
        x_q = mx_quantize(x, "e4m3")

        t_fc1 = time_ms(lambda: tf.routed_fc1_swiglu_quant(x_q, w1, offsets, p_s))
        h_q = tf.routed_fc1_swiglu_quant(x_q, w1, offsets, p_s)
        t_fc3 = time_ms(lambda: tf.routed_fc3(h_q, w2, offsets))
        y_tri = tf.routed_fc3(h_q, w2, offsets)

        t_cuda, y_cuda = float("nan"), None
        if cuda is not None:
            from rl_engine.moe.contract import ExpertBatch

            batch = ExpertBatch(
                x=x,
                expert_offsets=offsets,
                p_s=p_s,
                w1=w1,
                w2=w2,
                lora=None,
                output_slot=torch.arange(M, dtype=torch.int32, device="cuda"),
                numeric_profile=cuda.numeric_profile,
            )
            t_cuda = time_ms(lambda: cuda.forward(batch, x_q))
            y_cuda = cuda.forward(batch, x_q)

        rows_time.append((M, t_fc1, t_fc3, t_fc1 + t_fc3, t_cuda))
        if M in scored:
            ref = fp64_reference(x_q, w1, w2, offsets, p_s, F)
            rows_acc.append(
                (
                    M,
                    rel(y_tri, ref),
                    rel(y_cuda, ref) if y_cuda is not None else float("nan"),
                    rel(y_tri, y_cuda) if y_cuda is not None else float("nan"),
                )
            )

    print(f"\n{'M':>7} {'triton fc1':>12} {'triton fc3':>12} {'triton':>10} {'cuda':>10} {'ratio':>7}")
    for M, a, b, tot, c in rows_time:
        have = c == c  # NaN when the CUDA extension is unavailable
        cuda_col = f"{c:8.3f}ms" if have else f"{'-':>10}"
        ratio = f"{tot / c:6.2f}x" if have else f"{'-':>7}"
        print(f"{M:>7} {a:10.3f}ms {b:10.3f}ms {tot:8.3f}ms {cuda_col} {ratio}")

    if rows_acc:
        print(f"\n{'M':>7} {'triton vs fp64':>15} {'cuda vs fp64':>13} {'triton vs cuda':>15}")
        for M, t, c, tc in rows_acc:
            cols = "".join(f"{v:>13.2e}" if v == v else f"{'-':>13}" for v in (c, tc))
            print(f"{M:>7} {t:15.2e} {cols}")
        print("\nBoth lanes sit on the MX quantization floor; the middle two columns")
        print("agreeing is the result that matters -- portability costs time, not accuracy.")
    shared_tokens = [int(v) for v in args.shared_tokens.split(",") if v]
    if shared_tokens:
        bench_shared(tf, H, F, shared_tokens)
    return 0


def bench_shared(tf, hidden: int, ffn: int, tokens: list[int]) -> None:
    """Shared expert: fused fc1+SwiGLU, TP-invariant fc3, vs an unfused torch chain."""
    gen = torch.Generator().manual_seed(2)
    w1 = (torch.randn(2 * ffn, hidden, generator=gen) / hidden**0.5).to(torch.bfloat16).cuda()
    w2 = (torch.randn(hidden, ffn, generator=gen) / ffn**0.5).to(torch.bfloat16).cuda()

    def torch_chain(x):
        z = x @ w1.t()
        g, u = z[:, :ffn].float(), z[:, ffn:].float()
        return ((g * torch.sigmoid(g)) * u).to(torch.bfloat16) @ w2.t()

    print(f"\nshared expert H={hidden} F={ffn}")
    print(f"{'T':>7} {'fc1+swiglu':>11} {'fc3 (tp1)':>10} {'total':>9} {'torch':>9} {'TFLOPS':>7}")
    for t in tokens:
        x = (torch.randn(t, hidden, generator=gen) * 0.7).to(torch.bfloat16).cuda()
        h = tf.fused_shared_fc1_swiglu(x, w1)
        a = time_ms(lambda: tf.fused_shared_fc1_swiglu(x, w1))
        b = time_ms(lambda: tf.shared_fc3(h, w2))
        ref = time_ms(lambda: torch_chain(x))
        tflops = 6 * t * hidden * ffn / ((a + b) * 1e-3) / 1e12
        print(f"{t:>7} {a:9.3f}ms {b:8.3f}ms {a + b:7.3f}ms {ref:7.3f}ms {tflops:7.0f}")


if __name__ == "__main__":
    sys.exit(main())
