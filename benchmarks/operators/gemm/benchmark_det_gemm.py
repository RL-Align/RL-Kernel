# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Time batch-invariant GEMM and native PyTorch on the same inputs.

The standard WS1 suite exercises the legacy CUDA and Triton paths. The
Muse-Glimmer suite times decode, prefill and training with the model's fixed
BF16 tree. Its optional trace captures a separate invocation after timing.
"""

import argparse
import json
import math
from functools import partial
from pathlib import Path

import torch

from rl_engine.backends.cuda.gemm import deterministic_gemm
from rl_engine.reference.gemm import native_gemm

try:
    from rl_engine.backends.shared.triton.gemm import deterministic_gemm_triton

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

DEV = "cuda"
WARMUP, ITERS = 10, 50

SHAPES = [
    ("qkv", 4096, 4096, 12288),
    ("o_proj", 4096, 4096, 4096),
    ("mlp_up", 4096, 4096, 14336),
    ("mlp_dn", 4096, 14336, 4096),
    ("lm_head", 4096, 4096, 32000),
]


def _time(fn, a, b, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn(a, b)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(a, b)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


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
    out = [
        f"## det_gemm overhead — {dev} (SM{cap[0]}{cap[1]})",
        "",
        "| shape | M | K | N | cuBLAS tf32 | cuBLAS fp32 | det CUDA | det Triton | overhead |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n, M, K, N, t1, t2, t3, t4, ov in rows:
        out.append(
            f"| {n} | {M} | {K} | {N} | {t1:.3f} | {t2:.3f} | {t3:.3f} | {t4:.3f} | {ov:.1f}x |"
        )
    out += [
        "",
        "_Overhead = det CUDA vs cuBLAS (TF32 disabled). The det CUDA path uses "
        "SM90 TMA + mma.sync tensor cores with a fixed single-CTA-per-tile "
        "schedule (no split-K) for bitwise batch-invariance; both det paths "
        "trade speed for invariance. Throughput tuning is deferred per #146._",
    ]
    return "\n".join(out)


def _muse_time(op, x, weight, dy, warmup, iters, trace):
    step = op
    if dy is not None:

        def step(a, w):
            return torch.autograd.grad(op(a, w), (a, w), dy)

    launches = []
    with torch.inference_mode(dy is None):
        elapsed = _time(step, x, weight, warmup, iters)
        if trace:
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                step(x, weight)
                torch.cuda.synchronize()
            launches = [
                # start_us is relative to this case/backend's profiler session.
                # Compare start times only within the same trace entry.
                dict(name=e.name, start_us=e.time_range.start, duration_us=e.device_time_total)
                for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.CUDA
            ]
            if not launches:
                raise RuntimeError("no CUDA kernel events were captured for the requested trace")
    return elapsed, launches


def benchmark_muse_glimmer(warmup, iters, trace_out):
    import triton

    from rl_engine.models.muse_glimmer.dense_gemm import dense_gemm
    from rl_engine.models.muse_glimmer.operator_trace import MuseGlimmerSpec

    ops = {
        "native": lambda x, w: native_gemm(x, w.T),
        "cuda": partial(dense_gemm, backend="cuda"),
        "triton": partial(dense_gemm, backend="triton"),
    }
    device = torch.cuda.get_device_name()
    out = [
        f"## Muse-Glimmer BF16 dense GEMM on {device}",
        "",
        f"PyTorch {torch.__version__}, Triton {triton.__version__}, {warmup} warmups, "
        f"{iters} timed iterations. Mean milliseconds per call. Training includes forward, "
        "dX and dW. Each named projection runs independently.",
        "",
        "Native time is 1x. A value above 1x means the strict backend took longer.",
        "",
        "| Nodes | Phase | M | K | N | Native ms | Strict CUDA ms | Strict Triton ms "
        "| CUDA time / native time | Triton time / native time |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    trace = []
    gen = torch.Generator(device=DEV).manual_seed(501)
    options = dict(device=DEV, dtype=torch.bfloat16, generator=gen)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        for (n, k), nodes in MuseGlimmerSpec.from_manifest().gemm_shape_groups().items():
            weight = torch.randn(n, k, **options) / math.sqrt(k)
            for phase, m in (("decode", 1), ("prefill", 96), ("training", 96)):
                train = phase == "training"
                x = torch.randn(m, k, **options).requires_grad_(train)
                dy = torch.randn(m, n, **options) if train else None
                weight.requires_grad_(train)
                times = {}
                for backend, op in ops.items():
                    times[backend], launches = _muse_time(
                        op, x, weight, dy, warmup, iters, trace_out
                    )
                    if trace_out:
                        trace.append(
                            dict(
                                nodes=nodes,
                                phase=phase,
                                m=m,
                                k=k,
                                n=n,
                                backend=backend,
                                kernels=sorted({launch["name"] for launch in launches}),
                                launches=launches,
                            )
                        )
                native, cuda, tri = times["native"], times["cuda"], times["triton"]
                out.append(
                    f"| {', '.join(nodes)} | {phase} | {m} | {k} | {n} | {native:.4f} | "
                    f"{cuda:.4f} | {tri:.4f} | {cuda / native:.2f}x | {tri / native:.2f}x |"
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduction
    if trace_out:
        Path(trace_out).write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", choices=("standard", "muse-glimmer"), default="standard")
    ap.add_argument("--warmup", type=int, help="warmup iterations per Muse-Glimmer case")
    ap.add_argument("--iters", type=int, help="timed iterations per Muse-Glimmer case")
    ap.add_argument("--trace-out", help="write GPU kernel names and per-launch CUPTI times to JSON")
    args = ap.parse_args()
    if args.model == "muse-glimmer":
        warmup = 2 if args.warmup is None else args.warmup
        iters = 5 if args.iters is None else args.iters
        if warmup < 1 or iters < 1:
            ap.error("warmup must be positive and iters must be positive")
        md = benchmark_muse_glimmer(warmup, iters, args.trace_out)
    else:
        if any(value is not None for value in (args.warmup, args.iters, args.trace_out)):
            ap.error("--warmup, --iters and --trace-out require --model muse-glimmer")
        name, cap = torch.cuda.get_device_name(), torch.cuda.get_device_capability()
        md = to_markdown(run(), name, cap)
    print("\n" + md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
