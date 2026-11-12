# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Seeded single-H20 local QKV timing; native manual VJP vs TP1 autograd."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from rl_engine.backends.cuda.model_specific.qwenimage.tp_qkv_bias_gemm import (
    QwenImageTPQKVBiasGemmCudaOp,
)
from rl_engine.backends.extension import _C
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session


def measure(fn, warmup, samples):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        held = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        del held
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    held = fn()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - before) / 1024**2
    del held
    return {"median_ms": statistics.median(times), "samples_ms": times, "peak_delta_mib": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 129, 4096, 6032, 6889])
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["pytorch", "native", "tp1"],
        default=["pytorch", "native", "tp1"],
    )
    parser.add_argument("--seed", type=int, default=386)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows) < 1 or args.samples < 1 or args.warmup < 0:
        parser.error("rows/samples must be positive and warmup must be nonnegative")
    dtype = getattr(torch, args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    op = QwenImageTPQKVBiasGemmCudaOp()
    root = Path(__file__).resolve().parents[3]
    sources = (
        "csrc/cuda/model_specific/qwenimage/det_linear.cu",
        "rl_engine/backends/cuda/model_specific/qwenimage/tp_qkv_bias_gemm.py",
        "rl_engine/ops/autograd/canonical_backward.py",
    )
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cpu_threads": torch.get_num_threads(),
        "precision": {"allow_tf32": False, "allow_bf16_reduced_precision_reduction": False},
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
        },
        "seed": args.seed,
        "dtype": args.dtype,
        "scope": "Single GPU; no TP communication. Native uses manual VJP and canonical "
        "collection; tp1 uses the public autograd wrapper and the same collection.",
        "results": [],
    }
    for rows in args.rows:
        gen = torch.Generator(device="cpu").manual_seed(args.seed)

        def rand(shape, scale=1):
            return (torch.randn(shape, generator=gen) * scale).to(device="cuda", dtype=dtype)

        x, w, b, grad = (
            rand((rows, 3072)),
            rand((9216, 3072), 3072**-0.5),
            rand((9216,), 0.1),
            rand((rows, 9216), 9216**-0.5),
        )
        x.requires_grad_()
        w.requires_grad_()
        b.requires_grad_()
        keys = torch.stack(
            (
                torch.zeros(rows, device="cuda", dtype=torch.int64),
                torch.arange(rows, device="cuda"),
            ),
            dim=1,
        )

        def native_fb():
            y = _C.qwenimage_det_linear_forward(x, w, b, 0, False)[0]
            dx = _C.qwenimage_det_linear_backward_input(grad.float(), w, dtype)
            with canonical_backward_session() as session:
                ws = session.register("qkv.weight", keys)
                bs = session.register("qkv.bias", keys)
                dw = session.submit_linear(
                    "qkv.weight",
                    ws,
                    x,
                    grad,
                    lambda a, g: _C.qwenimage_det_linear_backward_weight(
                        g.float(), a, dtype, False
                    )[0],
                )
                db = session.submit_rows(
                    "qkv.bias",
                    bs,
                    grad,
                    lambda g: _C.qwenimage_det_linear_backward_bias(g.float(), dtype),
                )
            session.validate_complete()
            return y, dx, dw, db

        def tp1_fb():
            with canonical_backward_session() as session:
                y = op(x, w, b, logical_keys=keys, parameter_id="qkv")
                grads = torch.autograd.grad(y, (x, w, b), grad)
            session.validate_complete()
            return y, *grads

        def torch_fb():
            y = F.linear(x, w, b)
            return y, *torch.autograd.grad(y, (x, w, b), grad)

        # The reused native path is the byte-level control for the new adapter.
        reference, adapted = native_fb(), tp1_fb()
        if not all(
            torch.equal(a.detach().view(torch.uint8), z.detach().view(torch.uint8))
            for a, z in zip(reference, adapted)
        ):
            raise RuntimeError("TP1 output or gradient bytes differ from the original native API")
        del reference, adapted
        for backend in args.backends:
            forward = {
                "pytorch": lambda: F.linear(x, w, b),
                "native": lambda: _C.qwenimage_det_linear_forward(x, w, b, 0, False)[0],
                "tp1": lambda: op(x, w, b),
            }[backend]
            backward = {"pytorch": torch_fb, "native": native_fb, "tp1": tp1_fb}[backend]
            with torch.no_grad():
                f = measure(forward, args.warmup, args.samples)
            fb = measure(backward, args.warmup, args.samples)
            item = {
                "rows": rows,
                "backend": backend,
                "forward": f,
                "forward_backward": fb,
                "fb_rows_per_second": 1000 * rows / fb["median_ms"],
            }
            report["results"].append(item)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"{backend} M={rows}: F={f['median_ms']:.3f} ms, "
                f"F+B={fb['median_ms']:.3f} ms, peak delta={fb['peak_delta_mib']:.2f} MiB",
                flush=True,
            )


if __name__ == "__main__":
    main()
