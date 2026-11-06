# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Seeded single-GPU attention output timing: PyTorch, WS1 and TP1."""

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

from rl_engine.backends.cuda.model_specific.qwenimage.tp_attn_out_bias_gemm import (
    QwenImageTPAttnOutBiasGemmCudaOp,
)
from rl_engine.backends.extension import _C
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.ops.autograd.vjp_fp32 import reduce_rows_fp32


def equal_bytes(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(
            a.detach().contiguous().view(torch.uint8), b.detach().contiguous().view(torch.uint8)
        )
    )


def measure(fn, warmup, samples, distributed=False):
    import torch.distributed as dist

    for _ in range(warmup):
        fn()
    times = []
    for _ in range(samples):
        if distributed:
            dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        held = fn()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if distributed:
            maximum = torch.tensor(elapsed, device="cuda")
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            elapsed = maximum.item()
        times.append(elapsed)
        del held
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    held = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del held
    if distributed:
        dist.barrier()
    return {
        "median_ms": statistics.median(times),
        "samples_ms": times,
        "base_allocated_mib": before / 1024**2,
        "peak_delta_mib": (peak - before) / 1024**2,
        "peak_total_allocated_mib": peak / 1024**2,
    }


def native_fb(x, weight, bias, grad, keys):
    """WS1 native arithmetic with the same canonical collection as TP1."""
    y = _C.attn_out_bias_gemm_cuda_forward(
        x.float().contiguous(),
        weight.float().contiguous(),
        bias.float().contiguous(),
        x.dtype == torch.bfloat16,
    )
    dx = _C.attn_out_tree_gemm_cuda(grad.float().contiguous(), weight.t().float().contiguous()).to(
        x.dtype
    )
    with canonical_backward_session() as session:
        ws, bs = session.register("attn_out.weight", keys), session.register("attn_out.bias", keys)
        dw = session.submit_linear(
            "attn_out.weight",
            ws,
            x,
            grad,
            lambda rows, grads: _C.attn_out_dw_left_fold_cuda(
                grads.float().contiguous(), rows.float().contiguous()
            ).to(weight.dtype),
        )
        db = session.submit_rows(
            "attn_out.bias", bs, grad, lambda rows: reduce_rows_fp32(rows).to(bias.dtype)
        )
    session.validate_complete()
    return y, dx, dw, db


def tp_calls(op, x, weight, bias, grad, keys, cuts=None):
    cuts = [(0, x.shape[0])] if cuts is None else cuts

    def forward():
        with torch.no_grad():
            return tuple(op(x[s:e], weight, bias) for s, e in cuts)

    def forward_backward():
        with canonical_backward_session() as session:
            ys = [
                op(x[s:e], weight, bias, logical_keys=keys[s:e], parameter_id="attn_out")
                for s, e in cuts
            ]
            grads = torch.autograd.grad(ys, (x, weight, bias), [grad[s:e] for s, e in cuts])
        session.validate_complete()
        return tuple(ys), *grads

    return forward, forward_backward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 129, 4096, 6032, 6889])
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=386)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows) < 1 or args.warmup < 0 or args.samples < 1:
        parser.error("rows/samples must be positive and warmup nonnegative")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dtype = getattr(torch, args.dtype)
    op = QwenImageTPAttnOutBiasGemmCudaOp()
    root = Path(__file__).resolve().parents[3]
    sources = [
        "csrc/cuda/gemm/attn_out_bias_gemm.cu",
        "rl_engine/backends/cuda/model_specific/qwenimage/tp_attn_out_bias_gemm.py",
        "rl_engine/ops/autograd/canonical_backward.py",
    ]
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cpu_threads": torch.get_num_threads(),
        "seed": args.seed,
        "dtype": args.dtype,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
        },
        "scope": (
            "Single GPU. WS1 and TP1 include canonical parameter collection; "
            "PyTorch is the ordinary performance control."
        ),
        "results": [],
    }
    for rows in args.rows:
        generator = torch.Generator().manual_seed(args.seed)

        def rand(shape, scale=1):
            return (
                (torch.randn(shape, generator=generator) * scale)
                .to(device="cuda", dtype=dtype)
                .requires_grad_()
            )

        x, weight, bias, grad = (
            rand((rows, 3072)),
            rand((3072, 3072), 3072**-0.5),
            rand((3072,), 0.1),
            rand((rows, 3072), 3072**-0.5).detach(),
        )
        keys = torch.stack(
            (
                torch.zeros(rows, dtype=torch.int64, device="cuda"),
                torch.arange(rows, device="cuda"),
            ),
            1,
        )
        tp_f, tp_fb = tp_calls(op, x, weight, bias, grad, keys)

        def torch_fb():
            y = F.linear(x, weight, bias)
            return y, *torch.autograd.grad(y, (x, weight, bias), grad)

        gold, actual = native_fb(x, weight, bias, grad, keys), tp_fb()
        actual = torch.cat(actual[0]), *actual[1:]
        assert all(equal_bytes(a, b) for a, b in zip(actual, gold))
        del gold, actual
        for backend, forward, fb in [
            ("pytorch", lambda: F.linear(x, weight, bias), torch_fb),
            (
                "ws1",
                lambda: _C.attn_out_bias_gemm_cuda_forward(
                    x.float(), weight.float(), bias.float(), dtype == torch.bfloat16
                ),
                lambda: native_fb(x, weight, bias, grad, keys),
            ),
            ("tp1", tp_f, tp_fb),
        ]:
            with torch.no_grad():
                f = measure(forward, args.warmup, args.samples)
            total = measure(fb, args.warmup, args.samples)
            report["results"].append(
                {
                    "rows": rows,
                    "backend": backend,
                    "forward": f,
                    "forward_backward": total,
                    "fb_rows_per_second": rows * 1000 / total["median_ms"],
                }
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"{backend} M={rows}: F={f['median_ms']:.3f} ms, "
                f"F+B={total['median_ms']:.3f} ms, "
                f"peak delta={total['peak_delta_mib']:.2f} MiB",
                flush=True,
            )


if __name__ == "__main__":
    main()
