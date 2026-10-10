# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Latency / TFLOP-s bench for the MLP down projection row (WS1).

Measures the model shapes of ``y = bf16(fp32_accum(x @ weight.T) + bias)`` --
``img_mlp.net.2`` and ``txt_mlp.net.2`` are both ``4096 x 12288 -> 3072`` -- for
a selectable backend, and gates every run against the independent fp32 CPU
reference before reporting timings (a fast-but-wrong kernel is not a result).

The three backends are imported lazily, so ``--backend triton`` never needs the
CUDA extension and ``--backend pytorch`` never needs Triton.

Usage::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \\
        benchmarks/operators/gemm/benchmark_mlp_down_gemm.py \\
        --backend triton --dtype bf16 --batch 4096 --seq 1
"""

from __future__ import annotations

import argparse

import torch

from rl_engine.reference.gemm.mlp_down_gemm import (
    mlp_down_gemm_reference_backward,
    mlp_down_gemm_reference_forward,
)

DEV = "cuda"
WARMUP, ITERS = 5, 20
BF16_ULP = 2.0**-8
# The fp32 CPU reference is a 32-wide-leaf mid-split tree evaluated in fp64
# emulation, so gating the full token dimension costs minutes.  The gate uses a
# fixed row window instead: a logical row's arithmetic does not depend on how
# many rows share the launch (that is this row's invariance promise).
DEFAULT_GATE_ROWS = 256


def _load_backend(name: str):
    if name == "triton":
        from rl_engine.backends.shared.triton.gemm.mlp_down_gemm import TritonMlpDownGemmOp

        return TritonMlpDownGemmOp()
    if name == "cuda":
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import CudaMlpDownGemmOp

        return CudaMlpDownGemmOp()
    if name == "pytorch":
        from rl_engine.reference.gemm.mlp_down_gemm import NativeMlpDownGemmOp

        return NativeMlpDownGemmOp()
    raise ValueError(f"unknown backend {name!r}")


def _inputs(rows: int, k_dim: int, n_dim: int, dtype: torch.dtype, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, k_dim, generator=gen)
    weight = torch.randn(n_dim, k_dim, generator=gen) / (k_dim**0.5)
    bias = torch.randn(n_dim, generator=gen) / (k_dim**0.5)
    return (
        x.to(dtype).to(DEV),
        weight.to(dtype).to(DEV),
        bias.to(dtype).to(DEV),
    )


DECLARED_MIN_IDENTICAL_FRACTION = 0.99
DECLARED_MAX_ULPS = 8.0


def _deviation(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    """``(identical, worst_uls_bf16_ref, worst_ulps_fp32_ref)``.

    The declared "bit-identical elements" bound compares the store against the
    correctly-rounded bf16 reference (a bf16 store never equals a raw fp32 value
    elementwise), while the worst-case distance to the raw fp32 reference bounds
    how far the single cast moves the result.
    """

    ref_bf16 = ref.to(torch.bfloat16)
    got_f = got.detach().float().cpu()
    ref_b = ref_bf16.float().cpu()
    ref_32 = ref.float().cpu()
    identical = float((got_f == ref_b).float().mean())
    ulp = BF16_ULP * ref_32.abs().max().clamp_min(1e-12)
    return (
        identical,
        float((got_f - ref_b).abs().max() / ulp),
        float((got_f - ref_32).abs().max() / ulp),
    )


def _time(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Milliseconds per call, via CUDA events around ``iters`` synchronized runs."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def _peak_extra_mb(fn) -> float:
    """Peak device bytes above the baseline already allocated, in MB.

    The template for this row's PR reports peak extra memory next to latency. For
    the mma contract that is exactly the output (plus, for the backward, the
    autograd saves and the three gradient tensors) -- there is no materialized
    intermediate. The portable fp32-tree path additionally materializes fp32
    copies of its bf16 operands, so its figure is several hundred MB larger.
    """

    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / (1024.0 * 1024.0)


def _accuracy_gate(op, x, weight, bias, gate_rows, backward: bool):
    """Compare the candidate against the independent fp32 CPU reference."""

    rows = min(gate_rows, x.size(0))
    xw, ww, bw = x[:rows].cpu(), weight.cpu(), bias.cpu()
    got = op(x[:rows].contiguous(), weight, bias=bias)
    ref = mlp_down_gemm_reference_forward(xw.float(), ww.float(), bw.float())
    out_identical, out_worst, out_worst_fp32 = _deviation(got, ref)
    passed = out_identical >= DECLARED_MIN_IDENTICAL_FRACTION and out_worst <= DECLARED_MAX_ULPS
    lines = [
        f"  forward   : identical={out_identical * 100:.4f}% (vs bf16 reference)  "
        f"worst={out_worst:.4f} bf16-ulp of max|ref|  "
        f"worst_vs_fp32={out_worst_fp32:.4f} ulp  ({rows} gate rows, dtype={got.dtype})",
    ]
    if backward:
        grad = torch.randn(rows, weight.size(0), generator=torch.Generator().manual_seed(11))
        grad = (grad / (weight.size(0) ** 0.5)).to(torch.bfloat16).to(DEV)
        xr = x[:rows].contiguous().detach().requires_grad_(True)
        wr = weight.detach().clone().requires_grad_(True)
        br = bias.detach().clone().requires_grad_(True)
        op(xr, wr, bias=br).backward(grad)
        ref_dx, ref_dw, ref_db = mlp_down_gemm_reference_backward(
            xw.float(), ww.float(), grad.cpu().float()
        )
        for name, got_grad, ref_grad in (
            ("dx", xr.grad, ref_dx),
            ("dW", wr.grad, ref_dw),
            ("db", br.grad, ref_db),
        ):
            identical, worst, worst_fp32 = _deviation(got_grad, ref_grad)
            passed = passed and identical >= DECLARED_MIN_IDENTICAL_FRACTION
            passed = passed and worst <= DECLARED_MAX_ULPS
            lines.append(
                f"  {name:9s}: identical={identical * 100:.4f}% (vs bf16 reference)  "
                f"worst={worst:.4f} bf16-ulp  worst_vs_fp32={worst_fp32:.4f} ulp"
            )
    return passed, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("triton", "pytorch", "cuda"), default="triton")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seq", type=int, default=1)
    parser.add_argument("--k-dim", type=int, default=12288)
    parser.add_argument("--n-dim", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--gate-rows", type=int, default=DEFAULT_GATE_ROWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = args.batch * args.seq
    dtype = torch.bfloat16
    op = _load_backend(args.backend)
    x, weight, bias = _inputs(rows, args.k_dim, args.n_dim, dtype, seed=args.seed)

    out = op(x, weight, bias=bias)
    assert out.dtype is torch.bfloat16 and out.shape == (rows, args.n_dim), (out.dtype, out.shape)

    gate_passed, gate_lines = _accuracy_gate(op, x, weight, bias, args.gate_rows, backward=True)

    xr = x.detach().clone().requires_grad_(True)
    wr = weight.detach().clone().requires_grad_(True)
    br = bias.detach().clone().requires_grad_(True)
    grad_out = torch.randn(rows, args.n_dim, generator=torch.Generator().manual_seed(7))
    grad_out = grad_out.to(dtype).to(DEV)

    def forward_only():
        op(x, weight, bias=bias)

    def forward_backward():
        loss = op(xr, wr, bias=br)
        torch.autograd.grad(loss, (xr, wr, br), grad_outputs=grad_out)

    fwd_ms = _time(forward_only, args.warmup, args.iters)
    fwd_bwd_ms = _time(forward_backward, args.warmup, args.iters)

    gemm_flops = 2.0 * rows * args.k_dim * args.n_dim
    fwd_mb = _peak_extra_mb(forward_only)
    fwd_bwd_mb = _peak_extra_mb(forward_backward)
    table = [
        "| case | backend | dtype | M | K | N | fwd ms | fwd TFLOP/s | fwd MB | "
        "fwd+bwd ms | fwd+bwd TFLOP/s | fwd+bwd MB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| mlp_down_gemm | {args.backend} | {args.dtype} | {rows} | {args.k_dim} | {args.n_dim} | "
        f"{fwd_ms:.3f} | {gemm_flops / fwd_ms / 1e9:.2f} | {fwd_mb:.0f} | {fwd_bwd_ms:.3f} | "
        f"{3 * gemm_flops / fwd_bwd_ms / 1e9:.2f} | {fwd_bwd_mb:.0f} |",
    ]
    # fwd+bwd = fwd + dx + dW, i.e. three equal-sized GEMMs (db is a fold).
    report = [
        f"mlp_down_gemm ({args.backend}, {torch.cuda.get_device_name()})",
        "",
        *table,
        "",
        "accuracy gate vs the fp32 CPU reference:",
        *gate_lines,
        f"  gate verdict: {'PASS' if gate_passed else 'FAIL'} "
        f"(declared: identical >= {DECLARED_MIN_IDENTICAL_FRACTION * 100:.0f}%, "
        f"worst <= {DECLARED_MAX_ULPS} bf16-ulp)",
        "",
    ]
    print("\n".join(report))
    if args.out:
        with open(args.out, "w") as handle:
            handle.write("\n".join(report) + "\n")
    if not gate_passed:
        raise SystemExit("accuracy gate failed")


if __name__ == "__main__":
    main()
