# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Latency / TFLOP-s bench for the MLP down projection row (WS1).

Measures the model shapes of ``y = bf16(fp32_accum(x @ weight.T) + bias)`` --
``img_mlp.net.2`` and ``txt_mlp.net.2`` are both ``4096 x 12288 -> 3072`` -- for
a selectable backend, and gates every run against the independent fp32 CPU
reference before reporting timings (a fast-but-wrong kernel is not a result).
``--backend pytorch`` is that reference itself, so its gate is reported as not
applicable rather than as a self-comparison; its timing runs the fp64-emulated
tree on the benchmark device, and the report labels that device explicitly.

The three backends are imported lazily, so ``--backend triton`` never needs the
CUDA extension and ``--backend pytorch`` never needs Triton. The PyTorch
reference is the fp64-emulated tree: its per-row cost is far above the kernels',
so ``--backend pytorch`` is timed at ``--gate-rows`` rows and the report says so.

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
    # The declared bound is an absolute error in ulps of the largest reference
    # magnitude ("worst <= 8 bf16-ulp of max|reference|", see the operator
    # page), not a per-element relative error: keep the normalizer global so the
    # reported metric is the published contract.
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
    # The PyTorch backend is the fp64-emulated tree: timing the model shape
    # would take minutes per iteration, so it is measured at the gate window
    # (its cost is linear in rows) and the report states the scale.
    timed_rows = min(rows, args.gate_rows) if args.backend == "pytorch" else rows
    x, weight, bias = _inputs(timed_rows, args.k_dim, args.n_dim, dtype, seed=args.seed)

    out = op(x, weight, bias=bias)
    assert out.dtype is torch.bfloat16 and out.shape == (timed_rows, args.n_dim), (
        out.dtype,
        out.shape,
    )

    # A `cuda` row takes one of two frozen contracts depending on the build,
    # the device and RL_KERNEL_MLP_DOWN_GEMM_BACKEND; record the resolved one so
    # the numbers are not compared across different arithmetic orders.
    resolved = args.backend
    if args.backend == "cuda":
        from rl_engine.backends.cuda.gemm.mlp_down_gemm import (
            mlp_down_gemm_backend_used,
            mlp_down_gemm_contract_used,
        )

        resolved = (
            f"cuda ({mlp_down_gemm_backend_used(x, weight)}, "
            f"{mlp_down_gemm_contract_used(x, weight)})"
        )

    if args.backend == "pytorch":
        # The candidate is the reference the gate compares against: a
        # self-comparison is not evidence, so report it as not applicable.
        gate_passed = True
        gate_lines = ["  not applicable: --backend pytorch is the accuracy gold itself"]
        gate_verdict = "  gate verdict: N/A (the candidate is the reference)"
    else:
        gate_passed, gate_lines = _accuracy_gate(
            op, x, weight, bias, args.gate_rows, backward=True
        )
        gate_verdict = (
            f"  gate verdict: {'PASS' if gate_passed else 'FAIL'} "
            f"(declared: identical >= {DECLARED_MIN_IDENTICAL_FRACTION * 100:.0f}%, "
            f"worst <= {DECLARED_MAX_ULPS} bf16-ulp)"
        )

    xr = x.detach().clone().requires_grad_(True)
    wr = weight.detach().clone().requires_grad_(True)
    br = bias.detach().clone().requires_grad_(True)
    grad_out = torch.randn(timed_rows, args.n_dim, generator=torch.Generator().manual_seed(7))
    grad_out = grad_out.to(dtype).to(DEV)

    def forward_only():
        op(x, weight, bias=bias)

    def forward_backward():
        loss = op(xr, wr, bias=br)
        torch.autograd.grad(loss, (xr, wr, br), grad_outputs=grad_out)

    fwd_ms = _time(forward_only, args.warmup, args.iters)
    fwd_bwd_ms = _time(forward_backward, args.warmup, args.iters)

    gemm_flops = 2.0 * timed_rows * args.k_dim * args.n_dim
    fwd_mb = _peak_extra_mb(forward_only)
    fwd_bwd_mb = _peak_extra_mb(forward_backward)
    reduced = timed_rows != rows
    # At a reduced row count the fixed launch cost dominates, so the TFLOP/s
    # columns would not be comparable with a full-shape row: suppress them. The
    # fp64-emulated reference is suppressed at any row count, because its cost is
    # a CPU-bound tree walk and a TFLOP/s figure would be meaningless next to a
    # kernel row even when it happens to run at the requested shape.
    suppressed = reduced or args.backend == "pytorch"
    fwd_tflops = "n/a" if suppressed else f"{gemm_flops / fwd_ms / 1e9:.2f}"
    fwd_bwd_tflops = "n/a" if suppressed else f"{3 * gemm_flops / fwd_bwd_ms / 1e9:.2f}"
    table = [
        "| case | backend | dtype | M | K | N | fwd ms | fwd TFLOP/s | fwd MB | "
        "fwd+bwd ms | fwd+bwd TFLOP/s | fwd+bwd MB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| mlp_down_gemm | {resolved} | {args.dtype} | {timed_rows} | {args.k_dim} | "
        f"{args.n_dim} | "
        f"{fwd_ms:.3f} | {fwd_tflops} | {fwd_mb:.0f} | {fwd_bwd_ms:.3f} | "
        f"{fwd_bwd_tflops} | {fwd_bwd_mb:.0f} |",
    ]
    # fwd+bwd = fwd + dx + dW, i.e. three equal-sized GEMMs (db is a fold).
    device_label = torch.cuda.get_device_name()
    if args.backend == "pytorch":
        device_label += ", fp64 reference executed on that device"
    report = [
        f"mlp_down_gemm ({resolved}, {device_label})",
        "",
        *table,
        "",
        "accuracy gate vs the fp32 CPU reference:",
        *gate_lines,
        gate_verdict,
    ]
    if reduced:
        report.append(
            f"note: timing uses {timed_rows} of {rows} rows; the fp64-emulated reference "
            "is not timed at the model shape (its cost is linear in rows), so its "
            "TFLOP/s columns are suppressed as not comparable to a full-shape row."
        )
    elif args.backend == "pytorch":
        report.append(
            "note: TFLOP/s is suppressed for the fp64-emulated reference: its cost is a "
            "CPU-bound tree walk, so the figure is not comparable to a kernel row."
        )
    report.append("")
    print("\n".join(report))
    if args.out:
        with open(args.out, "w") as handle:
            handle.write("\n".join(report) + "\n")
    if not gate_passed:
        raise SystemExit("accuracy gate failed")


if __name__ == "__main__":
    main()
