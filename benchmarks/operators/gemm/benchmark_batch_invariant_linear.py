# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark RL-Kernel's batch-invariant Triton linear against aiter's gemm_a16w16 on ROCm.

- baseline: aiter's Triton ``gemm_a16w16`` with its default config, written
  inline below;
- ours: ``TritonBatchInvariantLinearOp`` (rl_engine/backends/rocm/gemm/triton_gemm.py),
  the op the registry hands out, with the launch config for M from ``CONFIGS``.

Two parts, forward only, the DeepSeek-V4-Flash lm_head (x [M, 4096] bf16 times
weight [129280, 4096] bf16, fp32 logits):

1. device time per M (benchmarks/common/device_time.py), ours / baseline;
2. error of baseline and ours against x.float() @ w.float().T at M = 1024.

Batch invariance, launch-config invariance and dynamic shape are checked by
tests/backends/rocm/test_rocm_batch_invariant_linear.py and test_rocm_dynamic_shape.py.

Usage (ROCm GPU; aiter with the Triton gemm_a16w16 for the baseline):
    python benchmarks/operators/gemm/benchmark_batch_invariant_linear.py --output linear.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.device_time import (  # noqa: E402
    environment,
    launch_error,
    measure,
    ratio_cell,
    time_cell,
)

DEV = "cuda"
N, K = 129280, 4096  # DeepSeek-V4-Flash vocabulary and hidden size
OUT_DTYPE = torch.float32
PERF_M = [1, 4, 16, 17, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
ERROR_M = 1024


def lm_head_weight(seed=0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(N, K, generator=gen, device=DEV) / K**0.5).to(torch.bfloat16)


def rows(m, seed=0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(m, K, generator=gen, device=DEV).to(torch.bfloat16)


def ours():
    from rl_engine.backends.rocm.gemm.triton_gemm import TritonBatchInvariantLinearOp

    op = TritonBatchInvariantLinearOp()
    return lambda x, w: op(x, w, out_dtype=OUT_DTYPE)


def baseline():
    """Stock aiter with its default config: no ``config`` argument."""
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

    return lambda x, w: gemm_a16w16(x, w, None, dtype=OUT_DTYPE)


def _baseline_or_reason():
    """The baseline op, or "cannot launch: <reason>" if aiter does not have it."""
    try:
        return baseline()
    except Exception as exc:
        return launch_error(exc)


def section_time(m_values, weight):
    ours_op = ours()
    baseline_op = _baseline_or_reason()
    lines = [
        "## Device time, lm_head",
        "",
        f"Forward of x [M, {K}] bf16 times weight [{N}, {K}] bf16, fp32 logits, no bias. "
        "ms per call from GPU events around a CUDA graph, median (min-max) over replays. "
        "Baseline: aiter's gemm_a16w16 with its default config. Ours: RL-Kernel's "
        "batch-invariant Triton linear with the entry for M of `CONFIGS`. "
        "Ratio: ours / baseline, below 1 means ours is faster.",
        "",
        "| M | baseline (ms) | ours (ms) | ours / baseline (ratio) |",
        "|---|---|---|---|",
    ]
    for m in m_values:
        x = rows(m, seed=3)
        if isinstance(baseline_op, str):
            base = baseline_op
        else:
            base = measure(lambda x=x: baseline_op(x, weight))
        result = measure(lambda x=x: ours_op(x, weight))
        lines.append(
            f"| {m} | {time_cell(base)} | {time_cell(result)} | {ratio_cell(result, base)} |"
        )
        print(lines[-1], flush=True)
        del x
        torch.cuda.empty_cache()
    return lines + [""]


def section_error(weight):
    x = rows(ERROR_M, seed=7)
    ref = x.float() @ weight.float().t()
    lines = [
        "## Error against an fp32 linear",
        "",
        f"Forward of the lm_head at M = {ERROR_M}, random inputs. Reference: "
        "x.float() @ w.float().T. Largest and mean absolute error of the fp32 logits.",
        "",
        "| variant | largest absolute error | mean absolute error |",
        "|---|---|---|",
    ]
    variants = {"baseline: aiter, default config": _baseline_or_reason(), "ours": ours()}
    for name, op in variants.items():
        if isinstance(op, str):
            cells = [op, op]
        else:
            try:
                err = (op(x, weight).float() - ref).abs()
                cells = [f"{err.max().item():.3e}", f"{err.mean().item():.3e}"]
            except Exception as exc:
                cells = [launch_error(exc)] * 2
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", help="markdown report, rewritten after every part")
    parser.add_argument(
        "--m",
        type=lambda s: [int(v) for v in s.split(",")],
        default=PERF_M,
        help=f"comma-separated M values to time (default {','.join(map(str, PERF_M))})",
    )
    args = parser.parse_args()

    title = "# Batch-invariant Triton linear on ROCm: device time and error"
    lines = [title, "", environment(), ""]

    def flush():
        if args.output:
            Path(args.output).write_text("\n".join(lines) + "\n")

    with torch.no_grad():
        weight = lm_head_weight()
        lines += section_time(args.m, weight)
        flush()
        lines += section_error(weight) + [""]
    print("\n".join(lines))
    flush()


if __name__ == "__main__":
    main()
