# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""mHC pre device time: batch-invariant per-M configs vs aiter's own config (DSV4 RFC task 1, #508).

The mhc_pre op picks a config per M through the launch-config dispatch
(rl_engine/backends/rocm/config_utils.py); every config of the table splits K
the same way, so the result is batch invariant.

- baseline: mhc_pre_dsv4 with aiter's own config choice (splits K differently
  from M = 1024 on, not batch invariant; cannot launch at M = 8192-16383 on
  gfx942 with C = 4096);
- ours: the mhc_pre op, i.e. mhc_pre_dsv4 with the config the dispatch picks;
- reference: mhc_pre_dsv4 with FIXED_PRE_CONFIG for every M (no table);
- comparison: ours / baseline and reference / baseline, median device time.

M values include some between table buckets.

Forward only: the backward is the same PyTorch recompute for all. Device time
from GPU events; every case is measured in several independent trials and the
median and min-max over trials are reported. Shapes follow DeepSeek-V4-Flash-0731:
n=4 streams, C=4096, bf16 residual, 20 Sinkhorn rounds.

Usage (GPU; aiter with the DeepSeek-V4 mHC ops and AITER_TRITON_ONLY=1):
    python benchmarks/operators/mhc/benchmark_mhc_pre.py --output artifacts/benchmarks/mhc_pre.md
"""

import argparse
import statistics

import torch

N, C = 4, 4096
MIX = 2 * N + N * N
M_VALUES = [1, 16, 17, 64, 100, 256, 1024, 2048, 3000, 4096, 8192, 16384, 32768, 65536]
WARMUP, ITERS, TRIALS = 10, 50, 5


def _time_ms(fn):
    """Mean device time per call over ITERS calls, after WARMUP calls."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def _measure(fn):
    """(median, min, max) over TRIALS, or the launch error as a string."""
    try:
        samples = [_time_ms(fn) for _ in range(TRIALS)]
    except Exception as exc:  # aiter's own config cannot launch at some M
        return f"cannot launch: {type(exc).__name__}"
    return statistics.median(samples), min(samples), max(samples)


def _cell(result):
    if isinstance(result, str):
        return result
    median, low, high = result
    return f"{median:.4f} ({low:.4f}-{high:.4f})"


def _ratio(result, baseline):
    if isinstance(result, str) or isinstance(baseline, str):
        return "n/a: a variant cannot launch"
    return f"{result[0] / baseline[0]:.2f}x"


def run(m_values):
    from aiter.ops.triton.fusions import mhc_pre_dsv4

    from rl_engine.backends.rocm.mhc.aiter_mhc import FIXED_PRE_CONFIG, AiterMHCPreOp

    ours_op = AiterMHCPreOp()
    gen = torch.Generator(device="cuda").manual_seed(0)
    fn = torch.randn(MIX, N * C, generator=gen, device="cuda") / (N * C) ** 0.5
    scale = 0.5 + torch.rand(3, generator=gen, device="cuda")
    base = 0.1 * torch.randn(MIX, generator=gen, device="cuda")

    rows = []
    for m in m_values:
        residual = torch.randn(m, N, C, generator=gen, device="cuda").to(torch.bfloat16)
        with torch.no_grad():
            baseline = _measure(lambda: mhc_pre_dsv4(residual, fn, scale, base))
            ours = _measure(lambda: ours_op(residual, fn, scale, base))
            fixed = _measure(
                lambda: mhc_pre_dsv4(residual, fn, scale, base, config=FIXED_PRE_CONFIG)
            )
        row = (
            m,
            _cell(baseline),
            _cell(ours),
            _ratio(ours, baseline),
            _cell(fixed),
            _ratio(fixed, baseline),
        )
        rows.append(row)
        print(" | ".join(str(c) for c in row), flush=True)
        residual = None  # free before the next M
        torch.cuda.empty_cache()
    return rows


def to_markdown(rows, device):
    header = (
        f"mHC pre forward device time, ms per call, on {device}: n={N}, C={C}, bf16, "
        f"20 Sinkhorn rounds. Median over {TRIALS} trials of {ITERS} calls (min-max).\n\n"
        "| M | baseline: aiter's own config, not batch invariant | ours: mhc_pre op "
        "| ours / baseline (median) | reference: FIXED_PRE_CONFIG "
        "| reference / baseline (median) |\n"
        "|---|---|---|---|---|---|\n"
    )
    return header + "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--m-values", type=int, nargs="+", default=M_VALUES)
    parser.add_argument("--output", help="also write the markdown table to this file")
    args = parser.parse_args()

    device = torch.cuda.get_device_name(0)
    table = to_markdown(run(args.m_values), device)
    print(table)
    if args.output:
        with open(args.output, "w") as f:
            f.write(table)


if __name__ == "__main__":
    main()
