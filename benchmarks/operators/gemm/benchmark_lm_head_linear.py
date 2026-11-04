# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""lm_head GEMM device time: batch-invariant linear vs aiter's own config (DSV4 RFC task 9, #510).

The GEMM is the lm_head of DeepSeek-V4-Flash-0731: x [M, 4096] bf16 (the final
RMSNorm output) times head.weight [129280, 4096] bf16, fp32 logits [M, 129280].
head.weight is BF16 in the checkpoint and x holds bf16 values, so bf16 inputs
with fp32 accumulation and output have the precision of the fp32 reference.

- baseline: aiter gemm_a16w16 with aiter's own config choice, which is not
  batch invariant;
- ours: aiter gemm_a16w16 with the tuned per-M table (--config-table, aiter
  M_LEQ_x JSON), called the way AiterBatchInvariantLinearOp calls it;
- references: FIXED_LINEAR_CONFIG (batch_invariant_linear without a table);
  the Triton det_gemm lm_head with fp32 output (TritonLMHeadOp.forward_fp32);
  torch.matmul (hipBLASLt, not batch invariant);
- comparison: ours / baseline, median device time.

Also checked per M: ours gives the same output bits as FIXED_LINEAR_CONFIG,
i.e. the tuned table keeps the parameters that change the bits.

Device time from GPU events: every case is measured in several independent
trials; the table gives the median and min-max over trials of the mean time
per call.

Usage (GPU; aiter with the Triton gemm_a16w16 and AITER_TRITON_ONLY=1):
    python benchmarks/operators/gemm/benchmark_lm_head_linear.py --config-table \\
        rl_engine/backends/rocm/configs/gfx942/gemm_a16w16/GEMM-A16W16-N=129280-K=4096.json \\
        --output artifacts/benchmarks/lm_head_linear.md
"""

import argparse
import json
import statistics

import torch

N, K = 129280, 4096  # DeepSeek-V4-Flash-0731 lm_head: vocab size, hidden size
# Bucket bounds plus M values between buckets (masked tile path, EVEN_MN false).
M_VALUES = [1, 16, 17, 64, 256, 1024, 3000, 4096, 8192]
WARMUP, TRIALS = 3, 5
# Calls per trial: enough for about TARGET_MS per trial, within [MIN_ITERS, MAX_ITERS].
TARGET_MS, MIN_ITERS, MAX_ITERS = 50.0, 3, 100


def _time_ms(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _measure(fn):
    """(median, min, max) over TRIALS, or the launch error as a string."""
    try:
        fn()  # compiles on first use
        torch.cuda.synchronize()
        one = _time_ms(fn, 1)
        iters = int(min(MAX_ITERS, max(MIN_ITERS, TARGET_MS / one)))
        for _ in range(WARMUP):
            fn()
        samples = [_time_ms(fn, iters) for _ in range(TRIALS)]
    except Exception as exc:  # a config can exceed LDS or registers
        return f"cannot launch: {type(exc).__name__}"
    return statistics.median(samples), min(samples), max(samples)


def _cell(result):
    if isinstance(result, str):
        return result
    median, low, high = result
    return f"{median:.4f} ({low:.4f}-{high:.4f})"


def _torch_matmul():
    """torch.mm with fp32 output if this torch supports out_dtype, else bf16 output."""
    a = torch.ones(16, 32, dtype=torch.bfloat16, device="cuda")
    try:
        torch.mm(a, a.t(), out_dtype=torch.float32)
        return (lambda x, w: torch.mm(x, w.t(), out_dtype=torch.float32)), "fp32 output"
    except (TypeError, RuntimeError):
        return (lambda x, w: torch.matmul(x, w.t())), "bf16 output"


def run(m_values, table):
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

    from rl_engine.backends.rocm.config_utils import select_entry
    from rl_engine.backends.rocm.gemm.aiter_gemm import FIXED_LINEAR_CONFIG
    from rl_engine.backends.shared.triton.embedding.lm_head import TritonLMHeadOp

    def aiter_linear(x, weight, config):
        if config is not None:
            config = dict(config, SPLITK_BLOCK_SIZE=K)  # NUM_KSPLIT=1: all of K
        return gemm_a16w16(x, weight, None, dtype=torch.float32, config=config, backend="triton")

    matmul, matmul_label = _torch_matmul()
    det_gemm_lm_head = TritonLMHeadOp()
    gen = torch.Generator(device="cuda").manual_seed(0)
    weight = (torch.randn(N, K, generator=gen, device="cuda") / K**0.5).to(torch.bfloat16)

    rows = []
    for m in m_values:
        x = torch.randn(m, K, generator=gen, device="cuda").to(torch.bfloat16)
        ours_config = select_entry(table, m)
        with torch.no_grad():
            baseline = _measure(lambda: aiter_linear(x, weight, None))
            ours = _measure(lambda: aiter_linear(x, weight, ours_config))
            fixed = _measure(lambda: aiter_linear(x, weight, FIXED_LINEAR_CONFIG))
            det_gemm = _measure(lambda: det_gemm_lm_head.forward_fp32(x, weight))
            torch_ref = _measure(lambda: matmul(x, weight))
            y_ours = aiter_linear(x, weight, ours_config)
            y_fixed = aiter_linear(x, weight, FIXED_LINEAR_CONFIG)
            differing = int((y_ours.view(torch.int32) != y_fixed.view(torch.int32)).sum())
        bits = "same" if differing == 0 else f"differs: {differing}/{y_ours.numel()}"
        if isinstance(baseline, str) or isinstance(ours, str):
            ratio = "n/a: a variant cannot launch"
        else:
            ratio = f"{ours[0] / baseline[0]:.3f}x"
        cells = [_cell(r) for r in (baseline, ours)] + [ratio]
        cells += [_cell(r) for r in (fixed, det_gemm, torch_ref)] + [bits]
        rows.append((m, cells))
        print(f"M={m}: " + " | ".join(cells), flush=True)
        x = y_ours = y_fixed = None  # free before the next M
        torch.cuda.empty_cache()
    return rows, matmul_label


def to_markdown(rows, device, table_path, matmul_label):
    header = (
        f"lm_head GEMM device time, ms per call, on {device}: x [M, {K}] bf16 times "
        f"weight [{N}, {K}] bf16, fp32 logits. Median over {TRIALS} trials (min-max). "
        f"Tuned table: `{table_path}`.\n\n"
        "| M | baseline: aiter's own config, not batch invariant | ours: tuned table "
        "| ours / baseline (median) | reference: FIXED_LINEAR_CONFIG "
        "| reference: Triton det_gemm lm_head, fp32 output "
        f"| reference: torch.matmul, hipBLASLt, {matmul_label}, not batch invariant "
        "| ours bits vs FIXED_LINEAR_CONFIG |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    return header + "\n".join(f"| {m} | " + " | ".join(c) + " |" for m, c in rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-table", required=True, help="tuned aiter M_LEQ_x JSON")
    parser.add_argument("--m-values", type=int, nargs="+", default=M_VALUES)
    parser.add_argument("--output", help="also write the markdown table to this file")
    args = parser.parse_args()

    with open(args.config_table) as f:
        table = json.load(f)
    device = torch.cuda.get_device_name(0)
    rows, matmul_label = run(args.m_values, table)
    report = to_markdown(rows, device, args.config_table, matmul_label)
    print(report)
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)


if __name__ == "__main__":
    main()
