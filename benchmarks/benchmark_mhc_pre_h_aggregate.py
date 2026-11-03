#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Compare explicit MHC forward/backward latency and peak allocated memory."""

import argparse
import pathlib
import statistics
import sys
from functools import partial

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rl_engine.kernels.ops.cuda.mhc import MHCPreHAggregateCudaOp  # noqa: E402
from rl_engine.kernels.ops.pytorch.mhc import NativeMHCPreHAggregateOp  # noqa: E402
from rl_engine.kernels.ops.triton.mhc import MHCPreHAggregateTritonOp  # noqa: E402
from rl_engine.mhc import oracle  # noqa: E402


def assert_bytes(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8)), "P1 oracle bytes differ"


def measure(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)
        del result
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    extra_mib = (torch.cuda.max_memory_allocated() - baseline) / 2**20
    del result
    return statistics.median(times), extra_mib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 129, 1024])
    parser.add_argument("--hidden", type=int, choices=(128, 4096), default=4096)
    parser.add_argument("--grad-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    if min(args.tokens) <= 0 or args.warmup < 0 or args.repeat <= 0:
        parser.error("tokens and repeat must be positive; warmup must be nonnegative")
    ops = {"PyTorch": NativeMHCPreHAggregateOp(), "CUDA": MHCPreHAggregateCudaOp(),
           "Triton": MHCPreHAggregateTritonOp()}
    grad_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.grad_dtype]
    print(f"GPU: {torch.cuda.get_device_name()}; PyTorch: {torch.__version__}; "
          f"CUDA: {torch.version.cuda}; D={args.hidden}; dH={args.grad_dtype}")
    print(f"Median CUDA-event latency; warmup={args.warmup}, repeat={args.repeat}. "
          "Extra peak allocated memory includes outputs and temporaries.")
    print("| T | Backend | Forward (us) | Backward (us) | Forward extra (MiB) | "
          "Backward extra (MiB) | Oracle bytes |")
    print("|---:|:---|---:|---:|---:|---:|:---|")
    torch.manual_seed(366)
    with torch.no_grad():
        for tokens in args.tokens:
            residual = torch.randn(tokens, 4, args.hidden, device="cuda", dtype=torch.bfloat16)
            pre = torch.randn(tokens, 4, device="cuda")
            dh = torch.randn(tokens, args.hidden, device="cuda", dtype=grad_dtype)
            expected = oracle.h_aggregate_fwd(pre, residual)
            expected_dr, expected_dp = oracle.h_aggregate_bwd(dh, pre, residual)
            for name, op in ops.items():
                assert_bytes(op(residual, pre), expected)
                dr, dp = op.backward_fp32(dh, residual, pre)
                assert_bytes(dr, expected_dr)
                assert_bytes(dp, expected_dp)
                del dr, dp
                forward_us, forward_mem = measure(
                    partial(op, residual, pre), args.warmup, args.repeat)
                backward_us, backward_mem = measure(
                    partial(op.backward_fp32, dh, residual, pre), args.warmup, args.repeat)
                print(f"| {tokens} | {name} | {forward_us:.2f} | {backward_us:.2f} | "
                      f"{forward_mem:.3f} | {backward_mem:.3f} | PASS |", flush=True)


if __name__ == "__main__":
    main()
