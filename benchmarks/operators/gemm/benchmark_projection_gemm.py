# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Bounded single-GPU Task 12 before/after timing; run as a module."""

import argparse
import hashlib
import json
import os
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch

from rl_engine.backends.cuda.gemm.det_gemm import DetGemmOp, det_gemm_backend
from rl_engine.backends.extension import _C

if __package__:
    from benchmarks.operators.gemm.projection_provenance import source_provenance
else:
    # The before/after harness also runs this file from a baseline checkout.
    from projection_provenance import source_provenance

SHAPES = (
    (1, 2048, 512),
    (129, 2048, 512),
    (129, 4096, 2048),
    (129, 2048, 32),
    (129, 2048, 1),
    (1, 2048, 248320),
)


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    torch.cuda.reset_peak_memory_stats()
    for start, end in pairs:
        start.record()
        output = fn()
        end.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in pairs]
    return {
        "dtype": str(output.dtype),
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[min(repeats - 1, int(repeats * 0.95))],
        "min_ms": min(samples),
        "max_ms": max(samples),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "samples_ms": samples,
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2:
        parser.error("warmup must be positive and repeats at least two")
    assert det_gemm_backend() == "sm90", "set RL_KERNEL_DET_GEMM_BACKEND=sm90"
    assert torch.cuda.device_count() == 1, "select exactly one idle GPU with CUDA_VISIBLE_DEVICES"
    assert torch.cuda.get_device_capability()[0] == 9, "Hopper benchmark"
    op = DetGemmOp()
    extension = Path(_C.__file__)
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        **source_provenance(),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "backend": det_gemm_backend(),
        "sm90_compiled": _C.det_gemm_sm90_compiled(),
        "method": (
            "CUDA events around public API, includes copies/allocations/kernels; no CUDA graphs"
        ),
        "seed": 642,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": [],
    }
    torch.manual_seed(report["seed"])
    torch.set_num_threads(4)
    for m, k, n in SHAPES:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        b = weight.t().contiguous()
        # Widening the BF16 result is an output-equivalent comparison for the
        # repaired API. The baseline forward_fp32 dtype is deliberately recorded.
        operations = {
            "linear_bf16": lambda a=a, weight=weight: op.linear(a, weight),
            "tree_widen_reference": lambda a=a, b=b: op(a, b).float(),
        }
        if hasattr(_C, "det_gemm_fwd_fp32"):
            operations["forward_fp32"] = lambda a=a, b=b: op.forward_fp32(a, b)
        else:
            report["cases"].append(
                {
                    "shape": [m, k, n],
                    "api": "forward_fp32",
                    "status": "binding_unavailable",
                }
            )
        for name, fn in operations.items():
            case = {
                "shape": [m, k, n],
                "api": name,
                **measure(fn, args.warmup, args.repeats),
            }
            report["cases"].append(case)
            print(
                json.dumps({key: value for key, value in case.items() if key != "samples_ms"}),
                flush=True,
            )
        del operations, a, weight, b
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
