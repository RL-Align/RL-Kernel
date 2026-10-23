#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Capture native fused training, recursive AD and torch.func provenance on the selected GPU."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from rl_engine import _C  # noqa: E402
from rl_engine.backends.cuda.logprob.logp import DeterministicLogpCUDAOp  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-order", type=int, default=8)
    args = parser.parse_args()
    if args.max_order < 2:
        parser.error("--max-order must be at least 2 for the recursive-path probe")
    extension = Path(_C.__file__).resolve()
    if extension.parent != SOURCE / "rl_engine":
        raise RuntimeError(f"Wrong extension loaded: {extension}")
    report = {
        "source_commit": subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(SOURCE), "status", "--porcelain"], text=True
            ).strip()
        ),
        "extension": str(extension),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "backend": "DeterministicLogpCUDAOp",
        "tested_derivative_order": args.max_order,
        "algorithm_order_limit": None,
        "cases": [],
    }
    op = DeterministicLogpCUDAOp()
    # Initialize CUPTI before recording evidence.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
        torch.ones(16, device="cuda").sum().item()
        torch.cuda.synchronize()
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        for indexed, out in ((False, False), (True, False), (False, True), (True, True)):
            for mode in ("fused_first_backward", "recursive", "torch_func"):
                x = torch.randn(4, 17, device="cuda", dtype=dtype, requires_grad=True)
                ids = torch.tensor([0, 3, 9, 16], device="cuda")
                rows = torch.tensor([0, 2, 2, -1, 7], device="cuda")
                acc = torch.float64 if dtype == torch.float64 else torch.float32
                g = torch.ones(4, device="cuda", dtype=acc)
                h = torch.randn_like(x) * 0.2
                if out:
                    name = ("indexed_out" if indexed else "out") + (
                        "_fp64" if dtype == torch.float64 else ""
                    )
                else:
                    name = ("indexed_" if indexed else "apply_") + (
                        "fp64" if dtype == torch.float64 else "fp32"
                    )

                def fn(z):
                    inputs = [z, ids]
                    if indexed:
                        inputs.append(rows)
                    if out:
                        inputs.append(z.new_zeros(ids.shape, dtype=acc))
                    return getattr(op, name)(*inputs)

                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                    if mode == "fused_first_backward":
                        result = torch.autograd.grad(fn(x), x, g)[0]
                    elif mode == "recursive":
                        scalar = (fn(x) * g).sum()
                        for order in range(args.max_order):
                            result = torch.autograd.grad(
                                scalar, x, create_graph=order + 1 < args.max_order
                            )[0]
                            scalar = (result * h).sum()
                    else:
                        result = torch.func.jvp(torch.func.grad(lambda z: fn(z).sum()), (x,), (h,))[
                            1
                        ]
                    torch.cuda.synchronize()
                kernels = sorted(
                    {
                        e.name
                        for e in trace.events()
                        if e.device_type == torch.autograd.DeviceType.CUDA
                    }
                )
                expected = [
                    (
                        "deterministic_logp_forward_fp64_kernel"
                        if dtype == torch.float64
                        else "deterministic_logp_forward_kernel"
                    )
                ]
                if mode == "fused_first_backward":
                    expected.append(
                        "deterministic_logp_backward_fp64_kernel"
                        if dtype == torch.float64
                        else "deterministic_logp_backward_kernel"
                    )
                else:
                    expected.extend(
                        [
                            "deterministic_logp_probability_kernel",
                            "deterministic_logp_row_sum_kernel",
                        ]
                    )
                missing = [k for k in expected if not any(k in actual for actual in kernels)]
                forbidden = [
                    k
                    for k in kernels
                    if any(
                        v in k.lower()
                        for v in (
                            "softmax",
                            "triton",
                            "batch_invariant_logp",
                            "deterministic_logp_double_backward_kernel",
                            "deterministic_logp_triple_backward_kernel",
                        )
                    )
                ]
                valid = not missing and not forbidden and bool(torch.isfinite(result).all())
                report["cases"].append(
                    {
                        "dtype": str(dtype),
                        "method": name,
                        "mode": mode,
                        "tested_order": (
                            args.max_order
                            if mode == "recursive"
                            else (2 if mode == "torch_func" else 1)
                        ),
                        "cuda_kernels": kernels,
                        "custom_cuda_kernels": [k for k in kernels if "deterministic_logp_" in k],
                        "aten_and_test_harness_kernels": [
                            k for k in kernels if "deterministic_logp_" not in k
                        ],
                        "missing": missing,
                        "forbidden": forbidden,
                        "status": "passed" if valid else "failed",
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(case["status"] != "passed" for case in report["cases"]):
        raise RuntimeError(f"Native backend verification failed; see {args.output}")
    print(
        f"Verified {len(report['cases'])} fused/recursive/torch.func CUDA traces: " f"{args.output}"
    )


if __name__ == "__main__":
    main()
