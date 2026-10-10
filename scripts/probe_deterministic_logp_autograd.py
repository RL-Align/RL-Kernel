#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Capture native forward/VJP/double/triple-backward provenance on the selected GPU."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from rl_engine import _C  # noqa: E402
from rl_engine.kernels.ops.cuda.loss.logp import DeterministicLogpCUDAOp  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
        "maximum_derivative_order": 3,
        "cases": [],
    }
    op = DeterministicLogpCUDAOp()
    # Initialize CUPTI before recording evidence.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
        torch.ones(16, device="cuda").sum().item()
        torch.cuda.synchronize()
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        for indexed, out in ((False, False), (True, False), (False, True), (True, True)):
            x = torch.randn(4, 257, device="cuda", dtype=dtype, requires_grad=True)
            ids = torch.tensor([0, 17, 129, 256], device="cuda")
            rows = torch.tensor([0, 2, 2, -1, 7], device="cuda")
            acc = torch.float64 if dtype == torch.float64 else torch.float32
            g = torch.ones(4, device="cuda", dtype=acc, requires_grad=True)
            h = torch.randn_like(x, requires_grad=True)
            a, b = torch.randn_like(x), torch.randn_like(g)
            buffer = torch.empty(4, device="cuda", dtype=acc)
            if out:
                name = ("indexed_out" if indexed else "out") + (
                    "_fp64" if dtype == torch.float64 else ""
                )
                inputs = (x, ids, rows, buffer) if indexed else (x, ids, buffer)
            else:
                name = ("indexed_" if indexed else "apply_") + (
                    "fp64" if dtype == torch.float64 else "fp32"
                )
                inputs = (x, ids, rows) if indexed else (x, ids)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                y = getattr(op, name)(*inputs)
                grad = torch.autograd.grad(y, x, g, create_graph=True)[0]
                dx, dg = torch.autograd.grad(grad, (x, g), h, create_graph=True)
                tx, tg, th = torch.autograd.grad((dx, dg), (x, g, h), (a, b))
                torch.cuda.synchronize()
            kernels = sorted(
                {
                    event.name
                    for event in trace.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                }
            )
            expected = (
                [
                    "deterministic_logp_forward_fp64_kernel",
                    "deterministic_logp_backward_fp64_kernel",
                ]
                if dtype == torch.float64
                else ["deterministic_logp_forward_kernel", "deterministic_logp_backward_kernel"]
            )
            expected.extend(
                [
                    "deterministic_logp_double_backward_kernel",
                    "deterministic_logp_triple_backward_kernel",
                ]
            )
            missing = [name for name in expected if not any(name in k for k in kernels)]
            forbidden = [
                k
                for k in kernels
                if "softmax" in k.lower()
                or "triton" in k.lower()
                or "batch_invariant_logp" in k.lower()
            ]
            valid = (
                not missing
                and not forbidden
                and all(bool(torch.isfinite(t).all()) for t in (dx, dg, tx, tg, th))
            )
            report["cases"].append(
                {
                    "dtype": str(dtype),
                    "method": name,
                    "cuda_kernels": kernels,
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
        f"Verified {len(report['cases'])} native forward/backward/double/triple-backward traces: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
