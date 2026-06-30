#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.gtest import run_operator_suite  # noqa: E402
from rl_engine.kernels.gtest.operator_specs import (  # noqa: E402
    make_candidate,
    make_operator_case,
    operator_names,
)


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"unsupported dtype: {value}")


def _select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return device


def _summarize(report: Any) -> None:
    print(f"suite={report.suite_name} passed={report.passed} pass_rate={report.pass_rate:.4f}")
    for candidate in report.candidates:
        print(
            f"candidate={candidate.candidate_name} backend={candidate.backend} "
            f"passed={candidate.passed} pass_rate={candidate.pass_rate:.4f}"
        )
        for case in candidate.cases:
            for output in case.outputs:
                print(
                    f"  case={case.case_name} output={output.output_index} "
                    f"shape={output.shape} dtype={output.candidate_dtype} "
                    f"max_abs={output.max_abs_error:.8e} "
                    f"mean_abs={output.mean_abs_error:.8e} "
                    f"max_rel={output.max_rel_error:.8e} "
                    f"tol=(atol={output.atol:.3e}, rtol={output.rtol:.3e}) "
                    f"passed={output.passed}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an operator candidate against a PyTorch gold path.")
    parser.add_argument("--op", choices=operator_names(), default="logp")
    parser.add_argument(
        "--candidate",
        default="registry",
        help="Candidate backend to validate, for example registry, pytorch, cuda, cuda-sm90.",
    )
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--vocab", type=int, default=257)
    parser.add_argument("--input-mode", choices=("random", "constant"), default="random")
    parser.add_argument("--constant-value", type=float, default=0.25)
    parser.add_argument("--token-value", type=int, default=0)
    parser.add_argument("--normalized-dim", type=int, default=4096)
    parser.add_argument("--k-dim", type=int, default=4096)
    parser.add_argument("--n-dim", type=int, default=4096)
    parser.add_argument("--theta", type=float, default=1.0e6)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--arch-key",
        default=None,
        help="Optional tolerance override key, for example sm90. Defaults to contract.default.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full structured report as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = _parse_dtype(args.dtype)
    device = _select_device(args.device)
    candidate = make_candidate(args)
    case = make_operator_case(args, dtype, device)
    report = run_operator_suite(args.op, candidates=[candidate], cases=[case])

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        _summarize(report)

    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
