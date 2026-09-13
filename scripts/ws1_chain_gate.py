#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 C10/C11: full Qwen3-8B Dense model-level chain gate."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import subprocess
import sys

if "--json" in sys.argv:
    os.environ["RL_KERNEL_LOG_STREAM"] = "stderr"

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.gtest.accelerator import (  # noqa: E402
    disable_tf32,
    resolve_device,
)
from rl_engine.kernels.gtest.chain_gate import (  # noqa: E402
    build_model,
    run_chain_gate,
    run_fp32_reference_cell,
)
from rl_engine.kernels.gtest.tolerance import load_contract  # noqa: E402
from rl_engine.testing.ws1_workload import load_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WS1 C10/C11 full-model chain gate")
    parser.add_argument(
        "--backend-profile",
        choices=("cuda_bf16", "triton_cuda_bf16", "ascend_bf16"),
        required=True,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Defaults to the backend profile's own accelerator (cuda or npu).",
    )
    parser.add_argument("--model", default="qwen3-8b-dense", choices=("qwen3-8b-dense",))
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16",))
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Execution RNG seed; defaults to the canonical manifest seed.",
    )
    parser.add_argument("--weights", choices=("required", "hf", "synthetic"), default="required")
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def _file_sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    try:
        device = resolve_device(args.device, profile=args.backend_profile)
    except RuntimeError as exc:
        print(
            f"ERROR: C10/C11 full-model gate needs a real device; "
            f"CPU-only is not a pass: {exc}",
            file=sys.stderr,
        )
        return 2
    if args.weights == "synthetic":
        print(
            "ERROR: synthetic weights cannot close C10/C11; use --weights required|hf",
            file=sys.stderr,
        )
        return 2
    disable_tf32(device.type)
    manifest = load_manifest()
    contract = load_contract()
    execution_seed = manifest.seed if args.seed is None else int(args.seed)
    log_stream = sys.stderr if args.json else sys.stdout
    with contextlib.redirect_stdout(log_stream):
        reference_cell = run_fp32_reference_cell(
            backend_profile=args.backend_profile,
            weights_mode="hf",
            weights_path=args.weights_path,
            device=device,
            manifest=manifest,
            run_backward=True,
        )
        model = build_model(
            backend_profile=args.backend_profile,
            weights_mode="hf",
            weights_path=args.weights_path,
            device=device,
            dtype=torch.bfloat16,
            manifest=manifest,
        )
        report = run_chain_gate(
            backend_profile=args.backend_profile,
            model=model,
            contract=contract,
            manifest=manifest,
            run_backward=True,
            run_train_infer=True,
            execution_seed=execution_seed,
            reference_cell=reference_cell,
        )
    payload = report.to_dict()
    payload.update(
        {
            "schema_version": "ws1-c10-c11-v5",
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
            "contract_sha256": _file_sha(
                REPO_ROOT / "rl_engine/kernels/gtest/tolerance_contract.json"
            ),
            "manifest_sha256": _file_sha(REPO_ROOT / "rl_engine/testing/ws1_manifest.json"),
            "cli": {
                "backend_profile": args.backend_profile,
                "model": args.model,
                "dtype": args.dtype,
                "seed": execution_seed,
            },
        }
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"C10 profile={report.backend_profile} passed={report.passed} "
            f"first_drift={report.first_drift} weights={report.weight_source}"
        )
        print(
            f"  gradient_scope={report.gradient_scope} "
            f"all_parameter_gradients={report.all_parameter_gradients} "
            f"names={','.join(sorted(report.required_grad_names))}"
        )
        for item in report.invariance:
            print(
                f"  inv {item.config_pair} max_abs={item.max_abs_error:.8e} "
                f"atol={item.atol} passed={item.passed}"
            )
        if report.aggregates is not None:
            print(f"  aggregates passed={report.aggregates.passed}")
        if report.train_infer is not None:
            print(f"  train_infer passed={report.train_infer.passed}")
        if report.train_infer_bn is not None:
            print(f"  train_infer_bn passed={report.train_infer_bn.passed}")
        for case_id, verdict in report.decode_prefill:
            print(f"  decode_prefill {case_id} passed={verdict.passed}")
        for item in report.accuracy_aggregates:
            print(f"  acc_agg kind={item.report_kind} passed={item.passed}")
        for item in report.accuracy:
            print(
                f"  acc {item.config_pair} max_abs={item.max_abs_error:.8e} "
                f"passed={item.passed}"
            )
        print(report.disclaimer)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
