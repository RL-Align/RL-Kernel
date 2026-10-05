#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""T06 local verification runner.

CPU oracle + negatives + recorded block always run.
CUDA: reuses validated native kernels or JIT-refreshes T06 attention, retaining
native DetGemm when available; otherwise JIT-compiles both.
The CUDA phase acquires /tmp/rl-kernel-t06-gpu.lock internally; do not wrap
this runner in another flock for that same path.

This is recorded-operator verification, not a live DSV4 / Qwen3 training run.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_pytest(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=line", *args]
    print("+", " ".join(cmd), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    return subprocess.call(cmd, cwd=ROOT, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report: dict = {
        "task_id": "T06",
        "contract_version": "p2-task-contract.v1",
        "scope": "recorded operators; not live DSV4/Qwen3 training",
        "verdicts": {},
    }

    cpu_args = [
        "tests/p2",
        "--ignore=tests/p2/test_mqa_joint_attention_cuda.py",
    ]
    cpu_rc = _run_pytest(cpu_args)
    report["verdicts"]["cpu_pytest"] = "PASS" if cpu_rc == 0 else "FAIL"
    if cpu_rc != 0:
        _write(report, args.json_out)
        return cpu_rc

    if args.skip_cuda:
        report["verdicts"]["cuda"] = "SKIP"
        _write(report, args.json_out)
        print(json.dumps(report, indent=2))
        return 0

    import torch

    from rl_engine.kernels.p2.cuda_runtime import ensure_t06_cuda_kernel

    if not torch.cuda.is_available():
        report["verdicts"]["cuda"] = "SKIP"
        report["verdicts"]["cuda_reason"] = "no GPU"
    else:
        try:
            with open("/tmp/rl-kernel-t06-gpu.lock", "a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                report["verdicts"]["cuda_source"] = ensure_t06_cuda_kernel()
                import pytest

                # Keep JIT injection visible; cover attention, o-proj and the
                # actual four-mode chain, including captured backward.
                cuda_rc = pytest.main([
                    "-q", "--tb=line",
                    str(ROOT / "tests/p2/test_mqa_joint_attention_cuda.py"),
                    str(ROOT / "tests/p2/test_mqa_joint_attention_nonfinite.py"),
                    str(ROOT / "tests/p2/test_o_proj_det_gemm.py"),
                    str(ROOT / "tests/p2/test_four_mode.py"),
                    str(ROOT / "tests/p2/test_mqa_joint_attention_negative.py"),
                ])
            report["verdicts"]["cuda_pytest"] = "PASS" if cuda_rc == 0 else "FAIL"
            report["verdicts"]["cuda"] = "PASS" if cuda_rc == 0 else "FAIL"
            if cuda_rc != 0:
                _write(report, args.json_out)
                return cuda_rc
        except Exception as exc:
            report["verdicts"]["cuda"] = "FAIL"
            report["verdicts"]["cuda_error"] = f"{type(exc).__name__}: {exc}"
            _write(report, args.json_out)
            print(json.dumps(report, indent=2))
            return 1

    _write(report, args.json_out)
    print(json.dumps(report, indent=2), flush=True)
    print("T06 verification", report["verdicts"], flush=True)
    return 0


def _write(report: dict, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
