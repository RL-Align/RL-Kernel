#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Sweep the WS1 C4 gradient gate over every adapter x required profile.

Runs ``check_gradient_invariance.py`` once per (profile, adapter) cell using the
C2-declared candidate, and prints the closeout evidence table. Each cell is
classified, so a red never hides behind a traceback:

``green``            the cell passed
``red_verdict``      a named gradient failed a C1 judgment
``red_no_backward``  a required differentiable node has no VJP
``blocked_hardware`` the declared candidate needs a GPU this box does not have
``blocked_c2``       C2 marks the node ``missing_required``
``skipped``          no C2 node to run (optional / profile-independent)

Exit code is non-zero when any cell is red or blocked, so it is safe to gate on.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.gtest.gradient_adapters import (  # noqa: E402
    GRADIENT_ADAPTERS,
    resolve_profile_candidate,
)
from rl_engine.testing.ws1_workload import load_manifest  # noqa: E402

GATE = REPO_ROOT / "scripts" / "check_gradient_invariance.py"
PROFILES = ("cuda_bf16", "triton_cuda_bf16", "ascend_bf16")


@dataclass
class CellResult:
    profile: str
    op_name: str
    candidate: str | None
    status: str
    detail: str
    failing_tensors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "op_name": self.op_name,
            "candidate": self.candidate,
            "status": self.status,
            "detail": self.detail,
            "failing_tensors": list(self.failing_tensors),
        }


def _classify(returncode: int, output: str) -> tuple[str, str, tuple[str, ...]]:
    if returncode == 0:
        return "green", "", ()
    if "has no backward" in output:
        return "red_no_backward", "candidate is not wired through torch.autograd", ()
    if "fallback forbidden" in output or "is not compiled" in output:
        return "blocked_hardware", "declared candidate needs a Hopper build", ()
    if "missing_required" in output:
        return "blocked_c2", "C2 marks this node missing_required", ()
    if "layout_supported" in output:
        return "skipped", "profile-independent; covered by the CPU contract test", ()
    tensors = tuple(
        sorted(
            {
                line.split("tensor=", 1)[1].split()[0]
                for line in output.splitlines()
                if "passed=False" in line and "tensor=" in line
            }
        )
    )
    if tensors:
        return "red_verdict", f"failed C1 judgment for {', '.join(tensors)}", tensors
    tail = next(
        (
            line
            for line in reversed(output.splitlines())
            if line.strip() and not line.startswith("INFO")
        ),
        "unknown failure",
    )
    return "red_verdict", tail.strip()[:200], ()


def _run_cell(profile: str, op_name: str, extra: list[str]) -> CellResult:
    adapter = GRADIENT_ADAPTERS[op_name]
    manifest = load_manifest()
    resolved = resolve_profile_candidate(adapter, profile, manifest)
    candidate = resolved["expected_backend_id"]
    if candidate is None:
        reason = {
            "missing_required": ("blocked_c2", "C2 marks this node missing_required"),
            "optional": ("skipped", "optional_fused with no C2 node"),
            "absent_not_required": ("skipped", "not declared supported and differentiable"),
        }.get(str(resolved["status"]), ("skipped", str(resolved["status"])))
        return CellResult(profile, op_name, None, reason[0], reason[1])

    proc = subprocess.run(
        [
            sys.executable,
            str(GATE),
            "--op",
            op_name,
            "--candidate",
            str(candidate),
            "--backend-profile",
            profile,
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    output = proc.stdout + proc.stderr
    status, detail, tensors = _classify(proc.returncode, output)
    return CellResult(profile, op_name, str(candidate), status, detail, tensors)


def main() -> None:
    parser = argparse.ArgumentParser(description="WS1 C4 gradient gate sweep")
    parser.add_argument("--profile", choices=PROFILES, action="append")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--vocab", type=int)
    parser.add_argument("--head-dim", type=int)
    args = parser.parse_args()

    extra: list[str] = []
    for flag, value in (
        ("--hidden", args.hidden),
        ("--vocab", args.vocab),
        ("--head-dim", args.head_dim),
    ):
        if value is not None:
            extra += [flag, str(value)]

    profiles = tuple(args.profile) if args.profile else PROFILES
    results = [
        _run_cell(profile, op_name, extra)
        for profile in profiles
        for op_name, adapter in GRADIENT_ADAPTERS.items()
        if adapter.requirement != "absent_not_required"
    ]

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for result in results:
            print(
                f"{result.profile:<17} {result.op_name:<21} "
                f"{result.candidate or '-':<11} {result.status:<17} {result.detail}"
            )
        counts: dict[str, int] = {}
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1
        print("\n" + ", ".join(f"{status}={n}" for status, n in sorted(counts.items())))

    if any(r.status != "green" for r in results if r.status != "skipped"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
