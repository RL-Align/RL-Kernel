#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Verify and compare sealed MiniMax-H3 trajectory artifacts.

    python tools/validation/models/h3_trajectory.py verify run.json
    python tools/validation/models/h3_trajectory.py compare a.json b.json \\
        [--tensors-a a.safetensors --tensors-b b.safetensors] [--out report.json] [--require-equal]

``compare`` prints the first drift (step, kind, modality, and with tensor files the
mismatch count and max abs difference) and exits 1 with ``--require-equal`` when the two
runs are not bitwise identical, so it can gate a strict claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from rl_engine.models.minimax_h3 import trajectory as tj  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("artifact", type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("lhs", type=Path)
    compare.add_argument("rhs", type=Path)
    compare.add_argument("--tensors-a", type=Path)
    compare.add_argument("--tensors-b", type=Path)
    compare.add_argument("--out", type=Path)
    compare.add_argument("--require-equal", action="store_true")
    args = parser.parse_args()

    if args.command == "verify":
        artifact = tj.loads(args.artifact.read_text())
        print(
            f"ok: {artifact['label']} {artifact['variant']} seed={artifact['seed']} "
            f"steps={artifact['steps']} sha256={artifact['artifact_sha256']}"
        )
        return 0
    lhs = tj.loads(args.lhs.read_text())
    rhs = tj.loads(args.rhs.read_text())
    report = tj.drift_report(lhs, rhs, tensors_lhs=args.tensors_a, tensors_rhs=args.tensors_b)
    text = json.dumps(report, indent=1, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    summary = {k: report[k] for k in ("labels", "steps", "compared", "differing", "bitwise_equal")}
    summary["first_drift"] = report["first_drift"]
    print(json.dumps(summary, indent=1))
    return 1 if (args.require_equal and not report["bitwise_equal"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
