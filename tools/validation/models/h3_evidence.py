#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Write one RFC #420 operator's evidence report (performance + accuracy) as JSON.

Run it on a clean tree; the report records the commit and whether the tree
was dirty. ``tools/validation/models/plot_h3_evidence.py`` turns the report into a figure.

    export RL_KERNEL_H3_WEIGHTS=<dir written by tools/weights/prepare_h3_weights.py>
    python tools/validation/models/h3_evidence.py --op timestep_sinusoid_h3 \\
        --out reports/experiments/h3-timestep-sinusoid-b200/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from rl_engine.runtime.registry import KernelRegistry  # noqa: E402
from rl_engine.validation.models.h3_chain import environment, git_state  # noqa: E402
from rl_engine.validation.models.h3_report import (  # noqa: E402
    ACCURACY,
    PERF_CASES,
    h3_weights_source,
    measure,
)
from rl_engine.validation.models.h3_weights import load_h3_manifest  # noqa: E402

_NEEDS_PINNED_MLP = (
    "timestep_mlp_fp32 evidence needs the pinned time_embedder weights; set "
    "RL_KERNEL_H3_WEIGHTS (tools/weights/prepare_h3_weights.py) "
    "or pass --allow-random-weights"
)


def measure_case(case: dict) -> dict:
    """Time one perf case and keep the weight source the case was built with next to the timings."""

    row = measure(case)
    if "weights" in case:
        row["weights"] = case["weights"]
    return row


def check_weight_source(op: str, report: dict, allow_random: bool) -> None:
    """Refuse a timestep_mlp_fp32 report that was measured without the pinned checkpoint."""

    if (
        op == "timestep_mlp_fp32"
        and report["accuracy"]["weights_source"] != "pinned"
        and not allow_random
    ):
        raise SystemExit(_NEEDS_PINNED_MLP)


def require_pinned_weights(op: str, allow_random: bool) -> None:
    """Stop before any measurement when timestep_mlp_fp32 would run on random weights."""

    if op == "timestep_mlp_fp32" and h3_weights_source() != "pinned" and not allow_random:
        raise SystemExit(_NEEDS_PINNED_MLP)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True, choices=sorted(ACCURACY))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-random-weights",
        action="store_true",
        help="write a timestep_mlp_fp32 report even without the pinned checkpoint "
        "(smoke runs only)",
    )
    args = parser.parse_args()
    require_pinned_weights(args.op, args.allow_random_weights)
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    registry = KernelRegistry()
    manifest = load_h3_manifest()
    report = {
        "kind": "h3_operator_report",
        "op": args.op,
        "rfc": manifest["rfc"],
        "model_revision": manifest["model_identity"]["revision"],
        "reference_commit": manifest["reference_implementation"]["commit"],
        **git_state(),
        "environment": environment(),
        "accuracy": ACCURACY[args.op](registry),
        "perf": [measure_case(case) for case in PERF_CASES[args.op](registry)],
    }
    check_weight_source(args.op, report, args.allow_random_weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"wrote {args.out} (commit {report['rl_kernel_commit'][:7]}, "
        f"dirty={report['tracked_tree_dirty']})"
    )


if __name__ == "__main__":
    main()
