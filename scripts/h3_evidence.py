#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Write one RFC #420 operator's evidence report (performance + accuracy) as JSON.

Run it on a clean tree; the report records the commit and whether the tree
was dirty. ``scripts/plot_h3_evidence.py`` turns the report into a figure.

    export RL_KERNEL_H3_WEIGHTS=<dir written by scripts/prepare_h3_weights.py>
    python scripts/h3_evidence.py --op timestep_sinusoid_h3 \\
        --out docs/usage/evidence/h3-timestep-sinusoid-b200/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rl_engine.kernels.registry import KernelRegistry  # noqa: E402
from rl_engine.testing.h3_chain import environment, git_state  # noqa: E402
from rl_engine.testing.h3_report import ACCURACY, PERF_CASES, measure  # noqa: E402
from rl_engine.testing.h3_weights import load_h3_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True, choices=sorted(ACCURACY))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
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
        "perf": [measure(case) for case in PERF_CASES[args.op](registry)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"wrote {args.out} (commit {report['rl_kernel_commit'][:7]}, "
        f"dirty={report['tracked_tree_dirty']})"
    )


if __name__ == "__main__":
    main()
