#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Sweep the one-block backward over layouts and seeds and record leaf errors in ulps.

Backs the ulp allowance of ``tests/models/minimax_h3/test_h3_one_block.py``:
for every leaf gradient it records the candidate's and diffusers' error against
the FP32 golden both as ``max|err| / max|golden|`` and in ulps of the leaf's
storage dtype at ``max|golden|``, and the per-leaf excess
``candidate_ulps - provider_ulps``.

    export RL_KERNEL_H3_WEIGHTS=<dir written by prepare_h3_weights.py --block>
    python tools/validation/models/h3_block_ulp_sweep.py --layouts tiny,small --seeds 0,1,2,3,4 \\
        --out ulp_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from rl_engine.validation.models import h3_block, h3_chain  # noqa: E402
from rl_engine.validation.models.h3_weights import load_h3_block_weights  # noqa: E402


def main() -> None:
    """Run the backward replay for every (layout, seed) and write per-leaf ulp errors."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", default="tiny,small")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("the sweep needs a CUDA device")

    weights = load_h3_block_weights("cuda")
    ops = h3_block.CandidateOps()
    cases = []
    for layout in args.layouts.split(","):
        for seed in (int(s) for s in args.seeds.split(",")):
            report = h3_block.run_block_backward(weights, layout=layout, seed=seed, ops=ops)
            leaves = {
                name: {
                    key: entry[key]
                    for key in (
                        "storage_dtype",
                        "candidate_rel_err_vs_golden",
                        "provider_rel_err_vs_golden",
                        "candidate_ulps_vs_golden",
                        "provider_ulps_vs_golden",
                    )
                }
                | {
                    "excess_ulps": entry["candidate_ulps_vs_golden"]
                    - entry["provider_ulps_vs_golden"]
                }
                for name, entry in report["leaves"].items()
            }
            worst = max(leaves, key=lambda n: leaves[n]["excess_ulps"])
            print(
                f"{layout} S={report['seq_len']} seed={seed}: worst excess "
                f"{leaves[worst]['excess_ulps']:+.2f} ulp ({worst})"
            )
            cases.append(
                {"layout": layout, "seq_len": report["seq_len"], "seed": seed, "leaves": leaves}
            )
            torch.cuda.empty_cache()

    bf16 = [
        leaf["excess_ulps"]
        for case in cases
        for leaf in case["leaves"].values()
        if leaf["storage_dtype"] == "bfloat16"
    ]
    evidence = {
        "kind": "h3_one_block_backward_ulp_sweep",
        **h3_chain.git_state(),
        "environment": h3_chain.environment(),
        "cases": cases,
        "max_bf16_excess_ulps": max(bf16),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"max BF16 excess: {max(bf16):+.2f} ulp; wrote {args.out}")


if __name__ == "__main__":
    main()
