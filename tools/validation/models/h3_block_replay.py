#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Replay one real MiniMax-H3 block forward and backward and write JSON evidence (RFC #420).

See ``rl_engine/validation/models/h3_block.py`` for what is compared.

    python tools/weights/prepare_h3_weights.py --out <dir> --block
    export RL_KERNEL_H3_WEIGHTS=<dir>
    python tools/validation/models/h3_block_replay.py --out block_replay.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from rl_engine.validation.models import h3_block, h3_chain  # noqa: E402
from rl_engine.validation.models.h3_cases import H3_BLOCK_LAYOUTS  # noqa: E402
from rl_engine.validation.models.h3_weights import (  # noqa: E402
    load_h3_block_weights,
    load_h3_manifest,
)


def main() -> None:
    """Run the forward (and optionally backward) block replay per layout and save the report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layouts",
        default="tiny,small,medium",
        help=f"comma list of packed FL2VA layouts: {', '.join(H3_BLOCK_LAYOUTS)}",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-backward", action="store_true", help="forward replay only")
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    layouts = args.layouts.split(",")
    unknown = sorted(set(layouts) - set(H3_BLOCK_LAYOUTS))
    if unknown:
        parser.error(f"unknown layouts: {unknown}")
    if not torch.cuda.is_available():
        raise SystemExit("the block replay needs a CUDA device")

    weights = load_h3_block_weights("cuda")
    manifest = load_h3_manifest()
    ops = h3_block.CandidateOps()

    forward, backward = [], []
    for layout in layouts:
        case = h3_block.run_block_case(
            weights, layout=layout, seed=args.seed, ops=ops, timing_repeats=args.timing_repeats
        )
        forward.append(case)
        print(
            f"{layout} S={case['seq_len']}: first_drift={case['first_drift']} "
            f"first_isolated_drift={case['first_isolated_drift']} "
            f"repeat_drift={case['first_repeat_drift']} batch_drift={case['first_batch_drift']} "
            f"out err cand {case['output']['candidate_rel_err_vs_golden']:.2e} "
            f"prov {case['output']['provider_rel_err_vs_golden']:.2e}"
        )
        if not args.no_backward:
            case = h3_block.run_block_backward(
                weights, layout=layout, seed=args.seed, ops=ops, timing_repeats=args.timing_repeats
            )
            backward.append(case)
            det = all(v["repeat_bitwise_equal"] for v in case["leaves"].values())
            print(
                f"{layout} backward: leaves {'det' if det else 'NONDET'}, "
                f"grad_repeat_drift={case['first_grad_repeat_drift']} "
                f"grad_batch_drift={case['first_grad_batch_drift']} "
                f"dX batch-row {case['leaves']['hidden']['batch_row_bitwise_equal']}"
            )
        torch.cuda.empty_cache()

    evidence = {
        "kind": "h3_one_block_replay",
        "rfc": manifest["rfc"],
        "model_revision": manifest["model_identity"]["revision"],
        "weights_sha256": manifest["weight_shards"],
        "reference_commit": manifest["reference_implementation"]["commit"],
        **h3_chain.git_state(),
        "environment": h3_chain.environment(),
        "nodes": [
            {
                "node": n.name,
                "inputs": list(n.inputs),
                "rfc_row": n.binding.rfc_row,
                "status": n.binding.status,
                "backend": n.binding.backend,
            }
            for n in h3_block.NODES
        ],
        "forward_cases": forward,
        "backward_cases": backward,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(evidence, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
