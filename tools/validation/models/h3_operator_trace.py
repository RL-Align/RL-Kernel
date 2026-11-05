#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Record, verify and compare MiniMax-H3 operator traces.

    # one-block replay of block 0 with the trace attached (needs CUDA + RL_KERNEL_H3_WEIGHTS)
    python tools/validation/models/h3_operator_trace.py block --layout tiny --out trace.json

    # fail-closed gate: exit 1 and list every violation
    python tools/validation/models/h3_operator_trace.py verify trace.json [--allow row interim]
    python tools/validation/models/h3_operator_trace.py verify trace.json --block 0

    # first call at which two traces disagree
    python tools/validation/models/h3_operator_trace.py diff a.json b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from rl_engine.models.minimax_h3 import fingerprint as fp  # noqa: E402
from rl_engine.models.minimax_h3 import operator_trace as ot  # noqa: E402


def _record_block(layout: str, seed: int) -> ot.OperatorTrace:
    import torch

    from rl_engine.validation.models import h3_block, h3_weights

    weights = h3_weights.load_h3_block_weights("cuda")
    ops = h3_block.CandidateOps()
    ctx = h3_block.make_block_context(weights, layout=layout, seed=seed, batch=1)
    trace = ot.OperatorTrace(
        mode="candidate", metadata={"layout": layout, "seed": seed, "block": 0}
    )
    ctx["node_observer"] = ot.block_observer(trace, block_index=0)
    with torch.no_grad():
        h3_block.run_graph("candidate", ctx, ops)
    return trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    block = sub.add_parser("block")
    block.add_argument("--layout", default="tiny")
    block.add_argument("--seed", type=int, default=0)
    block.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("trace", type=Path)
    verify.add_argument("--allow", nargs="*", default=["row"], choices=ot.STATUSES)
    verify.add_argument(
        "--block", type=int, default=None, help="expect only this transformer block"
    )
    diff = sub.add_parser("diff")
    diff.add_argument("lhs", type=Path)
    diff.add_argument("rhs", type=Path)
    args = parser.parse_args()

    if args.command == "block":
        trace = _record_block(args.layout, args.seed)
        args.out.write_text(trace.dumps())
        print(f"wrote {args.out} ({trace.digest})")
        print(json.dumps(trace.status_summary(), indent=1))
        return 0
    if args.command == "verify":
        trace = ot.OperatorTrace.loads(args.trace.read_text())
        paths = None
        if args.block is not None:
            prefix = f"transformer_blocks.{args.block}."
            paths = [node.path for node in fp.build_graph() if node.path.startswith(prefix)]
        problems = trace.violations(paths=paths, allowed_statuses=args.allow)
        if problems:
            print(f"{len(problems)} violation(s):", file=sys.stderr)
            for problem in problems[:60]:
                print("  " + problem, file=sys.stderr)
            return 1
        print(f"ok: {len(trace.calls)} calls, digest {trace.digest}")
        return 0
    lhs = ot.OperatorTrace.loads(args.lhs.read_text())
    rhs = ot.OperatorTrace.loads(args.rhs.read_text())
    difference = ot.first_difference(lhs, rhs)
    if difference is None:
        print("identical")
        return 0
    print(json.dumps(difference, indent=1))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
