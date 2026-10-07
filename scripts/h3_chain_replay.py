#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Replay the MiniMax-H3 conditioning chain and write JSON evidence (RFC #420).

See ``rl_engine/testing/h3_chain.py`` for what is compared.

    export RL_KERNEL_H3_WEIGHTS=<dir written by scripts/prepare_h3_weights.py>
    python scripts/h3_chain_replay.py --out chain_replay.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rl_engine.kernels.registry import KernelRegistry  # noqa: E402
from rl_engine.testing import h3_chain  # noqa: E402
from rl_engine.testing.h3_weights import (  # noqa: E402
    load_h3_conditioning_weights,
    load_h3_manifest,
)


def _ints(text: str) -> list[int]:
    return [int(v) for v in text.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default=",".join(s.name for s in h3_chain.STAGES))
    parser.add_argument("--timesteps", type=_ints, default=[1, 2, 4], help="comma list of T")
    parser.add_argument("--seq-lens", type=_ints, default=[3, 257, 4097], help="comma list of S")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--backward",
        action="store_true",
        help="also replay the chain backward (needs every stage): parameter-gradient "
        "determinism and accuracy for the separate-op, fused and diffusers chains",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("the chain replay needs a CUDA device")
    wanted = args.stages.split(",")
    unknown = sorted(set(wanted) - {s.name for s in h3_chain.STAGES})
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}")
    stages = [s for s in h3_chain.STAGES if s.name in wanted]

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    weights = load_h3_conditioning_weights("cuda")
    registry = KernelRegistry()
    manifest = load_h3_manifest()

    cases = []
    for num_timesteps in args.timesteps:
        for seq_len in args.seq_lens:
            case = h3_chain.run_case(
                registry,
                weights,
                num_timesteps=num_timesteps,
                seq_len=seq_len,
                seed=args.seed,
                stages=stages,
            )
            cases.append(case)
            summary = ", ".join(
                f"{e['stage']}="
                f"{'eq' if e['chained_vs_provider']['bitwise_equal'] else 'drift'}"
                f"(gold {e['chained_vs_golden']['max_abs']:.3e})"
                for e in case["stages"]
            )
            print(f"T={num_timesteps} S={seq_len}: {summary}; first_drift={case['first_drift']}")

    backward_cases = []
    if args.backward:
        for num_timesteps in args.timesteps:
            for seq_len in args.seq_lens:
                case = h3_chain.run_backward_case(
                    registry, weights, num_timesteps=num_timesteps, seq_len=seq_len, seed=args.seed
                )
                backward_cases.append(case)
                parts = []
                for mode in h3_chain.BACKWARD_MODES:
                    entries = case["leaves"].values()
                    det = all(e[mode]["repeat_bitwise_equal"] for e in entries)
                    worst = max(e[mode]["max_abs_vs_golden_over_absmax"] for e in entries)
                    parts.append(f"{mode}: {'det' if det else 'NONDET'} {worst:.1e}")
                summary = " | ".join(parts)
                print(f"backward T={num_timesteps} S={seq_len}: {summary}")

    evidence = {
        "kind": "h3_conditioning_chain_replay",
        "rfc": manifest["rfc"],
        "model_revision": manifest["model_identity"]["revision"],
        "weights_sha256": manifest["weight_shards"],
        "reference_commit": manifest["reference_implementation"]["commit"],
        **h3_chain.git_state(),
        "environment": h3_chain.environment(),
        "stages": [s.name for s in stages],
        "cases": cases,
        "backward_cases": backward_cases,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(evidence, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
