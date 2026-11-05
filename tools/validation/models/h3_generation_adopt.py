#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Move saved candidate-generation runs onto the model-tree identity (RFC #420, #539).

Runs started before the identity moved from a commit to the git trees of the model paths
(``h3_generation_resume.MODEL_PATHS``) carry the commit they started at. This re-labels
them for the current checkout, once, and only if every model path is byte-identical
between that commit and ``HEAD``; anything else is refused and nothing is written.

    python tools/validation/models/h3_generation_adopt.py --output <generation dir> \\
        --from-commit 4792c8a9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl_engine.validation.models.h3_generation_resume import adopt_identity  # noqa: E402
from tools.validation.models.h3_full_generation import run_identity  # noqa: E402


def main() -> None:
    """Adopt every seed directory of ``--output`` (all contract seeds by default)."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-commit", required=True)
    parser.add_argument("--seeds", help="comma list; default: the contract's seeds")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3]
    config_root = source / "examples/models/minimax_h3"
    contract = json.loads((config_root / "generation-contract.json").read_text())
    inputs = json.loads((config_root / "generation-inputs.json").read_text())
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else contract["seeds"]
    # Check every seed before changing any, so a refusal leaves all runs untouched.
    plans = []
    for seed in seeds:
        identity = run_identity(
            config_root,
            contract,
            inputs,
            candidate=True,
            preflight=False,
            seed=seed,
            settings=contract,
        )
        plans.append((args.output / str(seed), identity))
    for directory, identity in plans:
        adopt_identity(directory, identity, args.from_commit, cwd=source, dry_run=True)
    for directory, identity in plans:
        adopt_identity(directory, identity, args.from_commit, cwd=source)
        print(f"adopted {directory.name} (from {args.from_commit})")


if __name__ == "__main__":
    main()
