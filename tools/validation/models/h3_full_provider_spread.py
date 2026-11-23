#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""How far independent BF16 provider passes spread, and where the candidate falls.

Inputs are ``h3_full_compare`` outputs against the same frozen FP32 reference: one
for the candidate and one per provider variant (``h3_full_model --mode bf16-provider
--provider-variant ...``). For every tensor and metric (``max_abs``, ``relative_l2``)
it reports the providers' minimum and maximum error, the candidate's error, and
whether the candidate lies above the providers' maximum. It also applies the frozen
bounds to each provider variant, which shows whether the bounds hold for providers
themselves. Diagnostic only: it changes no bound and accepts nothing.

    python tools/validation/models/h3_full_provider_spread.py \\
        --thresholds examples/models/minimax_h3/full-model-thresholds.json \\
        --candidate candidate-comparison.json \\
        --provider default=default.json --provider sdpa-flash=flash.json ... \\
        --output spread.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("max_abs", "relative_l2")


def main() -> None:
    """Load the comparisons, compute per-tensor spreads, and write a summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--provider", action="append", required=True, help="name=comparison.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bounds = json.loads(args.thresholds.read_text())["bounds"]
    candidate = json.loads(args.candidate.read_text())["errors"]
    providers = {}
    for item in args.provider:
        name, _, path = item.partition("=")
        providers[name] = json.loads(Path(path).read_text())["errors"]
    names = sorted(bounds)
    for label, errors in [("candidate", candidate), *providers.items()]:
        missing = set(names) - set(errors)
        if missing:
            raise ValueError(f"{label} lacks {len(missing)} tensors, e.g. {sorted(missing)[:3]}")

    tensors = {}
    summary = {
        "tensors": len(names),
        "providers": sorted(providers),
        "candidate_failures": {m: 0 for m in METRICS},
        "candidate_above_provider_max": {m: 0 for m in METRICS},
        "provider_failures": {p: {m: 0 for m in METRICS} for p in providers},
    }
    for name in names:
        entry = {}
        for m in METRICS:
            values = [providers[p][name][m] for p in providers]
            cand = candidate[name][m]
            bound = bounds[name][m]
            entry[m] = {
                "bound": bound,
                "candidate": cand,
                "provider_min": min(values),
                "provider_max": max(values),
                "candidate_over_provider_max": cand / max(values) if max(values) else None,
            }
            summary["candidate_failures"][m] += cand > bound
            summary["candidate_above_provider_max"][m] += cand > max(values)
            for p in providers:
                summary["provider_failures"][p][m] += providers[p][name][m] > bound
        tensors[name] = entry
    worst = sorted(
        names,
        key=lambda n: tensors[n]["max_abs"]["candidate_over_provider_max"] or 0.0,
        reverse=True,
    )[:20]
    summary["worst_candidate_over_provider_max"] = [
        {"tensor": n, **tensors[n]["max_abs"]} for n in worst
    ]
    args.output.write_text(json.dumps({"summary": summary, "tensors": tensors}, indent=2) + "\n")
    print(json.dumps(summary, indent=2)[:4000])


if __name__ == "__main__":
    main()
