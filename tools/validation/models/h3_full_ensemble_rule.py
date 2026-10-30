#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Full-model acceptance from an ensemble of independent BF16 provider configurations.

The single-configuration rule (``1.5 * one provider's error + floor``) accepts only the
configuration it was frozen from: providers that differ only in their SDPA backend or
BLAS library fail it on tens of tensors. This rule is calibrated on several provider
configurations, compared with the same frozen FP32 reference, and contains no candidate
data:

* E[t] is the largest ``relative_l2`` error any configuration has on tensor ``t``;
* the bound is ``k * E[t] + floor`` (floors as in the single-configuration rule);
* ``k`` is the smallest factor, rounded up to 0.1, for which every configuration held
  out passes against the maximum of the others (leave-one-out);
* ``max_abs``, set by one element, is reported but not gated: held-out configurations
  exceed the others' maximum by several times on it.

    h3_full_ensemble_rule.py calibrate --provider NAME=comparison.json ... \\
        --output rule.json
    h3_full_ensemble_rule.py check --rule rule.json --candidate comparison.json \\
        --output out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

# ``h3_full_compare`` writes the first status without ``--thresholds`` (an independent provider
# against the FP32 reference) and one of the others with the frozen thresholds (a candidate).
PROVIDER_STATUS = "frozen_reference_bounds"
CANDIDATE_STATUSES = ("passed", "failed")


def _floor(name: str) -> float:
    gradient = (
        ".grad" in name or name.startswith("parameter.") or name.startswith("input_gradients.")
    )
    return 5e-3 if gradient else 1e-3


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate(args) -> None:
    """Build the ensemble bounds and the leave-one-out factor from provider comparisons."""

    providers, sources = {}, {}
    for item in args.provider:
        name, _, path = item.partition("=")
        report = json.loads(Path(path).read_text())
        if report.get("diagnostic_provider_nodes"):
            raise ValueError(f"{name} is a hybrid diagnostic, not a provider configuration")
        if report.get("status") != PROVIDER_STATUS:
            raise ValueError(
                f"{name} is not a provider comparison (status {report.get('status')!r}); a "
                "candidate comparison must not enter the rule it is judged by"
            )
        providers[name] = report["errors"]
        sources[name] = {
            "sha256": _digest(Path(path)),
            "reference": report["reference_report_sha256"],
        }
    if len({s["reference"] for s in sources.values()}) != 1:
        raise ValueError("provider comparisons use different FP32 references")
    # Configurations whose errors are identical to an earlier one are one configuration.
    distinct = {}
    for name in sorted(providers):
        if not any(providers[name] == providers[kept] for kept in distinct):
            distinct[name] = providers[name]
    if len(distinct) < 3:
        raise ValueError("need at least three distinct provider configurations")
    names = sorted(next(iter(distinct.values())))
    loo = {}
    for held in distinct:
        others = [distinct[o] for o in distinct if o != held]
        worst = 0.0
        for t in names:
            top = max(o[t]["relative_l2"] for o in others)
            if top > 0:
                worst = max(worst, (distinct[held][t]["relative_l2"] - _floor(t)) / top)
        loo[held] = worst
    factor = math.ceil(max(loo.values()) * 10) / 10
    ensemble = {t: max(d[t]["relative_l2"] for d in distinct.values()) for t in names}
    rule = {
        "status": "frozen_ensemble_rule",
        "rule": "relative_l2 <= k * max over provider configurations + floor; max_abs reported",
        "factor_k": factor,
        "leave_one_out_worst_ratio": loo,
        "providers": sorted(distinct),
        "duplicates_dropped": sorted(set(providers) - set(distinct)),
        "sources": sources,
        "reference_report_sha256": next(iter(sources.values()))["reference"],
        "ensemble_relative_l2": ensemble,
        "ensemble_max_abs": {t: max(d[t]["max_abs"] for d in distinct.values()) for t in names},
        "bounds_relative_l2": {t: factor * ensemble[t] + _floor(t) for t in names},
    }
    args.output.write_text(json.dumps(rule, indent=2) + "\n")
    print(
        f"k = {factor} from leave-one-out {loo}; "
        f"{len(distinct)} configurations, {len(names)} tensors"
    )


def check(args) -> None:
    """Apply a frozen ensemble rule to one candidate comparison; exit 1 on any failure."""

    rule = json.loads(args.rule.read_text())
    report = json.loads(args.candidate.read_text())
    if report["reference_report_sha256"] != rule["reference_report_sha256"]:
        raise ValueError("candidate was compared with a different FP32 reference")
    if report.get("diagnostic_provider_nodes"):
        raise ValueError("a hybrid diagnostic is not a candidate result")
    if report.get("status") not in CANDIDATE_STATUSES:
        raise ValueError(
            f"not a candidate comparison (status {report.get('status')!r}): a provider comparison "
            "has no thresholds to be judged by"
        )
    errors = report["errors"]
    if errors.keys() != rule["bounds_relative_l2"].keys():
        raise ValueError("tensor inventory differs from the rule")
    failures = [
        t for t in sorted(errors) if errors[t]["relative_l2"] > rule["bounds_relative_l2"][t]
    ]
    ratios = {t: errors[t]["relative_l2"] / rule["bounds_relative_l2"][t] for t in errors}
    max_abs_ratio = {
        t: errors[t]["max_abs"] / rule["ensemble_max_abs"][t]
        for t in errors
        if rule["ensemble_max_abs"][t]
    }
    out = {
        "status": "failed" if failures else "passed",
        "rule_sha256": _digest(args.rule),
        "candidate_comparison_sha256": _digest(args.candidate),
        "failures": failures,
        "worst_relative_l2_over_bound": sorted(ratios.items(), key=lambda x: -x[1])[:10],
        "outputs": {t: errors[t] for t in ("video", "audio") if t in errors},
        "max_abs_over_ensemble_max_reported": sorted(max_abs_ratio.items(), key=lambda x: -x[1])[
            :10
        ],
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"{out['status']}: {len(failures)} failures; worst {out['worst_relative_l2_over_bound'][0]}"
    )
    if failures:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--provider", action="append", required=True, help="name=comparison.json")
    cal.add_argument("--output", type=Path, required=True)
    chk = sub.add_parser("check")
    chk.add_argument("--rule", type=Path, required=True)
    chk.add_argument("--candidate", type=Path, required=True)
    chk.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibrate(args) if args.cmd == "calibrate" else check(args)


if __name__ == "__main__":
    main()
