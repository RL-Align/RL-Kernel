# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Freeze full-H3 reference error bounds, or evaluate a candidate without changing them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.validation.models.h3_full_model import digest


def tensors(root):
    payload = torch.load(root / "tensors.pt", map_location="cpu", weights_only=True)
    yield "video", payload["video"]
    yield "audio", payload["audio"]
    for section in ("nodes", "input_gradients"):
        for name, value in sorted(payload[section].items()):
            yield section + "." + name, value
    for name, row in sorted(json.loads((root / "gradients.json").read_text()).items()):
        path = root / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"gradient artifact changed: {path}")
        yield "parameter." + name, torch.load(path, map_location="cpu", weights_only=True)


def compare(left, right):
    a, b = tensors(left), tensors(right)
    from itertools import zip_longest

    for x, y in zip_longest(a, b):
        if x is None or y is None or x[0] != y[0]:
            raise ValueError("different full-model tensor inventories")
        name, value = x
        other = y[1]
        if (
            value.shape != other.shape
            or not torch.isfinite(value).all()
            or not torch.isfinite(other).all()
        ):
            raise ValueError(f"nonfinite/shape mismatch: {name}")
        delta = value.float() - other.float()
        yield name, dict(
            max_abs=delta.abs().max().item(),
            relative_l2=(delta.double().norm() / value.double().norm().clamp_min(1e-12)).item(),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--other", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--thresholds", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("never overwrite calibration or failed evidence")
    ref = json.loads((args.reference / "report.json").read_text())
    other = json.loads((args.other / "report.json").read_text())
    if ref["mode"] != "fp32-reference" or ref["status"] != "execution_complete_acceptance_pending":
        raise ValueError("independent completed FP32 reference required")
    if other["status"] != "execution_complete_acceptance_pending":
        raise ValueError("other execution is incomplete")
    reference_sha = digest(args.reference / "report.json")
    if args.thresholds:
        contract = json.loads(args.thresholds.read_text())
        if other["mode"] != "candidate" or contract["reference_report_sha256"] != reference_sha:
            raise ValueError("wrong candidate/reference contract")
        if other.get("thresholds_sha256") != digest(args.thresholds):
            raise ValueError("thresholds were not frozen before candidate execution")
    elif other["mode"] != "bf16-provider":
        raise ValueError("only independent provider can establish thresholds")
    rows = dict(compare(args.reference, args.other))
    report = dict(
        reference_report_sha256=reference_sha,
        other_report_sha256=digest(args.other / "report.json"),
    )
    if args.thresholds:
        if rows.keys() != contract["bounds"].keys():
            raise ValueError("threshold tensor inventory differs")
        failures = [
            name
            for name, row in rows.items()
            if any(row[k] > contract["bounds"][name][k] for k in row)
        ]
        report.update(
            status="failed" if failures else "passed",
            errors=rows,
            failures=failures,
            first_drift=failures[0] if failures else None,
            thresholds_sha256=digest(args.thresholds),
        )
    else:
        # Committed before candidate execution. Forward and gradient tensors use
        # separate floors; the existing block-0 golden contract is unchanged.
        bounds = {
            name: {
                key: 1.5 * value
                + (
                    5e-3
                    if ".grad" in name
                    or name.startswith("parameter.")
                    or name.startswith("input_gradients.")
                    else 1e-3
                )
                for key, value in row.items()
            }
            for name, row in rows.items()
        }
        report.update(
            status="frozen_reference_bounds",
            rule="1.5 * independent BF16 provider error + fixed floor",
            bounds=bounds,
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.thresholds and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
