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
    report = json.loads((root / "report.json").read_text())
    if digest(root / "tensors.pt") != report["tensors_sha256"]:
        raise ValueError("full-model tensor evidence changed")
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


def compare(left, right, norms=None):
    """Per tensor ``max_abs`` / ``relative_l2``; ``norms``, if given, gets each tensor L2 norms."""

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
        if norms is not None:
            norms[name] = dict(
                reference=value.double().norm().item(), other=other.double().norm().item()
            )
        yield name, dict(
            max_abs=delta.abs().max().item(),
            relative_l2=(delta.double().norm() / value.double().norm().clamp_min(1e-12)).item(),
        )


def drift_order(report, names):
    trace = report["trace"]
    if "execution_order" in trace:
        observed = trace["execution_order"]
        return [name for name in observed if name in names] + [
            name for name in names if name not in observed
        ]
    forward = trace.get("forward_order", list(trace["calls"]))
    backward = trace.get("backward_order", list(trace["backward_calls"]))
    ordered = ["nodes." + name for name in forward]
    ordered += ["nodes." + name + ".grad" for name in backward]
    ordered += [name for name in names if name not in ordered]
    return [name for name in ordered if name in names]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--other", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--thresholds", type=Path)
    p.add_argument("--reference-manifest", type=Path)
    p.add_argument("--reference-root", type=Path)
    args = p.parse_args()
    if bool(args.reference_manifest) != bool(args.reference_root):
        raise ValueError("--reference-manifest and --reference-root must be given together")
    if (
        args.reference_root
        and args.reference.resolve() != (args.reference_root / "fp32-reference").resolve()
    ):
        # The manifest verifies the files under the root; the tensors are read from --reference.
        # Anything else could be compared against a run the manifest never verified.
        raise ValueError("--reference must be <reference-root>/fp32-reference, the verified run")
    if args.output.exists():
        raise ValueError("never overwrite calibration or failed evidence")
    from tools.validation.models.h3_evidence_integrity import verify_inventory

    if args.thresholds and not (args.reference_root and args.reference_manifest):
        raise ValueError("candidate comparison requires pinned reference inventory")
    if args.reference_manifest:
        verify_inventory(args.reference_root, args.reference_manifest)
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
    norms: dict[str, dict[str, float]] = {}
    rows = dict(compare(args.reference, args.other, norms if args.thresholds else None))
    report = dict(
        reference_report_sha256=reference_sha,
        other_report_sha256=digest(args.other / "report.json"),
    )
    if args.thresholds:
        if rows.keys() != contract["bounds"].keys():
            raise ValueError("threshold tensor inventory differs")
        failures = [
            name
            for name in drift_order(other if "execution_order" in other["trace"] else ref, rows)
            if any(rows[name][k] > contract["bounds"][name][k] for k in rows[name])
        ]
        hybrid = other.get("diagnostic_provider_nodes") or []
        report.update(
            status="diagnostic_only" if hybrid else ("failed" if failures else "passed"),
            diagnostic_provider_nodes=hybrid,
            errors=rows,
            tensor_l2=norms,
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
            errors=rows,
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.thresholds and failures and not report.get("diagnostic_provider_nodes"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
