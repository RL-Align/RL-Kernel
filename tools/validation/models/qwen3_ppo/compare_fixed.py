# SPDX-License-Identifier: Apache-2.0
"""Freeze independent HF error contracts or assess candidates without changing bounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.validation.models.qwen3_ppo.fixed_contract import read_tensor, sha


def errors(x, y):
    """Return the FP64 max absolute error and relative L2 error (relative to ``x``) of ``x - y``.

    Raises ``ValueError`` on a shape mismatch or nonfinite values.
    """
    if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("nonfinite or mismatched tensor comparison")
    delta = x.double() - y.double()
    return dict(
        max_abs=float(delta.abs().max()),
        relative_l2=float(delta.norm() / x.double().norm().clamp_min(1e-12)),
    )


def floor(name):
    """Return the absolute bound floor: 1e-7 for updates, 1e-5 for scores, 1e-6 otherwise."""
    if "/update/" in name:
        return 1e-7
    if name.endswith("/scores"):
        return 1e-5
    return 1e-6


def report(root):
    """Return ``root/report.json``; raise ``ValueError`` unless it is a completed fixed run."""
    row = json.loads((root / "report.json").read_text())
    if row["status"] not in ("fixed_reference_complete", "fixed_candidate_complete"):
        raise ValueError("incomplete fixed-rollout run")
    return row


def same_inputs(a, b):
    """Raise ``ValueError`` unless two run reports were made from the same fixed inputs.

    They must match in role, rollout and targets manifests, config, optimizer, model revision
    and tensor inventory.
    """
    for key in (
        "role",
        "rollout_manifest_sha256",
        "targets_manifest_sha256",
        "config",
        "optimizer",
        "model_revision",
    ):
        if a[key] != b[key]:
            raise ValueError(f"fixed comparison differs: {key}")
    if a["artifacts"].keys() != b["artifacts"].keys():
        raise ValueError("tensor inventories differ")


def main():
    """Compare every evidence tensor of two fixed-rollout runs and write the result JSON.

    Without ``--thresholds``, HF reference runs freeze bounds of 4x the max error plus a
    floor. Each ``--extra-control`` is compared with ``--reference`` like ``--other``, and the
    bound takes the larger error per tensor. With ``--thresholds``, the other run passes or
    fails on ``max_abs`` against threshold bounds pinned in the acceptance manifest, and the
    process exits with status 1 on any failure. Raises ``ValueError`` if ``--output`` already
    exists.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--other", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--thresholds", type=Path)
    p.add_argument("--acceptance-manifest", type=Path)
    p.add_argument("--reference-root", type=Path)
    p.add_argument("--tp-comparison", action="store_true")
    p.add_argument("--extra-control", type=Path, action="append", default=[])
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("never overwrite reference or failed comparison")
    if a.thresholds and a.extra_control:
        raise ValueError("controls establish tolerances; candidates are compared one at a time")
    left, right = report(a.reference), report(a.other)
    extra = [report(c) for c in a.extra_control]
    for control in [right, *extra]:
        same_inputs(left, control)
    if not a.thresholds and any(
        r["status"] != "fixed_reference_complete" for r in [left, right, *extra]
    ):
        raise ValueError("only independent HF controls may establish tolerances")
    contract = json.loads(a.thresholds.read_text()) if a.thresholds else None
    if (
        contract
        and not a.tp_comparison
        and contract["reference_report_sha256"] != sha(a.reference / "report.json")
    ):
        raise ValueError("thresholds refer to a different reference")
    if contract:
        if not a.acceptance_manifest or not a.reference_root:
            raise ValueError("candidate comparison requires committed acceptance manifest")
        from tools.validation.models.qwen3_ppo.fixed_contract import verify_reference

        frozen = verify_reference(a.reference_root, a.acceptance_manifest, metadata_only=True)
        if right.get("acceptance_manifest_sha256") != sha(a.acceptance_manifest):
            raise ValueError("candidate did not record this acceptance manifest")
        if sha(a.thresholds) not in frozen["files"].values():
            raise ValueError("thresholds not pinned in acceptance manifest")
        if a.tp_comparison:
            if (
                left.get("tp") != 1
                or right.get("tp") != 2
                or left["precision"] != right["precision"]
                or left["framework"] != right["framework"]
            ):
                raise ValueError("invalid same-precision TP comparison")
    rows = {}
    control_rows: dict[str, dict] = {str(c): {} for c in a.extra_control}
    for name, item in left["artifacts"].items():
        reference = read_tensor(a.reference, item)
        rows[name] = errors(reference, read_tensor(a.other, right["artifacts"][name]))
        for path, control in zip(a.extra_control, extra):
            control_rows[str(path)][name] = errors(
                reference, read_tensor(path, control["artifacts"][name])
            )
    result = dict(
        reference_report_sha256=sha(a.reference / "report.json"),
        other_report_sha256=sha(a.other / "report.json"),
    )
    if contract:
        if rows.keys() != contract["bounds"].keys():
            raise ValueError("threshold tensor inventory differs")
        failures = [
            name
            for name, value in rows.items()
            if value["max_abs"] > contract["bounds"][name]["max_abs"]
        ]
        result.update(
            status="failed" if failures else "passed",
            errors=rows,
            failures=failures,
            thresholds_sha256=sha(a.thresholds),
        )
    else:
        controls = [rows, *control_rows.values()]
        result.update(
            status="frozen_reference_bounds",
            rule="4 * largest independent control max_abs error + metric floor",
            bounds={
                n: {
                    "max_abs": 4 * max(c[n]["max_abs"] for c in controls) + floor(n),
                    "relative_l2_control": max(c[n]["relative_l2"] for c in controls),
                }
                for n in rows
            },
        )
        if a.extra_control:
            result["extra_controls"] = [
                dict(
                    report_sha256=sha(path / "report.json"),
                    implementation_control=control.get("implementation_control"),
                    errors=control_rows[str(path)],
                )
                for path, control in zip(a.extra_control, extra)
            ]
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    if contract and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
