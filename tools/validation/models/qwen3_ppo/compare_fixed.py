# SPDX-License-Identifier: Apache-2.0
"""Freeze independent HF error contracts or assess candidates without changing bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from tools.validation.models.qwen3_ppo.fixed_contract import (
    CONFIG,
    OPTIMIZER,
    fixed_tp_sizes,
    read_tensor,
    sha,
    verify_checkpoint,
)

# Each run rounds a weight at most twice per AdamW step (decoupled decay, then the step), half
# an FP32 ulp each, so two runs that take the same step differ by at most two ulps.
STEP_ROUNDING_ULPS = 2
UPDATE_RULE = (
    "an update tensor passes if its max_abs error is within its frozen bound, or, for FP32 runs, "
    "if every element's update difference equals the AdamW step difference predicted from both "
    f"runs' own gradients and grad norms within {STEP_ROUNDING_ULPS} FP32 ulps of the weight"
)


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


def fp32_ulp(x):
    """Return the FP32 spacing at ``|x|``, elementwise, as FP64."""
    a = x.abs().float()
    return (torch.nextafter(a, torch.full_like(a, float("inf"))) - a).double()


def checkpoint_weight(checkpoint, name):
    """Return the checkpoint tensor ``name`` as FP64, or ``None`` if the checkpoint lacks it."""
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    if name not in index:
        return None
    with safe_open(str(checkpoint / index[name]), "pt") as f:
        return f.get_tensor(name).double()


def adamw_directions(root, row, name, steps):
    """Return AdamW's FP64 directions ``m_hat / (sqrt(v_hat) + eps)`` for ``steps`` updates.

    They are built from the run's own unclipped gradients of ``name`` and its grad norms,
    clipped by ``max_grad_norm / (grad_norm + 1e-6)`` as both trainers clip.
    """
    (b1, b2), eps = OPTIMIZER["betas"], OPTIMIZER["eps"]
    m = v = torch.zeros((), dtype=torch.float64)
    out = []
    for t in range(steps):
        g = read_tensor(root, row["artifacts"][f"{t}/gradient/{name}"]).double()
        norm = float(read_tensor(root, row["artifacts"][f"{t}/grad_norm"]))
        g = g * min(1.0, CONFIG.max_grad_norm / (norm + 1e-6))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        out.append((m / (1 - b1 ** (t + 1))) / ((v / (1 - b2 ** (t + 1))).sqrt() + eps))
    return out


def explained_update(key, p0, left_root, left, right_root, right):
    """Return how far an update difference is from the AdamW step difference of its gradients.

    ``key`` is ``"<t>/update/<name>"`` and ``p0`` the parameter before the first update, which
    both runs start from. Both runs' weights before step ``t`` are ``p0`` plus their own
    recorded updates. The predicted difference is
    ``-lr * (dir_right - dir_left) - lr * weight_decay * (p_right - p_left)``; the tolerance is
    :data:`STEP_ROUNDING_ULPS` FP32 ulps of the largest of both runs' weights before and after
    the step. Returns ``passed``, the worst element and its residual in ulps.
    """
    t, name = int(key.split("/", 1)[0]), key.split("/update/", 1)[1]
    lr, wd = OPTIMIZER["lr"], OPTIMIZER["weight_decay"]
    d_left = adamw_directions(left_root, left, name, t + 1)[t]
    d_right = adamw_directions(right_root, right, name, t + 1)[t]
    p_left, p_right = p0.clone(), p0.clone()
    for s in range(t):
        p_left += read_tensor(left_root, left["artifacts"][f"{s}/update/{name}"]).double()
        p_right += read_tensor(right_root, right["artifacts"][f"{s}/update/{name}"]).double()
    u_left = read_tensor(left_root, left["artifacts"][key]).double()
    u_right = read_tensor(right_root, right["artifacts"][key]).double()
    predicted = -lr * (d_right - d_left) - lr * wd * (p_right - p_left)
    residual = (u_right - u_left - predicted).abs()
    largest = torch.stack(
        [p_left.abs(), (p_left + u_left).abs(), p_right.abs(), (p_right + u_right).abs()]
    ).amax(0)
    ulps = residual / fp32_ulp(largest)
    worst = int(ulps.flatten().argmax())
    return dict(
        passed=bool((ulps <= STEP_ROUNDING_ULPS).all()),
        max_residual_ulps=float(ulps.flatten()[worst]),
        max_residual=float(residual.max()),
        worst_element=worst,
        observed=float((u_right - u_left).flatten()[worst]),
        predicted=float(predicted.flatten()[worst]),
    )


def assess(rows, bounds, explain):
    """Return ``(failures, explained)`` for candidate errors ``rows`` against frozen ``bounds``.

    A tensor over its bound fails unless it is an update and ``explain(name)`` returns a passing
    :func:`explained_update` result; ``explain`` returns ``None`` where the rule does not apply.
    """
    failures, explained = [], {}
    for name, value in rows.items():
        if value["max_abs"] <= bounds[name]["max_abs"]:
            continue
        check = explain(name) if "/update/" in name else None
        if check is not None:
            explained[name] = dict(check, max_abs=value["max_abs"], bound=bounds[name]["max_abs"])
        if check is None or not check["passed"]:
            failures.append(name)
    return failures, explained


def load_contract(path):
    """Return ``(bounds contract, SHA-256)`` of the thresholds file, both from one read.

    Raises ``ValueError`` unless the file is a frozen bounds contract: a JSON object with a
    non-empty ``bounds`` mapping and the ``reference_report_sha256`` it was frozen against.
    """
    raw = Path(path).read_bytes()
    contract = json.loads(raw)
    if (
        not isinstance(contract, dict)
        or not isinstance(contract.get("bounds"), dict)
        or not contract["bounds"]
        or "reference_report_sha256" not in contract
    ):
        raise ValueError("thresholds file is not a frozen bounds contract")
    return contract, hashlib.sha256(raw).hexdigest()


def expected_thresholds(other, *, tp_comparison):
    """Return the acceptance-manifest name of the bounds file a comparison must use.

    Against the HF FP32 reference: an FP32 run uses ``<role>-fp32-repeat-bounds.json`` and a
    BF16 run ``<role>-cross-precision-bounds.json``. A same-precision TP comparison uses
    ``<role>-<precision>-repeat-bounds.json``.
    """
    role, precision = other["role"], other["precision"]
    if tp_comparison or precision == "fp32":
        return f"{role}-{precision}-repeat-bounds.json"
    return f"{role}-cross-precision-bounds.json"


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
    process exits with status 1 on any failure. With ``--checkpoint`` and two FP32 runs, an
    update tensor over its bound still passes if :func:`explained_update` finds its difference
    equal to the AdamW step difference of the two runs' gradients (:data:`UPDATE_RULE`); such
    tensors are listed under ``explained_updates``. Raises ``ValueError`` if ``--output``
    already exists.
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
    p.add_argument("--checkpoint", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("never overwrite reference or failed comparison")
    if a.thresholds is not None and a.extra_control:
        raise ValueError("controls establish tolerances; candidates are compared one at a time")
    left, right = report(a.reference), report(a.other)
    extra = [report(c) for c in a.extra_control]
    for control in [right, *extra]:
        same_inputs(left, control)
    if a.thresholds is None and any(
        r["status"] != "fixed_reference_complete" for r in [left, right, *extra]
    ):
        raise ValueError("only independent HF controls may establish tolerances")
    contract, thresholds_sha256 = (
        load_contract(a.thresholds) if a.thresholds is not None else (None, None)
    )
    if (
        contract is not None
        and not a.tp_comparison
        and contract["reference_report_sha256"] != sha(a.reference / "report.json")
    ):
        raise ValueError("thresholds refer to a different reference")
    if contract is not None:
        if not a.acceptance_manifest or not a.reference_root:
            raise ValueError("candidate comparison requires committed acceptance manifest")
        from tools.validation.models.qwen3_ppo.fixed_contract import verify_reference

        frozen = verify_reference(a.reference_root, a.acceptance_manifest, metadata_only=True)
        if right.get("acceptance_manifest_sha256") != sha(a.acceptance_manifest):
            raise ValueError("candidate did not record this acceptance manifest")
        expected = expected_thresholds(right, tp_comparison=a.tp_comparison)
        if frozen["files"].get(expected) != thresholds_sha256:
            raise ValueError(f"thresholds are not the pinned {expected}")
        if a.tp_comparison:
            tp = frozen["matrix"]["tp"]
            if (
                left.get("tp") not in tp
                or right.get("tp") not in tp
                or not left["tp"] < right["tp"]
                or left["precision"] != right["precision"]
                or left["framework"] != right["framework"]
            ):
                raise ValueError("invalid same-precision TP comparison")
            sizes = fixed_tp_sizes(right["framework"], right["precision"], right["role"])
            if sizes != (left["tp"], right["tp"]):
                raise ValueError("this row has no TP2-vs-TP4 comparison (fixed_tp_sizes)")
        for run in (left, right):
            if run["status"] != "fixed_candidate_complete":
                continue
            sizes = fixed_tp_sizes(run["framework"], run["precision"], run["role"])
            if run.get("tp") not in sizes:
                raise ValueError(
                    f"TP{run.get('tp')} is not a size of the {run['framework']} "
                    f"{run['precision']} {run['role']} row {sizes}"
                )
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
        fp32 = left.get("precision") == right.get("precision") == "fp32"
        checkpoint_verified = []

        def explain(name):
            if not (fp32 and a.checkpoint):
                return None
            if not checkpoint_verified:
                # The checkpoint's weights set the tolerance, so it must be the pinned one.
                verify_checkpoint(a.checkpoint)
                checkpoint_verified.append(True)
            p0 = checkpoint_weight(a.checkpoint, name.split("/update/", 1)[1])
            if p0 is None:
                return None
            return explained_update(name, p0, a.reference, left, a.other, right)

        failures, explained = assess(rows, contract["bounds"], explain)
        result.update(
            status="failed" if failures else "passed",
            update_rule=UPDATE_RULE,
            errors=rows,
            failures=failures,
            explained_updates=explained,
            thresholds_sha256=thresholds_sha256,
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
    if contract is not None and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
