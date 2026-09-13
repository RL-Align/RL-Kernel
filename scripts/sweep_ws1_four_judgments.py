#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Sweep the WS1 C8 four-judgment matrix.

By default this only classifies cells (CPU-safe). Pass ``--execute`` on a GPU
host to run representative case accuracy plus C3/C4 logical invariance.
SM90-only or resource-blocked cells stay ``pending_hopper``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from collections import defaultdict
from typing import Any, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.gtest.four_judgment_matrix import (  # noqa: E402
    C8_REQUIRED_OPS,
    JUDGMENTS,
    PROFILES,
    MatrixCell,
    MatrixReport,
    build_classified_matrix,
)
from rl_engine.kernels.gtest.gradient_adapters import (  # noqa: E402
    get_adapter,
    resolve_profile_candidate,
)
from rl_engine.testing.ws1_workload import load_manifest  # noqa: E402

C3 = REPO_ROOT / "scripts" / "check_forward_invariance.py"
C4 = REPO_ROOT / "scripts" / "check_gradient_invariance.py"
C2_CASE = REPO_ROOT / "scripts" / "ws1_candidate_evidence.py"


def _classify_process(
    returncode: int, output: str, *, kind: str, hopper: bool = False
) -> tuple[str, str]:
    if returncode == 0:
        return "green", f"{kind} gate passed"
    if "has no backward" in output:
        return "red", "candidate is not wired through torch.autograd"
    hopper_needed = (
        "is not compiled" in output
        or "needs a Hopper" in output
        or "requires Hopper" in output
        or "fallback forbidden" in output
    )
    if hopper_needed and not hopper:
        return "pending_hopper", "declared candidate needs a Hopper build"
    if "missing_required" in output:
        return "red", "C2 marks this node missing_required"
    if "layout_supported" in output:
        return "N/A", "profile-independent; covered by the CPU contract test"
    return "red", output.strip().splitlines()[-1][:200] if output.strip() else f"{kind} gate failed"


def _parse_json_blob(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _observed_from_gate(payload: dict[str, Any] | None) -> dict[str, str] | None:
    if not payload:
        return None
    provenance = payload.get("backend_provenance") or {}
    backend = provenance.get("actual_backend") or payload.get("observed_actual_backend")
    kernel = payload.get("observed_kernel_id")
    if not backend or not kernel:
        return None
    return {"backend": str(backend), "kernel": str(kernel)}


def _run_gate(
    script: pathlib.Path, op_name: str, candidate: str, profile: str
) -> tuple[int, str, dict[str, Any] | None]:
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--op",
            op_name,
            "--candidate",
            candidate,
            "--backend-profile",
            profile,
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    combined = proc.stdout + proc.stderr
    return proc.returncode, combined, _parse_json_blob(proc.stdout)


def _run_case_gate(case_id: str, profile: str, *, gradient: bool) -> tuple[int, str]:
    command = [
        sys.executable,
        str(C2_CASE),
        "--profile",
        profile,
        "--case-id",
        case_id,
        "--emit-json",
        "-",
    ]
    if gradient:
        command.append("--check-grad")
    proc = subprocess.run(command, capture_output=True, text=True, cwd=str(REPO_ROOT))
    return proc.returncode, proc.stdout + proc.stderr


def _is_hopper() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 9)


def _execute_matrix(base: MatrixReport, profiles: Sequence[str] = PROFILES) -> MatrixReport:
    manifest = load_manifest()
    if _is_hopper():
        base = build_classified_matrix(manifest, allow_sm90=True, profiles=profiles)
    invariance: dict[tuple[str, str], dict[str, tuple[str, str, dict[str, str] | None]]] = {}
    for profile in profiles:
        for op_name in C8_REQUIRED_OPS:
            sample = next(
                cell for cell in base.cells if cell.profile == profile and cell.op_name == op_name
            )
            if sample.status in {"pending_hopper", "N/A"}:
                continue
            resolved = resolve_profile_candidate(get_adapter(op_name), profile, manifest)
            candidate = resolved["expected_backend_id"]
            if not candidate:
                continue
            c3_code, c3_out, c3_payload = _run_gate(C3, op_name, str(candidate), profile)
            c4_code, c4_out, c4_payload = _run_gate(C4, op_name, str(candidate), profile)
            hopper = _is_hopper()
            fwd_status, fwd_detail = _classify_process(
                c3_code, c3_out, kind="forward", hopper=hopper
            )
            grad_status, grad_detail = _classify_process(
                c4_code, c4_out, kind="gradient", hopper=hopper
            )

            def _actual(payload: dict[str, Any] | None) -> dict[str, str]:
                observed = _observed_from_gate(payload) or {}
                return {
                    # Record the launched C2 candidate id (cuda / cuda-sm90 / triton).
                    "backend": str(candidate),
                    "kernel": observed.get("kernel") or str(resolved.get("candidate_path") or ""),
                }

            invariance[(profile, op_name)] = {
                "forward_invariance": (fwd_status, fwd_detail, _actual(c3_payload)),
                "gradient_invariance": (grad_status, grad_detail, _actual(c4_payload)),
            }

    accuracy: dict[tuple[str, str], dict[str, tuple[str, str, dict[str, Any] | None]]] = {}
    case_by_id = {case["case_id"]: case for case in manifest.representative_cases}
    for cell in base.cells:
        if not cell.judgment.endswith("accuracy") or not cell.case_id:
            continue
        key = (cell.profile, cell.case_id)
        if key in accuracy:
            continue
        case = case_by_id[cell.case_id]
        if cell.status in {"pending_hopper", "N/A"}:
            continue
        g_code, g_out = _run_case_gate(cell.case_id, cell.profile, gradient=True)
        try:
            payload, _ = json.JSONDecoder().raw_decode(g_out[g_out.index("{") :])
            case_result = payload["cases"][0]
            judgment_status = case_result.get("judgment_status", {})
            resource_blocked = case_result.get("runtime_status") == "blocked_resource"
        except (ValueError, KeyError, IndexError, json.JSONDecodeError):
            case_result = {}
            judgment_status = {}
            resource_blocked = False
        actual = {
            "backend": str(case_result.get("actual_backend_id") or case["actual_backend_id"]),
            "kernel": str(
                case_result.get("actual_kernel_config_id") or case["actual_kernel_config_id"]
            ),
        }
        accuracy[key] = {
            "forward_accuracy": (
                ("green" if judgment_status.get("forward_accuracy") else "red"),
                (
                    "representative case forward accuracy passed"
                    if judgment_status.get("forward_accuracy")
                    else (
                        "required untested: resource blocked (OOM)"
                        if resource_blocked
                        else g_out[-400:]
                    )
                ),
                actual,
            ),
            "gradient_accuracy": (
                ("green" if judgment_status.get("gradient_accuracy") else "red"),
                (
                    "representative case gradient accuracy passed"
                    if judgment_status.get("gradient_accuracy")
                    else (
                        "required untested: resource blocked (OOM)"
                        if resource_blocked
                        else g_out[-400:]
                    )
                ),
                actual,
            ),
        }

    cells: list[MatrixCell] = []
    for cell in base.cells:
        actual: dict[str, Any] | None = None
        if cell.judgment.endswith("accuracy") and cell.case_id:
            acc_update = accuracy.get((cell.profile, cell.case_id), {}).get(cell.judgment)
            if acc_update is None:
                update = None
            else:
                status, detail, actual = acc_update
                update = (status, detail, actual)
        else:
            update = invariance.get((cell.profile, cell.op_name), {}).get(cell.judgment)
        if update is None:
            cells.append(cell)
            continue
        status, detail, actual = update
        cells.append(
            MatrixCell(
                profile=cell.profile,
                op_name=cell.op_name,
                judgment=cell.judgment,
                tier=cell.tier,
                case_id=cell.case_id,
                status=status,
                detail=detail,
                candidate=cell.candidate,
                expected_kernel_config_id=cell.expected_kernel_config_id,
                actual_backend_id=(None if actual is None else str(actual["backend"])),
                actual_kernel_config_id=(None if actual is None else str(actual["kernel"])),
                evidence_kind=cell.evidence_kind,
            )
        )
    counts: dict[str, int] = defaultdict(int)
    for cell in cells:
        counts[cell.status] += 1
    return MatrixReport(cells=tuple(cells), counts=dict(counts))


def _environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    try:
        import torch

        from rl_engine.kernels.gtest.accelerator import (
            compute_capability,
            device_name,
            is_available,
            npu_available,
        )

        info["pytorch"] = torch.__version__
        info["cuda_runtime"] = getattr(torch.version, "cuda", None)
        if npu_available():
            npu = torch.device("npu", 0)
            info["npu_name"] = device_name(npu)
            info["npu_soc"] = compute_capability(npu)
            try:
                import torch_npu

                info["torch_npu"] = getattr(torch_npu, "__version__", "unknown")
            except Exception:
                info["torch_npu"] = None
        if is_available("cuda"):
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["compute_capability"] = ".".join(
                str(x) for x in torch.cuda.get_device_capability(0)
            )
            try:
                info["driver"] = (
                    subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                        text=True,
                    )
                    .splitlines()[0]
                    .strip()
                )
            except Exception:
                info["driver"] = None
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = str(exc)
    try:
        import triton

        info["triton"] = getattr(triton, "__version__", "unknown")
    except Exception:
        info["triton"] = None
    return info


def _git_identity() -> dict[str, Any]:
    def _run(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""

    porcelain = _run("status", "--porcelain")
    ignored_suffixes = ("ws1-c8-ci.json", "ws1-c8-execute.json")
    dirty_lines = [
        line
        for line in porcelain.splitlines()
        if line.strip() and not any(line.endswith(suffix) for suffix in ignored_suffixes)
    ]
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty_lines),
    }


def _execute_payload(report: MatrixReport) -> dict[str, Any]:
    manifest = load_manifest()
    return {
        "schema_version": "ws1-c8-execute-v2",
        "git": _git_identity(),
        "environment": _environment(),
        "workload": {
            "workload_id": manifest.workload_id,
            "manifest_version": manifest.raw.get("version"),
            "fixture_identity_sha256": manifest.raw.get("fixture_identity_sha256"),
        },
        "command": "python scripts/sweep_ws1_four_judgments.py --execute --json",
        "threshold_source": "rl_engine/kernels/gtest/tolerance_contract.json",
        "fallback_policy": "forbidden; required untested is red; pack is N/A with C2/C4 reason",
        "counts": dict(report.counts),
        "cells": [cell.to_dict() for cell in report.cells],
    }


def _print_table(report: MatrixReport) -> None:
    grouped: dict[tuple[str, str], list[MatrixCell]] = defaultdict(list)
    for cell in report.cells:
        grouped[(cell.profile, cell.op_name)].append(cell)
    for (profile, op_name), cells in grouped.items():
        by_j = {cell.judgment: cell for cell in cells if cell.tier == "primary"}
        statuses = " ".join(
            f"{j.split('_')[0][0]}{j.split('_')[1][0]}={by_j[j].status}" for j in JUDGMENTS
        )
        sample = cells[0]
        print(
            f"{profile:<17} {op_name:<21} {sample.candidate or '-':<11} "
            f"{statuses}  {sample.detail}"
        )
    print("\n" + ", ".join(f"{k}={v}" for k, v in sorted(report.counts.items())))


def main() -> None:
    parser = argparse.ArgumentParser(description="WS1 C8 four-judgment matrix sweep")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run C3/C4 on runnable cells (requires the profile's accelerator). "
        "Default is classify-only.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=PROFILES,
        help="Backend profile to cover; repeatable. Defaults to every required "
        "profile. One host rarely has both a GPU and an NPU, so each vendor's "
        "CI job passes its own profiles here and C11 needs all jobs green.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--allow-pending-hopper",
        action="store_true",
        help="Do not fail when declared cuda-sm90 cells are pending_hopper (non-Hopper CI).",
    )
    args = parser.parse_args()

    profiles = tuple(args.profile) if args.profile else PROFILES
    report = build_classified_matrix(profiles=profiles)
    if args.execute:
        report = _execute_matrix(report, profiles)
    if args.json:
        payload = _execute_payload(report) if args.execute else report.to_dict()
        print(json.dumps(payload, indent=2))
    else:
        _print_table(report)
    if any(cell.status == "red" for cell in report.cells):
        raise SystemExit(1)
    if (
        any(cell.status == "pending_hopper" for cell in report.cells)
        and not args.allow_pending_hopper
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
