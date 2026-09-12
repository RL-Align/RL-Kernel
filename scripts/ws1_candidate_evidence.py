#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Execute WS1 C2 representative CUDA/Triton cases and emit runtime provenance."""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.gtest import run_operator_suite  # noqa: E402
from rl_engine.kernels.gtest.operator_specs import make_candidate, make_operator_case  # noqa: E402
from rl_engine.testing.ws1_workload import WorkloadError, load_manifest  # noqa: E402


def _object_path(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _case_args(case: dict[str, Any], seed: int) -> SimpleNamespace:
    try:
        shape = case["shape"]
        operator_spec = case["operator_spec"]
    except KeyError as exc:
        raise WorkloadError(f"candidate case missing {exc.args[0]!r}") from exc
    common: dict[str, Any] = {
        "op": operator_spec,
        "candidate": case["expected_backend_id"],
        "arch_key": None,
        "input_mode": "random",
        "constant_value": 0.25,
        "token_value": 0,
        "normalized_dim": 4096,
        "k_dim": 4096,
        "n_dim": 4096,
        "theta": 1.0e6,
        "eps": 1.0e-6,
        "seed": seed,
    }
    try:
        if operator_spec == "det_gemm":
            common.update(batch=1, seq=shape["M"], k_dim=shape["K"], n_dim=shape["N"])
        elif operator_spec == "attention":
            common.update(
                batch=shape["B"],
                seq=shape["Sq"],
                skv=shape["Skv"],
                n_heads=shape["Hq"],
                n_kv_heads=shape["Hkv"],
                causal=1,
                use_padding=0,
                scale_mode="default",
            )
        elif operator_spec in {"logp", "batch_invariant_logp"}:
            common.update(batch=shape["B"], seq=shape["T"], vocab=shape["vocab"])
        elif operator_spec == "rms_norm":
            common.update(batch=1, seq=shape["T"], normalized_dim=4096)
        elif operator_spec == "qk_norm":
            common.update(
                batch=1,
                seq=shape["T"],
                n_heads=1,
                head_dim=int(shape.get("head_dim", 128)),
            )
        elif operator_spec in {"silu", "swiglu", "rope"}:
            common.update(batch=1, seq=shape["T"])
        elif operator_spec in {"embedding", "lm_head"}:
            common.update(
                batch=1,
                seq=shape["T"],
                normalized_dim=4096,
                vocab=151936,
            )
        else:
            raise WorkloadError(f"unsupported representative operator_spec {operator_spec!r}")
    except KeyError as exc:
        raise WorkloadError(
            f"case {case.get('case_id')!r} {operator_spec!r} shape missing {exc.args[0]!r}"
        ) from exc
    return SimpleNamespace(**common)


def run_case(
    case: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
    check_grad: bool = False,
) -> dict[str, Any]:
    args = _case_args(case, seed)
    candidate = make_candidate(args)
    actual_path = _object_path(candidate.fn)
    if candidate.backend != case["expected_backend_id"]:
        raise WorkloadError(
            f"case {case['case_id']} resolved backend {candidate.backend!r}, expected "
            f"{case['expected_backend_id']!r}"
        )
    if actual_path != case["expected_kernel_config_id"]:
        raise WorkloadError(
            f"case {case['case_id']} resolved kernel {actual_path!r}, expected "
            f"{case['expected_kernel_config_id']!r}"
        )

    operator_case = make_operator_case(args, torch.bfloat16, device)
    report = run_operator_suite(
        case["operator_spec"],
        candidates=[candidate],
        cases=[operator_case],
        check_grad=check_grad,
        grad_mode="random",
        grad_seed=seed + 1000,
    )
    synchronize(device)
    candidate_report = report.candidates[0]
    output_checks = [
        {
            "shape": list(output.shape),
            "dtype": output.candidate_dtype,
            "max_abs_error": output.max_abs_error,
            "judgment": output.judgment,
            "tensor": output.message,
            "passed": output.passed,
        }
        for checked_case in candidate_report.cases
        for output in checked_case.outputs
    ]
    return {
        "case_id": case["case_id"],
        "fixture_id": case["fixture_id"],
        "operator_spec": case["operator_spec"],
        "expected_backend_id": case["expected_backend_id"],
        "actual_backend_id": candidate.backend,
        "expected_kernel_config_id": case["expected_kernel_config_id"],
        "actual_kernel_config_id": actual_path,
        "algorithm_property": case["algorithm_property"],
        "shape": case["shape"],
        "runtime_status": "passed" if report.passed else "failed",
        "judgment_status": {
            judgment: all(item["passed"] for item in output_checks if item["judgment"] == judgment)
            for judgment in ("forward_accuracy", "gradient_accuracy")
            if any(item["judgment"] == judgment for item in output_checks)
        },
        "outputs": output_checks,
    }


from rl_engine.kernels.gtest.accelerator import (  # noqa: E402
    arch_key,
    device_name,
    device_type_for_profile,
    empty_cache,
    is_available,
    resolve_device,
    runtime_version,
    synchronize,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run manifest-pinned WS1 representative candidates on a real accelerator."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--profile",
        action="append",
        choices=("cuda_bf16", "triton_cuda_bf16", "ascend_bf16"),
        help="Profile to run; repeatable. Defaults to the profiles this host can run.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (e.g. cuda:0 or npu:0). Defaults to the selected "
        "profiles' accelerator.",
    )
    parser.add_argument("--case-id", action="append", help="Optional case_id filter.")
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Include C8 operator case_ids (norm/elementwise/embedding). "
            "Default is C2 gemm/attention/logprob only."
        ),
    )
    parser.add_argument(
        "--check-grad",
        action="store_true",
        help="Also run the manifest-pinned candidate-vs-FP32-reference VJP.",
    )
    parser.add_argument("--emit-json", default="-", help="Output path, or '-' for stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        if args.profile:
            profiles = set(args.profile)
        else:
            # One host has either a GPU or an NPU, never both; default to the
            # profiles its accelerator can actually execute rather than
            # reporting a fabricated pass for the other vendor.
            profiles = {
                name
                for name in ("cuda_bf16", "triton_cuda_bf16", "ascend_bf16")
                if is_available(device_type_for_profile(name))
            }
        if not profiles:
            print(
                "error: no accelerator available for runtime candidate evidence",
                file=sys.stderr,
            )
            return 2
        device_types = {device_type_for_profile(name) for name in profiles}
        if len(device_types) > 1:
            print(
                f"error: profiles {sorted(profiles)} span device types "
                f"{sorted(device_types)}; run one device type per invocation",
                file=sys.stderr,
            )
            return 2
        selected_ids = set(args.case_id or ())
        default_families = {"gemm", "attention", "logprob"}
        cases = [
            case
            for case in manifest.representative_cases
            if profiles.intersection(case["profile_ids"])
            and (not selected_ids or case["case_id"] in selected_ids)
            and (args.all or selected_ids or case["family"] in default_families)
        ]
        resolved_ids = {case["case_id"] for case in cases}
        if selected_ids - resolved_ids:
            unknown = sorted(selected_ids - resolved_ids)
            raise WorkloadError(f"unknown or profile-filtered case IDs: {unknown}")
        device = resolve_device(args.device, profile=sorted(profiles)[0])
        log_stream = sys.stderr if args.emit_json == "-" else sys.stdout
        with contextlib.redirect_stdout(log_stream):
            results = []
            for i, case in enumerate(cases):
                try:
                    results.append(
                        run_case(
                            case,
                            seed=manifest.seed + i,
                            device=device,
                            check_grad=args.check_grad,
                        )
                    )
                    empty_cache(device.type)
                except RuntimeError as exc:
                    message = str(exc)
                    if "out of memory" not in message.lower():
                        raise
                    # A 6 GiB card cannot materialize the pinned full-vocab
                    # candidate/reference pair. Preserve the case-level
                    # evidence and continue; this is a resource blocker, never
                    # a pass or a silent fallback.
                    empty_cache(device.type)
                    results.append(
                        {
                            "case_id": case["case_id"],
                            "fixture_id": case["fixture_id"],
                            "operator_spec": case["operator_spec"],
                            "expected_backend_id": case["expected_backend_id"],
                            "actual_backend_id": case["actual_backend_id"],
                            "expected_kernel_config_id": case["expected_kernel_config_id"],
                            "actual_kernel_config_id": case["actual_kernel_config_id"],
                            "algorithm_property": case["algorithm_property"],
                            "shape": case["shape"],
                            "runtime_status": "blocked_resource",
                            "error": message,
                            "judgment_status": {},
                            "outputs": [],
                        }
                    )
        fixture_identity_sha256 = manifest.raw["fixture_identity_sha256"]
        payload = {
            "schema_version": "ws1-c2-runtime-provenance-v1",
            "workload_id": manifest.workload_id,
            "fixture_identity_sha256": fixture_identity_sha256,
            "execution_dtype": "bfloat16",
            "device": {
                "index": device.index,
                "type": device.type,
                "name": device_name(device),
                "compute_capability": arch_key(device),
                "execution_world_size": 1,
            },
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "accelerator_runtime": runtime_version(device.type),
            },
            "profiles": sorted(profiles),
            "passed": bool(results)
            and all(result["runtime_status"] == "passed" for result in results),
            "cases": results,
        }
    except (
        RuntimeError,
        ValueError,
        WorkloadError,
        KeyError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.emit_json == "-":
        sys.stdout.write(rendered)
    else:
        path = Path(args.emit_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"wrote: {path}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
