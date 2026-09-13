#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Run the WS1 C3 forward invariance gate on a real GPU."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.gtest import (  # noqa: E402
    BackendProvenance,
    assert_forward_batch_invariant,
    load_contract,
)
from rl_engine.kernels.gtest.accelerator import (  # noqa: E402
    arch_key,
    candidate_family,
    device_name,
    disable_tf32,
    resolve_device,
)
from rl_engine.kernels.gtest.forward_invariance import build_config_matrix  # noqa: E402
from rl_engine.kernels.gtest.gradient_adapters import (  # noqa: E402
    GRADIENT_ADAPTERS,
    get_adapter,
    load_adapter_gold,
    load_adapter_operator,
    make_forward_runner,
    resolve_profile_candidate,
)
from rl_engine.kernels.gtest.tolerance import resolve_dtype_policy  # noqa: E402
from rl_engine.testing.ws1_workload import load_manifest  # noqa: E402


def _object_path(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _candidate_family(candidate: str) -> str:
    return candidate_family(candidate)


def _validate_candidate_selection(
    *, manifest: Any, profile: str, op_name: str, candidate: str
) -> dict[str, Any]:
    adapter = get_adapter(op_name)
    resolved = resolve_profile_candidate(adapter, profile, manifest)
    if resolved["status"] == "missing_required":
        raise RuntimeError(
            f"profile {profile!r} node {adapter.chain_node!r} is missing_required; "
            "missing required candidates are red, not fallback or N/A"
        )
    if resolved["status"] == "absent_not_required":
        raise RuntimeError(f"adapter {op_name!r} is not declared supported and differentiable")
    expected_family = manifest.backend_profiles[profile]["backend_family"]
    actual_family = _candidate_family(candidate)
    if adapter.requirement != "layout_supported" and actual_family != expected_family:
        raise RuntimeError(
            f"candidate {candidate!r} belongs to {actual_family!r}, but profile "
            f"{profile!r} requires {expected_family!r}"
        )
    expected = resolved["expected_backend_id"]
    if expected is not None and candidate != expected:
        raise RuntimeError(
            f"candidate {candidate!r} does not match the C2 declaration "
            f"{expected!r} for {profile}/{adapter.chain_node}"
        )
    return resolved


def _summarize(report: Any) -> None:
    print(
        f"op={report.op_name} profile={report.backend_profile} "
        f"candidate={report.candidate_id} passed={report.passed}"
    )
    print(
        f"  device={report.device} cc={report.compute_capability} seed={report.seed} "
        f"provenance_valid={report.provenance_valid}"
    )
    for acc in report.accuracy_reports:
        detail = acc.details[0]
        print(
            f"  accuracy config={acc.config_id} max_abs={detail.max_abs_error:.8e} "
            f"max_rel={detail.max_rel_error:.8e} passed={acc.passed}"
        )
    for inv in report.invariance_reports:
        detail = inv.details[0]
        print(
            f"  invariance pair={detail.config_pair} transform={inv.transform_kind} "
            f"max_abs={detail.max_abs_error:.8e} passed={inv.passed}"
        )
    if report.logprob_smoke is not None:
        print(f"  selected_logprob_smoke passed={report.logprob_smoke.passed}")


def parse_args() -> argparse.Namespace:
    runnable = [
        name
        for name, adapter in GRADIENT_ADAPTERS.items()
        if adapter.requirement != "absent_not_required"
    ]
    parser = argparse.ArgumentParser(description="WS1 C3 forward invariance GPU gate")
    parser.add_argument("--op", choices=sorted(runnable), default="rms_norm")
    parser.add_argument(
        "--candidate", required=True, help="Manifest-declared CUDA/Triton/Ascend candidate"
    )
    parser.add_argument(
        "--backend-profile",
        choices=("cuda_bf16", "triton_cuda_bf16", "ascend_bf16"),
        required=True,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Defaults to the backend profile's own accelerator (cuda or npu).",
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        device = resolve_device(args.device, profile=args.backend_profile)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: C3 required-profile evidence needs a real device: {exc}") from exc
    if args.vocab <= 240:
        raise SystemExit("ERROR: --vocab must cover every fixed C2 workload token id")

    contract = load_contract()
    manifest = load_manifest()
    adapter = get_adapter(args.op)
    if adapter.requirement == "layout_supported":
        raise SystemExit(
            f"ERROR: {args.op!r} is layout_supported and profile-independent; "
            "per-profile GPU evidence would require fabricating backend provenance. "
            "Its forward contract is covered by tests/test_forward_invariance.py"
        )
    resolved = _validate_candidate_selection(
        manifest=manifest,
        profile=args.backend_profile,
        op_name=args.op,
        candidate=args.candidate,
    )
    cc = arch_key(device)
    if args.candidate == "cuda-sm90" and cc != "sm90":
        raise SystemExit(
            "ERROR: cuda-sm90 candidate requested on non-SM90 hardware; fallback forbidden"
        )

    candidate_op = load_adapter_operator(args.op, args.candidate)
    gold_fn = load_adapter_gold(args.op)
    policy = resolve_dtype_policy(contract)
    family = _candidate_family(args.candidate)
    tf32_enabled = disable_tf32(device.type)

    provenance = BackendProvenance(
        backend_profile=args.backend_profile,
        requested_backend=manifest.backend_profiles[args.backend_profile]["backend_family"],
        actual_backend=family,
        execution_dtype=policy.execution_dtype,
        accumulation_dtype=policy.accumulation_dtype,
        output_dtype=policy.output_dtype_default,
        reference_dtype=policy.reference_dtype,
        candidate_tf32_enabled=tf32_enabled,
        reference_tf32_enabled=tf32_enabled,
    )
    kernel_id = _object_path(candidate_op)
    shape_kwargs = {
        "hidden": args.hidden,
        "vocab_size": args.vocab,
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "head_dim": args.head_dim,
    }
    candidate_runner = make_forward_runner(
        args.op,
        candidate_op,
        device=device,
        dtype=torch.bfloat16,
        reference=False,
        backend_family=family,
        kernel_id=kernel_id,
        **shape_kwargs,
    )
    probe = candidate_runner(
        next(config for config in build_config_matrix(manifest) if config.is_canonical)
    )
    observed_dtype = probe.output_dtype
    report = assert_forward_batch_invariant(
        candidate_runner,
        contract=contract,
        manifest=manifest,
        backend_profile=args.backend_profile,
        provenance=provenance,
        gold_fn=make_forward_runner(
            args.op,
            gold_fn,
            device=device,
            dtype=torch.bfloat16,
            reference=True,
            **shape_kwargs,
        ),
        op_class=adapter.op_class,
        dtype=torch.bfloat16,
        op_name=args.op,
        include_logprob_smoke=adapter.op_class == "logprob",
        candidate_id=f"{kernel_id}::{resolved.get('expected_backend_id')}",
        device=f"{device}:{device_name(device)}",
        compute_capability=cc,
        observed_actual_backend=family,
        observed_kernel_id=kernel_id,
        observed_output_dtype=observed_dtype,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        _summarize(report)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
