# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 numerical contract loader and resolver (#267 / C1 of #266).

This module is the sole authority for:
- dtype policy (BF16 execution, FP32 reference/accumulation, FP8 out)
- four-judgment tolerances
- comparison roles
- chain-level logprob aggregates (max_abs_dlogp / approx_kl0 / clipfrac0)

Gates must obtain thresholds only through the resolvers defined here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_CONTRACT_PATH = Path(__file__).with_name("tolerance_contract.json")

JUDGMENTS = (
    "forward_accuracy",
    "forward_invariance",
    "gradient_accuracy",
    "gradient_invariance",
)
OP_CLASSES = ("elementwise", "reduction", "logprob", "attention")
MANDATORY_DTYPES = ("float32", "bfloat16")
OPTIONAL_DTYPES = ("float16",)
OUT_OF_SCOPE_DTYPES = ("float8",)
ALL_DTYPES = MANDATORY_DTYPES + OPTIONAL_DTYPES + OUT_OF_SCOPE_DTYPES
CHAIN_AGGREGATE_METRICS = ("max_abs_dlogp", "approx_kl0", "clipfrac0")
INVARIANCE_JUDGMENTS = ("forward_invariance", "gradient_invariance")
REPORT_KINDS = (
    "forward_accuracy",
    "forward_invariance",
    "train_infer_logprob_parity",
    "gradient_accuracy",
    "gradient_invariance",
)


class ContractError(ValueError):
    """Base error for contract load / resolve failures."""


class ContractSchemaError(ContractError):
    """Contract JSON failed schema validation."""


class ContractResolveError(ContractError):
    """A resolve request cannot be satisfied under the contract."""


@dataclass(frozen=True)
class DtypePolicy:
    """Resolved WS1 dtype / TF32 / FP8 policy."""

    execution_dtype: str
    accumulation_dtype: str
    reference_dtype: str
    output_dtype_default: str
    logprob_aggregates_dtype: str
    fp8: str
    fp16_status: str
    tf32_reference: str
    tf32_candidate_execution: str
    backend_profiles: tuple[str, ...]
    backend_private_tolerance_relaxation: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendProvenance:
    """Actual backend and dtype facts persisted by a WS1 report."""

    backend_profile: str
    requested_backend: str
    actual_backend: str
    execution_dtype: str
    accumulation_dtype: str
    output_dtype: str
    reference_dtype: str
    candidate_tf32_enabled: bool
    reference_tf32_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToleranceSupport:
    """Schema-level support result, including explicit N/A and out-of-scope cells."""

    judgment: str
    op_class: str
    dtype_name: str
    status: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonRoles:
    """lhs/rhs roles for a report kind."""

    report_kind: str
    comparison_lhs_role: str
    comparison_rhs_role: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToleranceSpec:
    """Resolved tolerance for one (judgment, op_class, dtype) request."""

    judgment: str
    op_class: str
    dtype_name: str
    status: str
    mode: str
    atol: float
    rtol: float
    comparison_lhs_role: str
    comparison_rhs_role: str
    backend_profile: str | None = None
    arch_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LogprobAggregates:
    """Three chain-level logprob aggregates (FP32)."""

    max_abs_dlogp: float
    approx_kl0: float
    clipfrac0: float
    active_token_count: int
    clip_interval: tuple[float, float]
    report_kind: str
    comparison_lhs_role: str
    comparison_rhs_role: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["clip_interval"] = list(self.clip_interval)
        return data


@dataclass(frozen=True)
class AggregateMetricVerdict:
    metric: str
    value: float
    threshold: float
    passed: bool


@dataclass(frozen=True)
class LogprobAggregateVerdict:
    aggregates: LogprobAggregates
    metrics: tuple[AggregateMetricVerdict, ...]
    passed: bool
    report_kind: str
    comparison_lhs_role: str
    comparison_rhs_role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregates": self.aggregates.to_dict(),
            "metrics": [asdict(m) for m in self.metrics],
            "passed": self.passed,
            "report_kind": self.report_kind,
            "comparison_lhs_role": self.comparison_lhs_role,
            "comparison_rhs_role": self.comparison_rhs_role,
        }


def load_contract(
    path: str | Path = _CONTRACT_PATH,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Load the WS1 dtype/operator-class tolerance contract."""

    with Path(path).open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    if validate:
        validate_contract_schema(contract)
    return contract


def validate_contract_schema(contract: Mapping[str, Any]) -> None:
    """Validate four-judgment schema, dtype policy, roles, and aggregates."""

    if not isinstance(contract, Mapping):
        raise ContractSchemaError("contract must be a mapping")

    for key in (
        "version",
        "policy",
        "comparison_roles",
        "judgments",
        "chain_logprob_aggregates",
    ):
        if key not in contract:
            raise ContractSchemaError(f"contract missing required key {key!r}")

    _validate_policy(contract["policy"])
    _validate_comparison_roles(contract["comparison_roles"])
    _validate_judgments(contract["judgments"])
    _validate_chain_aggregates(contract["chain_logprob_aggregates"])
    _validate_compat_views(contract)


def resolve_dtype_policy(contract: Mapping[str, Any]) -> DtypePolicy:
    """Resolve independent execution / accumulation / output / reference dtypes."""

    policy = contract["policy"]
    output = policy["output_dtype"]
    tf32 = policy["tf32"]
    fp16 = policy["fp16"]
    return DtypePolicy(
        execution_dtype=str(policy["execution_dtype"]),
        accumulation_dtype=str(policy["accumulation_dtype"]),
        reference_dtype=str(policy["reference_dtype"]),
        output_dtype_default=(
            str(policy["execution_dtype"])
            if output["default"] == "execution"
            else str(output["default"])
        ),
        logprob_aggregates_dtype=str(output["logprob_aggregates"]),
        fp8=str(policy["fp8"]),
        fp16_status=str(fp16["status"]),
        tf32_reference=str(tf32["reference"]),
        tf32_candidate_execution=str(tf32["candidate_execution"]),
        backend_profiles=tuple(str(p) for p in policy["backend_profiles"]),
        backend_private_tolerance_relaxation=bool(policy["backend_private_tolerance_relaxation"]),
    )


def validate_backend_provenance(
    contract: Mapping[str, Any],
    provenance: BackendProvenance,
) -> BackendProvenance:
    """Fail closed when reported backend or dtype facts violate the WS1 profile."""

    policy = resolve_dtype_policy(contract)
    if provenance.backend_profile not in policy.backend_profiles:
        raise ContractResolveError(f"unknown backend_profile {provenance.backend_profile!r}")
    profile_contracts = contract["policy"]["backend_profile_contracts"]
    if provenance.backend_profile not in profile_contracts:
        raise ContractResolveError(
            f"missing backend_profile_contracts entry for {provenance.backend_profile!r}"
        )
    profile_contract = profile_contracts[provenance.backend_profile]
    expected_backend = str(profile_contract["backend_family"])
    for field_name, actual in (
        ("requested_backend", provenance.requested_backend),
        ("actual_backend", provenance.actual_backend),
    ):
        if actual != expected_backend:
            raise ContractResolveError(
                f"backend provenance mismatch for {field_name}: expected "
                f"{expected_backend!r}, got {actual!r}"
            )

    expected_dtypes = {
        "execution_dtype": policy.execution_dtype,
        "accumulation_dtype": policy.accumulation_dtype,
        "output_dtype": policy.output_dtype_default,
        "reference_dtype": policy.reference_dtype,
    }
    for field_name, expected in expected_dtypes.items():
        actual = normalize_dtype_name(getattr(provenance, field_name))
        if actual != expected:
            raise ContractResolveError(
                f"backend provenance mismatch for {field_name}: expected "
                f"{expected!r}, got {actual!r}"
            )
    for field_name in ("candidate_tf32_enabled", "reference_tf32_enabled"):
        if getattr(provenance, field_name):
            raise ContractResolveError(
                f"backend provenance reports {field_name}=true; WS1 requires disabled"
            )
    return provenance


def resolve_comparison_roles(
    contract: Mapping[str, Any],
    report_kind: str,
) -> ComparisonRoles:
    """Return lhs/rhs roles for a report kind."""

    roles_root = contract["comparison_roles"]
    forbidden = set(roles_root.get("forbidden", ()))
    by_kind = roles_root["by_report_kind"]
    if report_kind not in by_kind:
        raise ContractResolveError(f"unknown report_kind {report_kind!r}")
    entry = by_kind[report_kind]
    lhs = str(entry["comparison_lhs_role"])
    rhs = str(entry["comparison_rhs_role"])
    for role in (lhs, rhs):
        if role in forbidden:
            raise ContractResolveError(
                f"forbidden comparison role {role!r} for report_kind {report_kind!r}"
            )
        if role not in roles_root["allowed"]:
            raise ContractResolveError(
                f"unknown comparison role {role!r} for report_kind {report_kind!r}"
            )
    return ComparisonRoles(
        report_kind=report_kind,
        comparison_lhs_role=lhs,
        comparison_rhs_role=rhs,
    )


def assert_comparison_roles(
    contract: Mapping[str, Any],
    report_kind: str,
    comparison_lhs_role: str,
    comparison_rhs_role: str,
) -> ComparisonRoles:
    """Hard-fail if report roles are reversed, unknown, or forbidden."""

    expected = resolve_comparison_roles(contract, report_kind)
    if comparison_lhs_role in contract["comparison_roles"].get("forbidden", ()):
        raise ContractResolveError(f"forbidden comparison_lhs_role {comparison_lhs_role!r}")
    if comparison_rhs_role in contract["comparison_roles"].get("forbidden", ()):
        raise ContractResolveError(f"forbidden comparison_rhs_role {comparison_rhs_role!r}")
    if (
        comparison_lhs_role != expected.comparison_lhs_role
        or comparison_rhs_role != expected.comparison_rhs_role
    ):
        raise ContractResolveError(
            f"role mismatch for {report_kind!r}: expected "
            f"lhs={expected.comparison_lhs_role!r}, rhs={expected.comparison_rhs_role!r}; "
            f"got lhs={comparison_lhs_role!r}, rhs={comparison_rhs_role!r}"
        )
    return expected


def resolve_tolerance(
    contract: Mapping[str, Any],
    *,
    judgment: str,
    op_class: str,
    dtype: str | Any,
    arch_key: str | None = None,
    backend_profile: str | None = None,
) -> ToleranceSpec:
    """Resolve one four-judgment tolerance cell.

    ``cuda_bf16`` and ``triton_cuda_bf16`` share the same rows. Backend-private
    threshold relaxation is forbidden.
    """

    if judgment not in JUDGMENTS:
        raise ContractResolveError(f"unknown judgment {judgment!r}")
    if op_class not in OP_CLASSES:
        raise ContractResolveError(f"unknown op_class {op_class!r}")

    dtype_name = _dtype_name(dtype)
    policy = resolve_dtype_policy(contract)

    if backend_profile is not None:
        if backend_profile not in policy.backend_profiles:
            raise ContractResolveError(
                f"unknown backend_profile {backend_profile!r}; "
                f"allowed={list(policy.backend_profiles)}"
            )
        if policy.backend_private_tolerance_relaxation:
            raise ContractResolveError(
                "backend_private_tolerance_relaxation must remain false under WS1 C1"
            )

    if dtype_name in OUT_OF_SCOPE_DTYPES:
        raise ContractResolveError(
            f"dtype {dtype_name!r} is out of scope for WS1 (FP8 requests hard-fail)"
        )

    support = resolve_tolerance_support(
        contract,
        judgment=judgment,
        op_class=op_class,
        dtype=dtype_name,
        arch_key=arch_key,
    )
    judgment_root = contract["judgments"][judgment]
    cell = _lookup_cell(judgment_root, op_class=op_class, dtype_name=dtype_name, arch_key=arch_key)
    if cell is None:
        raise ContractResolveError(
            f"missing declared cell for judgment={judgment!r}, "
            f"op_class={op_class!r}, dtype={dtype_name!r}"
        )

    status = support.status
    if status == "out_of_scope":
        raise ContractResolveError(
            f"cell out_of_scope for judgment={judgment!r}, "
            f"op_class={op_class!r}, dtype={dtype_name!r}"
        )
    if status == "not_applicable":
        raise ContractResolveError(
            f"cell not_applicable for judgment={judgment!r}, "
            f"op_class={op_class!r}, dtype={dtype_name!r}; "
            "callers must not request non-applicable judgments without an explicit N/A path"
        )
    if status not in {"applicable", "optional"}:
        raise ContractResolveError(
            f"invalid status {status!r} for judgment={judgment!r}, "
            f"op_class={op_class!r}, dtype={dtype_name!r}"
        )

    mode = str(cell.get("mode", judgment_root.get("default_mode", "tolerance")))
    if "atol" not in cell or "rtol" not in cell:
        raise ContractResolveError(
            f"cell missing atol/rtol for judgment={judgment!r}, "
            f"op_class={op_class!r}, dtype={dtype_name!r}"
        )
    atol = float(cell["atol"])
    rtol = float(cell["rtol"])

    if judgment in INVARIANCE_JUDGMENTS and status in {"applicable", "optional"}:
        if mode != "bitwise" or atol != 0.0 or rtol != 0.0:
            raise ContractResolveError(
                f"Batch/Chunk invariance requires bitwise atol=0 rtol=0; got "
                f"mode={mode!r}, atol={atol}, rtol={rtol} for {judgment}/{op_class}/{dtype_name}"
            )

    roles = resolve_comparison_roles(contract, judgment)
    return ToleranceSpec(
        judgment=judgment,
        op_class=op_class,
        dtype_name=dtype_name,
        status=status,
        mode=mode,
        atol=atol,
        rtol=rtol,
        comparison_lhs_role=roles.comparison_lhs_role,
        comparison_rhs_role=roles.comparison_rhs_role,
        backend_profile=backend_profile,
        arch_key=arch_key,
    )


def resolve_tolerance_support(
    contract: Mapping[str, Any],
    *,
    judgment: str,
    op_class: str,
    dtype: str | Any,
    arch_key: str | None = None,
) -> ToleranceSupport:
    """Resolve schema support without pretending N/A cells have thresholds."""

    if judgment not in JUDGMENTS:
        raise ContractResolveError(f"unknown judgment {judgment!r}")
    if op_class not in OP_CLASSES:
        raise ContractResolveError(f"unknown op_class {op_class!r}")
    dtype_name = _dtype_name(dtype)
    cell = _lookup_cell(
        contract["judgments"][judgment],
        op_class=op_class,
        dtype_name=dtype_name,
        arch_key=arch_key,
    )
    if cell is None:
        raise ContractResolveError(
            f"missing declared cell for judgment={judgment!r}, "
            f"op_class={op_class!r}, dtype={dtype_name!r}"
        )
    status = str(cell.get("status", ""))
    if status not in {"applicable", "optional", "not_applicable", "out_of_scope"}:
        raise ContractResolveError(
            f"invalid support status {status!r} for {judgment}/{op_class}/{dtype_name}"
        )
    reason = cell.get("reason")
    return ToleranceSupport(
        judgment=judgment,
        op_class=op_class,
        dtype_name=dtype_name,
        status=status,
        reason=str(reason) if reason is not None else None,
    )


def resolve_chain_aggregate_thresholds(
    contract: Mapping[str, Any],
    metric_name: str,
    execution_dtype: str | Any,
) -> float:
    """Named resolve for max_abs_dlogp / approx_kl0 / clipfrac0 thresholds."""

    if metric_name not in CHAIN_AGGREGATE_METRICS:
        raise ContractResolveError(
            f"unknown chain aggregate metric {metric_name!r}; "
            f"only {list(CHAIN_AGGREGATE_METRICS)} are allowed"
        )
    dtype_name = _dtype_name(execution_dtype)
    metrics = contract["chain_logprob_aggregates"]["metrics"]
    by_dtype = metrics[metric_name]["by_execution_dtype"]
    if dtype_name not in by_dtype:
        raise ContractResolveError(
            f"missing chain aggregate threshold for metric={metric_name!r}, "
            f"execution_dtype={dtype_name!r}"
        )
    return float(by_dtype[dtype_name]["threshold"])


def compute_logprob_aggregates(
    lhs_logp: Any,
    rhs_logp: Any,
    active_mask: Any,
    *,
    contract: Mapping[str, Any],
    report_kind: str,
    clip_interval: Sequence[float] | tuple[float, float],
    comparison_lhs_role: str,
    comparison_rhs_role: str,
) -> LogprobAggregates:
    """Compute the three chain-level logprob aggregates in FP32.

    ``dlogp = lhs_logp - rhs_logp`` on active selected tokens only.
    Empty active set / NaN / Inf → hard fail.
    """

    assert_comparison_roles(contract, report_kind, comparison_lhs_role, comparison_rhs_role)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ContractResolveError("torch is required for aggregate computation") from exc

    if len(clip_interval) != 2:
        raise ContractResolveError("clip_interval must be a length-2 [lo, hi] pair")
    lo, hi = float(clip_interval[0]), float(clip_interval[1])
    if not (lo < hi):
        raise ContractResolveError(f"clip_interval requires lo < hi, got [{lo}, {hi}]")

    lhs_tensor = torch.as_tensor(lhs_logp).detach().float()
    rhs_tensor = torch.as_tensor(rhs_logp).detach().float()
    mask_tensor = torch.as_tensor(active_mask).detach().bool()
    if lhs_tensor.shape != rhs_tensor.shape or lhs_tensor.shape != mask_tensor.shape:
        raise ContractResolveError(
            f"lhs/rhs/mask shape mismatch: {tuple(lhs_tensor.shape)} vs "
            f"{tuple(rhs_tensor.shape)} vs {tuple(mask_tensor.shape)}"
        )
    lhs = lhs_tensor.reshape(-1)
    rhs = rhs_tensor.reshape(-1)
    mask = mask_tensor.reshape(-1)
    active = int(mask.sum().item())
    if active == 0:
        raise ContractResolveError("empty active-token set is a hard fail for logprob aggregates")

    dlogp = lhs[mask] - rhs[mask]
    if not torch.isfinite(dlogp).all():
        raise ContractResolveError("NaN/Inf in dlogp is a hard fail for logprob aggregates")

    ratio0 = torch.exp(dlogp)
    if not torch.isfinite(ratio0).all():
        raise ContractResolveError("NaN/Inf in ratio0 is a hard fail for logprob aggregates")

    max_abs_dlogp = float(dlogp.abs().max().item())
    approx_kl0 = float((ratio0 - 1.0 - dlogp).mean().item())
    outside = (ratio0 < lo) | (ratio0 > hi)
    clipfrac0 = float(outside.float().mean().item())

    for name, value in (
        ("max_abs_dlogp", max_abs_dlogp),
        ("approx_kl0", approx_kl0),
        ("clipfrac0", clipfrac0),
    ):
        if not math.isfinite(value):
            raise ContractResolveError(f"NaN/Inf in aggregate {name} is a hard fail")

    return LogprobAggregates(
        max_abs_dlogp=max_abs_dlogp,
        approx_kl0=approx_kl0,
        clipfrac0=clipfrac0,
        active_token_count=active,
        clip_interval=(lo, hi),
        report_kind=report_kind,
        comparison_lhs_role=comparison_lhs_role,
        comparison_rhs_role=comparison_rhs_role,
    )


def judge_logprob_aggregates(
    aggregates: LogprobAggregates,
    contract: Mapping[str, Any],
    *,
    execution_dtype: str | Any,
    clip_interval: Sequence[float] | tuple[float, float] | None = None,
) -> LogprobAggregateVerdict:
    """Judge all three chain logprob aggregates; all must pass."""

    assert_comparison_roles(
        contract,
        aggregates.report_kind,
        aggregates.comparison_lhs_role,
        aggregates.comparison_rhs_role,
    )

    if clip_interval is not None:
        lo, hi = float(clip_interval[0]), float(clip_interval[1])
        if (lo, hi) != aggregates.clip_interval:
            raise ContractResolveError(
                "clip_interval mismatch between compute and judge "
                f"(computed={aggregates.clip_interval}, judge=({lo}, {hi}))"
            )

    metrics: list[AggregateMetricVerdict] = []
    for name in CHAIN_AGGREGATE_METRICS:
        threshold = resolve_chain_aggregate_thresholds(contract, name, execution_dtype)
        value = float(getattr(aggregates, name))
        if not math.isfinite(value):
            raise ContractResolveError(f"NaN/Inf in aggregate {name} is a hard fail")
        metrics.append(
            AggregateMetricVerdict(
                metric=name,
                value=value,
                threshold=threshold,
                passed=value <= threshold,
            )
        )
    require_all = bool(contract["chain_logprob_aggregates"].get("require_all", True))
    passed = all(m.passed for m in metrics) if require_all else any(m.passed for m in metrics)
    return LogprobAggregateVerdict(
        aggregates=aggregates,
        metrics=tuple(metrics),
        passed=passed,
        report_kind=aggregates.report_kind,
        comparison_lhs_role=aggregates.comparison_lhs_role,
        comparison_rhs_role=aggregates.comparison_rhs_role,
    )


def default_clip_interval(contract: Mapping[str, Any]) -> tuple[float, float]:
    """Return the contract default clip interval for clipfrac0."""

    interval = contract["chain_logprob_aggregates"]["default_clip_interval"]
    return float(interval[0]), float(interval[1])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_policy(policy: Mapping[str, Any]) -> None:
    required = (
        "execution_dtype",
        "accumulation_dtype",
        "reference_dtype",
        "output_dtype",
        "fp8",
        "fp16",
        "tf32",
        "backend_profiles",
        "backend_profile_contracts",
        "backend_private_tolerance_relaxation",
    )
    for key in required:
        if key not in policy:
            raise ContractSchemaError(f"policy missing {key!r}")
    if policy["execution_dtype"] != "bfloat16":
        raise ContractSchemaError("policy.execution_dtype must be bfloat16 for WS1")
    if policy["accumulation_dtype"] != "float32":
        raise ContractSchemaError("policy.accumulation_dtype must be float32 for WS1")
    if policy["reference_dtype"] != "float32":
        raise ContractSchemaError("policy.reference_dtype must be float32 for WS1")
    if policy["fp8"] != "out_of_scope":
        raise ContractSchemaError("policy.fp8 must be out_of_scope for WS1")
    output = policy["output_dtype"]
    for key in ("default", "logprob_aggregates"):
        if key not in output:
            raise ContractSchemaError(f"policy.output_dtype missing {key!r}")
    if output["logprob_aggregates"] != "float32":
        raise ContractSchemaError("logprob aggregates must be computed in float32")
    if output["default"] != "execution":
        raise ContractSchemaError("policy.output_dtype.default must follow execution")
    if policy["fp16"].get("status") != "optional":
        raise ContractSchemaError("policy.fp16.status must be optional for WS1")
    tf32 = policy["tf32"]
    for key in ("reference", "candidate_execution"):
        if key not in tf32:
            raise ContractSchemaError(f"policy.tf32 missing {key!r}")
        if tf32[key] != "disabled":
            raise ContractSchemaError(
                f"policy.tf32.{key} must be 'disabled' under the WS1 single policy"
            )
    profiles = list(policy["backend_profiles"])
    profile_contracts = policy["backend_profile_contracts"]
    required_profile_families = {
        "cuda_bf16": "cuda",
        "triton_cuda_bf16": "triton",
        "ascend_bf16": "ascend",
    }
    for required_profile, expected_family in required_profile_families.items():
        if required_profile not in profiles:
            raise ContractSchemaError(f"policy.backend_profiles must include {required_profile!r}")
        if required_profile not in profile_contracts:
            raise ContractSchemaError(
                f"policy.backend_profile_contracts missing {required_profile!r}"
            )
        actual_family = profile_contracts[required_profile].get("backend_family")
        if actual_family != expected_family:
            raise ContractSchemaError(
                f"profile {required_profile!r} requires backend_family "
                f"{expected_family!r}, got {actual_family!r}"
            )
    if policy["backend_private_tolerance_relaxation"] is not False:
        raise ContractSchemaError("backend_private_tolerance_relaxation must be false")


def _validate_comparison_roles(roles_root: Mapping[str, Any]) -> None:
    for key in ("allowed", "forbidden", "by_report_kind"):
        if key not in roles_root:
            raise ContractSchemaError(f"comparison_roles missing {key!r}")
    forbidden = set(roles_root["forbidden"])
    for name in ("baseline", "singleton_aggregate"):
        if name not in forbidden:
            raise ContractSchemaError(f"comparison_roles.forbidden must include {name!r}")
    by_kind = roles_root["by_report_kind"]
    for kind in REPORT_KINDS:
        if kind not in by_kind:
            raise ContractSchemaError(f"comparison_roles.by_report_kind missing {kind!r}")
        entry = by_kind[kind]
        for role_key in ("comparison_lhs_role", "comparison_rhs_role"):
            if role_key not in entry:
                raise ContractSchemaError(
                    f"comparison_roles.by_report_kind[{kind!r}] missing {role_key!r}"
                )
            role = entry[role_key]
            if role in forbidden:
                raise ContractSchemaError(f"report_kind {kind!r} uses forbidden role {role!r}")
            if role not in roles_root["allowed"]:
                raise ContractSchemaError(f"report_kind {kind!r} uses unknown role {role!r}")


def _validate_judgments(judgments: Mapping[str, Any]) -> None:
    for judgment in JUDGMENTS:
        if judgment not in judgments:
            raise ContractSchemaError(f"judgments missing {judgment!r}")
        root = judgments[judgment]
        if "by_op_class" not in root:
            raise ContractSchemaError(f"judgments[{judgment!r}] missing by_op_class")
        by_op = root["by_op_class"]
        for op_class in OP_CLASSES:
            if op_class not in by_op:
                raise ContractSchemaError(
                    f"judgments[{judgment!r}].by_op_class missing {op_class!r}"
                )
            dtype_map = by_op[op_class]
            for dtype_name in ALL_DTYPES:
                if dtype_name not in dtype_map:
                    raise ContractSchemaError(
                        f"missing cell judgments[{judgment!r}][{op_class!r}][{dtype_name!r}]"
                    )
                cell = dtype_map[dtype_name]
                status = cell.get("status")
                if status is None:
                    raise ContractSchemaError(
                        f"cell missing status: {judgment}/{op_class}/{dtype_name}"
                    )
                if dtype_name in OUT_OF_SCOPE_DTYPES:
                    if status != "out_of_scope":
                        raise ContractSchemaError(
                            f"FP8 cell must be out_of_scope: {judgment}/{op_class}/{dtype_name}"
                        )
                    continue
                if status == "not_applicable" and not cell.get("reason"):
                    raise ContractSchemaError(
                        f"not_applicable cell requires reason: {judgment}/{op_class}/{dtype_name}"
                    )
                if dtype_name in MANDATORY_DTYPES and status not in {
                    "applicable",
                    "not_applicable",
                }:
                    raise ContractSchemaError(
                        f"mandatory dtype cell must be applicable: "
                        f"{judgment}/{op_class}/{dtype_name} status={status!r}"
                    )
                if status in {"applicable", "optional"}:
                    for thr in ("atol", "rtol", "mode"):
                        if thr not in cell:
                            raise ContractSchemaError(
                                f"cell missing {thr}: {judgment}/{op_class}/{dtype_name}"
                            )
                if judgment in INVARIANCE_JUDGMENTS and status in {"applicable", "optional"}:
                    mode = cell.get("mode")
                    atol = float(cell.get("atol", 1.0))
                    rtol = float(cell.get("rtol", 1.0))
                    if mode != "bitwise" or atol != 0.0 or rtol != 0.0:
                        raise ContractSchemaError(
                            f"invariance applicable/optional cells must be bitwise 0/0: "
                            f"{judgment}/{op_class}/{dtype_name}"
                        )


def _validate_chain_aggregates(root: Mapping[str, Any]) -> None:
    for key in (
        "compute_dtype",
        "require_all",
        "nan_inf_policy",
        "empty_active_token_set",
        "active_token_policy",
        "clip_interval_field",
        "dlogp_definition",
        "default_clip_interval",
        "sole_chain_level_logprob_metrics",
        "metrics",
    ):
        if key not in root:
            raise ContractSchemaError(f"chain_logprob_aggregates missing {key!r}")
    if root["compute_dtype"] != "float32":
        raise ContractSchemaError("chain aggregates must use compute_dtype=float32")
    if root["nan_inf_policy"] != "hard_fail":
        raise ContractSchemaError("nan_inf_policy must be hard_fail")
    if root["empty_active_token_set"] != "hard_fail":
        raise ContractSchemaError("empty_active_token_set must be hard_fail")
    if root["active_token_policy"] != "active selected tokens only":
        raise ContractSchemaError("active_token_policy must be 'active selected tokens only'")
    if root["clip_interval_field"] != "clip_interval":
        raise ContractSchemaError("clip_interval_field must be 'clip_interval'")
    if root["dlogp_definition"] != "comparison_lhs_logp - comparison_rhs_logp":
        raise ContractSchemaError("dlogp_definition does not match implementation")
    if not root["require_all"]:
        raise ContractSchemaError("require_all must be true for chain logprob aggregates")
    sole = list(root["sole_chain_level_logprob_metrics"])
    if set(sole) != set(CHAIN_AGGREGATE_METRICS) or len(sole) != 3:
        raise ContractSchemaError(
            "sole_chain_level_logprob_metrics must be exactly " f"{list(CHAIN_AGGREGATE_METRICS)}"
        )
    interval = root["default_clip_interval"]
    if len(interval) != 2 or float(interval[0]) >= float(interval[1]):
        raise ContractSchemaError("default_clip_interval must be [lo, hi] with lo < hi")
    metrics = root["metrics"]
    for name in CHAIN_AGGREGATE_METRICS:
        if name not in metrics:
            raise ContractSchemaError(f"chain metrics missing {name!r}")
        by_dtype = metrics[name].get("by_execution_dtype")
        if not isinstance(by_dtype, Mapping):
            raise ContractSchemaError(f"metric {name!r} missing by_execution_dtype")
        for dtype_name in ("bfloat16", "float32"):
            if dtype_name not in by_dtype or "threshold" not in by_dtype[dtype_name]:
                raise ContractSchemaError(f"metric {name!r} missing threshold for {dtype_name}")
        expected_formula = {
            "max_abs_dlogp": "max(abs(dlogp))",
            "approx_kl0": "mean(exp(dlogp) - 1 - dlogp)",
            "clipfrac0": "mean(1[exp(dlogp) outside clip_interval])",
        }[name]
        if metrics[name].get("formula") != expected_formula:
            raise ContractSchemaError(f"metric {name!r} formula does not match implementation")
        if metrics[name].get("pass_rule") != "value <= threshold":
            raise ContractSchemaError(f"metric {name!r} pass_rule must be 'value <= threshold'")


def _validate_compat_views(contract: Mapping[str, Any]) -> None:
    """Legacy accuracy / batch_invariance must mirror the four-judgment SSOT."""

    if "batch_invariance" not in contract:
        raise ContractSchemaError("compat key batch_invariance is required")
    bi = contract["batch_invariance"]
    if float(bi.get("atol", 1.0)) != 0.0 or float(bi.get("rtol", 1.0)) != 0.0:
        raise ContractSchemaError("batch_invariance must remain bitwise 0/0")

    if "accuracy" not in contract:
        raise ContractSchemaError("compat key accuracy is required")
    accuracy = contract["accuracy"]["default"]
    fwd = contract["judgments"]["forward_accuracy"]["by_op_class"]
    for op_class in OP_CLASSES:
        if op_class not in accuracy:
            raise ContractSchemaError(f"compat accuracy missing op_class {op_class!r}")
        for dtype_name in MANDATORY_DTYPES + OPTIONAL_DTYPES:
            if dtype_name not in accuracy[op_class]:
                raise ContractSchemaError(f"compat accuracy missing {op_class}/{dtype_name}")
            cell = fwd[op_class][dtype_name]
            if cell.get("status") not in {"applicable", "optional"}:
                continue
            acc = accuracy[op_class][dtype_name]
            if float(acc["atol"]) != float(cell["atol"]) or float(acc["rtol"]) != float(
                cell["rtol"]
            ):
                raise ContractSchemaError(
                    f"compat accuracy mismatch vs forward_accuracy for " f"{op_class}/{dtype_name}"
                )


def _lookup_cell(
    judgment_root: Mapping[str, Any],
    *,
    op_class: str,
    dtype_name: str,
    arch_key: str | None,
) -> Mapping[str, Any] | None:
    base = judgment_root.get("by_op_class", {}).get(op_class, {}).get(dtype_name)
    if arch_key is not None:
        arch_cell = (
            judgment_root.get("arch_overrides", {})
            .get(arch_key, {})
            .get(op_class, {})
            .get(dtype_name)
        )
        if arch_cell is not None:
            if base is None:
                return arch_cell
            return {**base, **arch_cell}
    return base


def normalize_dtype_name(dtype: str | Any) -> str:
    """Return the contract dtype name for a string alias or framework dtype."""
    if isinstance(dtype, str):
        name = dtype
        # Accept torch-style aliases.
        aliases = {
            "torch.float32": "float32",
            "torch.bfloat16": "bfloat16",
            "torch.float16": "float16",
            "torch.float8": "float8",
            "torch.float8_e4m3fn": "float8",
            "torch.float8_e5m2": "float8",
            "torch.float8_e4m3fnuz": "float8",
            "torch.float8_e5m2fnuz": "float8",
            "float8_e4m3fn": "float8",
            "float8_e5m2": "float8",
            "float8_e4m3fnuz": "float8",
            "float8_e5m2fnuz": "float8",
            "fp32": "float32",
            "bf16": "bfloat16",
            "fp16": "float16",
            "fp8": "float8",
        }
        name = aliases.get(name, name)
        if name not in ALL_DTYPES:
            raise ContractResolveError(f"unsupported dtype name {dtype!r}")
        return name

    # torch.dtype without importing torch at module import time for non-torch tests.
    module = getattr(type(dtype), "__module__", "")
    qual = getattr(dtype, "name", None) or str(dtype)
    if module.startswith("torch") or "torch" in str(type(dtype)):
        mapping = {
            "torch.float32": "float32",
            "torch.bfloat16": "bfloat16",
            "torch.float16": "float16",
            "torch.float8": "float8",
            "torch.float8_e4m3fn": "float8",
            "torch.float8_e5m2": "float8",
            "torch.float8_e4m3fnuz": "float8",
            "torch.float8_e5m2fnuz": "float8",
            "float32": "float32",
            "bfloat16": "bfloat16",
            "float16": "float16",
            "float8": "float8",
            "float8_e4m3fn": "float8",
            "float8_e5m2": "float8",
            "float8_e4m3fnuz": "float8",
            "float8_e5m2fnuz": "float8",
        }
        # torch.dtype str is like "torch.float32"
        as_str = str(dtype)
        if as_str in mapping:
            return mapping[as_str]
        if qual in mapping:
            return mapping[qual]
        try:
            import torch

            if dtype is torch.float32:
                return "float32"
            if dtype is torch.bfloat16:
                return "bfloat16"
            if dtype is torch.float16:
                return "float16"
            for attr in (
                "float8_e4m3fn",
                "float8_e5m2",
                "float8_e4m3fnuz",
                "float8_e5m2fnuz",
            ):
                torch_dtype = getattr(torch, attr, None)
                if torch_dtype is not None and dtype is torch_dtype:
                    return "float8"
        except ImportError:  # pragma: no cover
            pass
    raise ContractResolveError(f"unsupported dtype: {dtype!r}")


def resolve_logprob_threshold(dtype: Any) -> float:
    dtype_name = normalize_dtype_name(dtype)
    try:
        raw_threshold = load_contract()["accuracy"]["default"]["logprob"][dtype_name]["atol"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"WS1 has no logprob threshold for dtype {dtype_name!r}") from exc
    if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, (int, float)):
        raise ValueError(f"invalid WS1 logprob threshold for dtype {dtype_name!r}")
    threshold = float(raw_threshold)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"invalid WS1 logprob threshold for dtype {dtype_name!r}")
    return threshold


def tolerance_contract_fingerprint() -> str:
    canonical = json.dumps(
        load_contract(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# Private compatibility alias for callers outside this package that have not
# migrated to the public normalizer yet.
_dtype_name = normalize_dtype_name


__all__ = [
    "ALL_DTYPES",
    "CHAIN_AGGREGATE_METRICS",
    "JUDGMENTS",
    "OP_CLASSES",
    "AggregateMetricVerdict",
    "BackendProvenance",
    "ComparisonRoles",
    "ContractError",
    "ContractResolveError",
    "ContractSchemaError",
    "DtypePolicy",
    "LogprobAggregateVerdict",
    "LogprobAggregates",
    "ToleranceSpec",
    "ToleranceSupport",
    "assert_comparison_roles",
    "compute_logprob_aggregates",
    "default_clip_interval",
    "judge_logprob_aggregates",
    "load_contract",
    "normalize_dtype_name",
    "resolve_chain_aggregate_thresholds",
    "resolve_comparison_roles",
    "resolve_dtype_policy",
    "resolve_logprob_threshold",
    "tolerance_contract_fingerprint",
    "resolve_tolerance",
    "resolve_tolerance_support",
    "validate_backend_provenance",
    "validate_contract_schema",
]
