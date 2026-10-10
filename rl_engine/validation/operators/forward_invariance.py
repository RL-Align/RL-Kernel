# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 C3 (#269): Forward config-invariance and backend provenance harness.

Provides a shared forward accuracy/invariance API so downstream gates (C8, C10)
do not invent private thresholds, canonicalize wrong tokens, or compare outputs
from silently-fallback backends.

All thresholds come from the C1 tolerance contract. Logical identity and config
transforms come from the C2 canonical workload.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rl_engine.config.workload import (
    LogicalBatch,
    PaddedBatch,
    PhysicalLayout,
    WS1Manifest,
    apply_chunking,
    apply_packing,
    apply_padding,
    batch_permutation_from_manifest,
    build_logical_batch,
    chunk_plan_from_manifest,
    load_manifest,
    permute_batch,
    restore_logical_order,
    restore_logical_order_from_padded,
)
from rl_engine.contracts.numerical import (
    BackendProvenance,
    ContractResolveError,
    LogprobAggregateVerdict,
    compute_logprob_aggregates,
    default_clip_interval,
    judge_logprob_aggregates,
    load_contract,
    normalize_dtype_name,
    resolve_comparison_roles,
    resolve_tolerance,
    validate_backend_provenance,
)

_normalize_dtype_name = normalize_dtype_name


@dataclass(frozen=True)
class ConfigSpec:
    """One workload configuration (batch/chunk/padding/packing variant)."""

    config_id: str
    transform_kind: str
    logical_batch: LogicalBatch
    physical_layout: PhysicalLayout | PaddedBatch
    is_canonical: bool = False


@dataclass(frozen=True)
class RuntimeObservation:
    """Runtime facts returned alongside one candidate output."""

    output: Any
    actual_backend: str
    kernel_id: str
    output_dtype: str
    device: str


@dataclass(frozen=True)
class TensorComparisonDetail:
    """Per-tensor comparison result with full diagnostics."""

    tensor_name: str
    config_pair: tuple[str, str]
    shape: tuple[int, ...]
    dtype: str
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    atol: float
    rtol: float
    passed: bool
    judgment: str
    comparison_lhs_role: str
    comparison_rhs_role: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AccuracyReport:
    """Forward accuracy: bf16_candidate vs fp32_reference."""

    config_id: str
    op_class: str
    dtype: str
    backend_profile: str
    details: tuple[TensorComparisonDetail, ...]
    passed: bool
    backend_provenance: BackendProvenance | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.backend_provenance is not None:
            data["backend_provenance"] = self.backend_provenance.to_dict()
        return data


@dataclass(frozen=True)
class InvarianceReport:
    """Forward invariance: transformed vs canonical (bitwise atol=0 rtol=0)."""

    canonical_config_id: str
    transformed_config_id: str
    transform_kind: str
    op_class: str
    dtype: str
    backend_profile: str
    details: tuple[TensorComparisonDetail, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LogprobSmokeResult:
    """Selected-logprob aggregate smoke on fixed workload."""

    config_id: str
    backend_profile: str
    verdict: LogprobAggregateVerdict
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "backend_profile": self.backend_profile,
            "verdict": self.verdict.to_dict(),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ForwardInvarianceReport:
    """Suite-level report combining accuracy, invariance, and logprob smoke."""

    op_name: str
    backend_profile: str
    accuracy_reports: tuple[AccuracyReport, ...]
    invariance_reports: tuple[InvarianceReport, ...]
    logprob_smoke: LogprobSmokeResult | None
    backend_provenance: BackendProvenance | None
    candidate_id: str
    device: str
    compute_capability: str | None
    seed: int
    fallback_reason: str | None
    passed: bool
    provenance_valid: bool
    metadata_valid: bool
    observed_kernel_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "backend_profile": self.backend_profile,
            "accuracy_reports": [r.to_dict() for r in self.accuracy_reports],
            "invariance_reports": [r.to_dict() for r in self.invariance_reports],
            "logprob_smoke": (self.logprob_smoke.to_dict() if self.logprob_smoke else None),
            "backend_provenance": (
                self.backend_provenance.to_dict() if self.backend_provenance else None
            ),
            "candidate_id": self.candidate_id,
            "device": self.device,
            "compute_capability": self.compute_capability,
            "seed": self.seed,
            "fallback_reason": self.fallback_reason,
            "passed": self.passed,
            "provenance_valid": self.provenance_valid,
            "metadata_valid": self.metadata_valid,
            "observed_kernel_id": self.observed_kernel_id,
        }


def build_config_matrix(
    manifest: WS1Manifest | None = None,
) -> list[ConfigSpec]:
    """Build the C2 primary 2x2 matrix + permutation + padding + packing configs."""

    m = manifest if manifest is not None else load_manifest()
    chunk_plan = chunk_plan_from_manifest(m)
    batch_bn = build_logical_batch(m)
    configs: list[ConfigSpec] = []

    packed_bn = apply_packing(batch_bn)
    configs.append(
        ConfigSpec(
            config_id="BN/full",
            transform_kind="canonical",
            logical_batch=batch_bn,
            physical_layout=packed_bn,
            is_canonical=True,
        )
    )

    chunked_bn = apply_chunking(batch_bn, chunk_size=chunk_plan.chunk_size)
    configs.append(
        ConfigSpec(
            config_id="BN/chunked",
            transform_kind="chunk",
            logical_batch=batch_bn,
            physical_layout=chunked_bn,
        )
    )

    for sample in batch_bn.samples:
        single_batch = LogicalBatch(
            workload_id=batch_bn.workload_id,
            seed=batch_bn.seed,
            samples=(sample,),
            cell_id="B1-singleton_aggregate/full",
        )
        packed_single = apply_packing(single_batch)
        configs.append(
            ConfigSpec(
                config_id=f"B1-singleton_aggregate/full/{sample.sample_id}",
                transform_kind="batch_size",
                logical_batch=single_batch,
                physical_layout=packed_single,
            )
        )

    for sample in batch_bn.samples:
        single_batch = LogicalBatch(
            workload_id=batch_bn.workload_id,
            seed=batch_bn.seed,
            samples=(sample,),
            cell_id="B1-singleton_aggregate/chunked",
        )
        chunked_single = apply_chunking(single_batch, chunk_size=chunk_plan.chunk_size)
        configs.append(
            ConfigSpec(
                config_id=f"B1-singleton_aggregate/chunked/{sample.sample_id}",
                transform_kind="chunk",
                logical_batch=single_batch,
                physical_layout=chunked_single,
            )
        )

    perm = batch_permutation_from_manifest(m)
    permuted = permute_batch(batch_bn, perm)
    packed_perm = apply_packing(permuted)
    configs.append(
        ConfigSpec(
            config_id="BN/permuted",
            transform_kind="permutation",
            logical_batch=permuted,
            physical_layout=packed_perm,
        )
    )

    padded_right = apply_padding(batch_bn, pad_side="right", manifest=m)
    configs.append(
        ConfigSpec(
            config_id="BN/padded_right",
            transform_kind="padding",
            logical_batch=batch_bn,
            physical_layout=padded_right,
            is_canonical=False,
        )
    )

    padded_left = apply_padding(batch_bn, pad_side="left", manifest=m)
    configs.append(
        ConfigSpec(
            config_id="BN/padded_left",
            transform_kind="padding",
            logical_batch=batch_bn,
            physical_layout=padded_left,
            is_canonical=False,
        )
    )

    return configs


def _compare_logical_tensors(
    canonical: torch.Tensor,
    transformed: torch.Tensor,
    *,
    judgment: str,
    contract: Mapping[str, Any],
    op_class: str,
    dtype: str | torch.dtype,
    backend_profile: str | None = None,
    tensor_name: str = "output",
    config_pair: tuple[str, str] = ("canonical", "transformed"),
) -> TensorComparisonDetail:
    """Compare two tensors aligned to the same logical token order."""

    spec = resolve_tolerance(
        contract,
        judgment=judgment,
        op_class=op_class,
        dtype=dtype,
        backend_profile=backend_profile,
    )
    atol, rtol = spec.atol, spec.rtol
    roles = resolve_comparison_roles(contract, judgment)

    canonical_fp32 = canonical.float()
    transformed_fp32 = transformed.float()

    if canonical_fp32.shape != transformed_fp32.shape:
        return TensorComparisonDetail(
            tensor_name=tensor_name,
            config_pair=config_pair,
            shape=tuple(transformed_fp32.shape),
            dtype=_normalize_dtype_name(transformed.dtype),
            max_abs_error=float("inf"),
            mean_abs_error=float("inf"),
            max_rel_error=float("inf"),
            atol=atol,
            rtol=rtol,
            passed=False,
            judgment=judgment,
            comparison_lhs_role=roles.comparison_lhs_role,
            comparison_rhs_role=roles.comparison_rhs_role,
        )

    abs_error = (canonical_fp32 - transformed_fp32).abs()
    if abs_error.numel() == 0:
        max_abs = 0.0
        mean_abs = 0.0
        max_rel = 0.0
    else:
        max_abs = float(abs_error.max().item())
        mean_abs = float(abs_error.mean().item())
        rel_error = abs_error / canonical_fp32.abs().clamp_min(1e-12)
        max_rel = float(rel_error.max().item())

    passed = bool(torch.allclose(transformed_fp32, canonical_fp32, atol=atol, rtol=rtol))

    return TensorComparisonDetail(
        tensor_name=tensor_name,
        config_pair=config_pair,
        shape=tuple(canonical_fp32.shape),
        dtype=_normalize_dtype_name(canonical.dtype),
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        max_rel_error=max_rel,
        atol=atol,
        rtol=rtol,
        passed=passed,
        judgment=judgment,
        comparison_lhs_role=roles.comparison_lhs_role,
        comparison_rhs_role=roles.comparison_rhs_role,
    )


def _validate_provenance(
    contract: Mapping[str, Any],
    provenance: BackendProvenance | None,
    backend_profile: str,
) -> bool:
    """Validate backend provenance; return False if silent/cross-profile fallback."""

    if provenance is None:
        return False
    try:
        validate_backend_provenance(contract, provenance)
    except ContractResolveError:
        return False
    if provenance.backend_profile != backend_profile:
        return False
    return True


def _collect_logical_outputs(
    op: Callable[..., Any] | Any,
    config: ConfigSpec,
    *,
    op_kwargs: Mapping[str, Any] | None = None,
) -> tuple[dict[tuple[str, int], torch.Tensor], RuntimeObservation | None]:
    """Run op on a config and restore outputs to logical (sample_id, position) order."""

    kwargs = dict(op_kwargs) if op_kwargs else {}
    if hasattr(op, "forward") and callable(op.forward):
        raw_output = op.forward(config=config, **kwargs)
    else:
        raw_output = op(config=config, **kwargs)

    observation = raw_output if isinstance(raw_output, RuntimeObservation) else None
    if observation is not None:
        raw_output = observation.output
    if isinstance(raw_output, dict):
        return raw_output, observation

    if isinstance(raw_output, torch.Tensor):
        if isinstance(config.physical_layout, PaddedBatch):
            if raw_output.shape != (
                len(config.physical_layout.restore_map),
                config.physical_layout.padded_len,
            ):
                raise ValueError(
                    f"padded output shape {tuple(raw_output.shape)} does not match "
                    f"({len(config.physical_layout.restore_map)}, "
                    f"{config.physical_layout.padded_len})"
                )
            return (
                restore_logical_order_from_padded(config.physical_layout, list(raw_output)),
                observation,
            )
        flat = raw_output.reshape(-1)
        return restore_logical_order(config.physical_layout, list(flat)), observation

    raise TypeError(f"op must return dict or Tensor, got {type(raw_output)!r}")


def _align_and_compare_invariance(
    canonical_map: dict[tuple[str, int], Any],
    transformed_map: dict[tuple[str, int], Any],
    *,
    contract: Mapping[str, Any],
    op_class: str,
    dtype: str | torch.dtype,
    backend_profile: str | None,
    canonical_id: str,
    transformed_id: str,
    tensor_name: str = "output",
    expected_keys: set[tuple[str, int]],
) -> TensorComparisonDetail:
    """Align two logical output maps and compare for bitwise invariance."""

    canonical_keys = set(canonical_map)
    transformed_keys = set(transformed_map)
    if (
        not expected_keys
        or not expected_keys.issubset(canonical_keys)
        or not expected_keys.issubset(transformed_keys)
    ):
        return TensorComparisonDetail(
            tensor_name=tensor_name,
            config_pair=(canonical_id, transformed_id),
            shape=(0,),
            dtype=(
                _normalize_dtype_name(dtype)
                if isinstance(dtype, str)
                else _normalize_dtype_name(dtype)
            ),
            max_abs_error=float("inf"),
            mean_abs_error=float("inf"),
            max_rel_error=float("inf"),
            atol=0.0,
            rtol=0.0,
            passed=False,
            judgment="forward_invariance",
            comparison_lhs_role="transformed_config",
            comparison_rhs_role="canonical_config",
        )

    shared_keys = sorted(expected_keys)
    canonical_vals = torch.stack([torch.as_tensor(canonical_map[k]) for k in shared_keys])
    transformed_vals = torch.stack([torch.as_tensor(transformed_map[k]) for k in shared_keys])

    return _compare_logical_tensors(
        canonical_vals,
        transformed_vals,
        judgment="forward_invariance",
        contract=contract,
        op_class=op_class,
        dtype=dtype,
        backend_profile=backend_profile,
        tensor_name=tensor_name,
        config_pair=(canonical_id, transformed_id),
    )


def assert_forward_batch_invariant(
    op: Callable[..., Any] | Any,
    configs: Sequence[ConfigSpec] | None = None,
    contract: Mapping[str, Any] | None = None,
    *,
    manifest: WS1Manifest | None = None,
    backend_profile: str,
    provenance: BackendProvenance | None = None,
    gold_fn: Callable[..., Any] | None = None,
    op_class: str = "logprob",
    dtype: torch.dtype = torch.bfloat16,
    op_name: str = "operator",
    op_kwargs: Mapping[str, Any] | None = None,
    include_logprob_smoke: bool = True,
    active_only: bool = True,
    candidate_id: str = "unspecified",
    device: str = "unspecified",
    compute_capability: str | None = None,
    fallback_reason: str | None = None,
    observed_actual_backend: str | None = None,
    observed_kernel_id: str | None = None,
    observed_output_dtype: str | None = None,
) -> ForwardInvarianceReport:
    """Run forward config-invariance and accuracy checks.

    This is the sole C3 API. C10 must reuse this harness/report schema.

    Args:
        op: Operator callable. Must accept (config=ConfigSpec, **op_kwargs) and
            return either a dict[(sample_id, position) -> Tensor] or a flat Tensor.
        configs: Config matrix; built from C2 manifest if None.
        contract: C1 tolerance contract; loaded from default path if None.
        manifest: C2 workload manifest; loaded from default path if None.
        backend_profile: Required profile id (cuda_bf16 or triton_cuda_bf16).
        provenance: Runtime-observed backend provenance. Missing provenance fails closed.
        gold_fn: FP32 reference callable for accuracy checks.
        op_class: Operator class for tolerance resolution.
        dtype: Execution dtype.
        op_name: Name for reporting.
        op_kwargs: Extra kwargs passed to op.
        include_logprob_smoke: Whether to run logprob aggregate smoke.
        active_only: Only compare active (non-prompt) tokens for invariance.

    Returns:
        ForwardInvarianceReport with accuracy, invariance, and logprob sub-reports.
    """

    loaded_contract = dict(contract or load_contract())
    m = manifest if manifest is not None else load_manifest()
    config_list = list(configs) if configs is not None else build_config_matrix(m)
    if not config_list:
        raise ValueError("configs must contain at least one configuration")
    if gold_fn is None:
        raise ValueError("gold_fn is required for forward accuracy")
    if include_logprob_smoke and op_class != "logprob":
        raise ValueError("selected-logprob smoke requires op_class='logprob'")

    provenance_valid = _validate_provenance(loaded_contract, provenance, backend_profile)
    if not provenance_valid and fallback_reason is None:
        fallback_reason = "missing or contract-invalid backend provenance"
    metadata_valid = (
        candidate_id != "unspecified"
        and device != "unspecified"
        and compute_capability is not None
        and fallback_reason is None
    )
    metadata_valid = metadata_valid and all(
        value is not None
        for value in (observed_actual_backend, observed_kernel_id, observed_output_dtype)
    )
    if provenance is not None and observed_actual_backend is not None:
        metadata_valid = metadata_valid and observed_actual_backend == provenance.actual_backend

    canonical_config = next((c for c in config_list if c.is_canonical), config_list[0])
    canonical_outputs, canonical_observation = _collect_logical_outputs(
        op, canonical_config, op_kwargs=op_kwargs
    )

    def expected_keys(config: ConfigSpec) -> set[tuple[str, int]]:
        return set(config.logical_batch.logical_keys(active_only=active_only))

    def validate_keys(
        outputs: Mapping[tuple[str, int], Any], config: ConfigSpec, label: str
    ) -> None:
        required = expected_keys(config)
        allowed = set(config.logical_batch.logical_keys(active_only=False))
        actual = set(outputs)
        if not required.issubset(actual) or not actual.issubset(allowed):
            raise ValueError(
                f"{label} output keys for {config.config_id!r} do not match the "
                "C2 logical identity"
            )

    canonical_keys = expected_keys(canonical_config)
    validate_keys(canonical_outputs, canonical_config, "canonical")
    if canonical_observation is not None:
        observed_device = str(canonical_observation.device)
        report_device = str(device)
        metadata_valid = metadata_valid and (
            provenance is not None
            and canonical_observation.actual_backend == provenance.actual_backend
            and canonical_observation.actual_backend == observed_actual_backend
            and canonical_observation.kernel_id == observed_kernel_id
            and _normalize_dtype_name(canonical_observation.output_dtype)
            == _normalize_dtype_name(observed_output_dtype)
            and (
                report_device == observed_device or report_device.startswith(observed_device + ":")
            )
            and _normalize_dtype_name(canonical_observation.output_dtype)
            == _normalize_dtype_name(next(iter(canonical_outputs.values())).dtype)
        )

    invariance_reports: list[InvarianceReport] = []
    for config in config_list:
        if config.is_canonical:
            continue
        transformed_outputs, observation = _collect_logical_outputs(op, config, op_kwargs=op_kwargs)
        if canonical_observation is not None and observation is not None:
            metadata_valid = metadata_valid and (
                observation.actual_backend == canonical_observation.actual_backend
                and observation.kernel_id == canonical_observation.kernel_id
                and observation.output_dtype == canonical_observation.output_dtype
            )
        validate_keys(transformed_outputs, config, "transformed")
        detail = _align_and_compare_invariance(
            canonical_outputs,
            transformed_outputs,
            contract=loaded_contract,
            op_class=op_class,
            dtype=dtype,
            backend_profile=backend_profile,
            canonical_id=canonical_config.config_id,
            transformed_id=config.config_id,
            expected_keys=expected_keys(config),
        )
        invariance_reports.append(
            InvarianceReport(
                canonical_config_id=canonical_config.config_id,
                transformed_config_id=config.config_id,
                transform_kind=config.transform_kind,
                op_class=op_class,
                dtype=_normalize_dtype_name(dtype),
                backend_profile=backend_profile,
                details=(detail,),
                passed=detail.passed,
            )
        )

    accuracy_reports: list[AccuracyReport] = []
    for config in config_list:
        candidate_outputs = (
            canonical_outputs
            if config.is_canonical
            else _collect_logical_outputs(op, config, op_kwargs=op_kwargs)[0]
        )
        gold_outputs = _collect_logical_outputs(gold_fn, config, op_kwargs=op_kwargs)[0]
        keys = expected_keys(config)
        validate_keys(candidate_outputs, config, "candidate accuracy")
        validate_keys(gold_outputs, config, "reference accuracy")
        ordered_keys = sorted(keys)
        candidate_vals = torch.stack([torch.as_tensor(candidate_outputs[k]) for k in ordered_keys])
        gold_vals = torch.stack([torch.as_tensor(gold_outputs[k]) for k in ordered_keys])
        acc_detail = _compare_logical_tensors(
            gold_vals,
            candidate_vals,
            judgment="forward_accuracy",
            contract=loaded_contract,
            op_class=op_class,
            dtype=dtype,
            backend_profile=backend_profile,
            tensor_name="selected_logprob" if op_class == "logprob" else "output",
            config_pair=(config.config_id, "fp32_reference"),
        )
        accuracy_reports.append(
            AccuracyReport(
                config_id=config.config_id,
                op_class=op_class,
                dtype=_normalize_dtype_name(dtype),
                backend_profile=backend_profile,
                details=(acc_detail,),
                passed=acc_detail.passed,
                backend_provenance=provenance,
            )
        )

    logprob_smoke: LogprobSmokeResult | None = None
    if include_logprob_smoke:
        logprob_smoke = _run_logprob_smoke(
            canonical_outputs,
            gold_fn,
            canonical_config,
            loaded_contract,
            m,
            backend_profile=backend_profile,
            op_kwargs=op_kwargs,
            active_keys=canonical_keys,
        )

    all_invariance_passed = all(r.passed for r in invariance_reports)
    all_accuracy_passed = all(r.passed for r in accuracy_reports)
    smoke_passed = logprob_smoke.passed if logprob_smoke is not None else True

    overall_passed = (
        all_invariance_passed
        and all_accuracy_passed
        and smoke_passed
        and provenance_valid
        and metadata_valid
    )

    return ForwardInvarianceReport(
        op_name=op_name,
        backend_profile=backend_profile,
        accuracy_reports=tuple(accuracy_reports),
        invariance_reports=tuple(invariance_reports),
        logprob_smoke=logprob_smoke,
        backend_provenance=provenance,
        candidate_id=candidate_id,
        device=device,
        compute_capability=compute_capability,
        seed=m.seed,
        fallback_reason=fallback_reason,
        passed=overall_passed,
        provenance_valid=provenance_valid,
        metadata_valid=metadata_valid,
        observed_kernel_id=observed_kernel_id,
    )


def _run_logprob_smoke(
    candidate_outputs: dict[tuple[str, int], Any],
    gold_fn: Callable[..., Any] | Any,
    config: ConfigSpec,
    contract: Mapping[str, Any],
    manifest: WS1Manifest,
    *,
    backend_profile: str,
    op_kwargs: Mapping[str, Any] | None = None,
    active_keys: set[tuple[str, int]] | None = None,
) -> LogprobSmokeResult:
    """Run selected-logprob aggregate smoke check."""

    gold_outputs = _collect_logical_outputs(gold_fn, config, op_kwargs=op_kwargs)[0]
    if active_keys is not None:
        shared = sorted(k for k in candidate_outputs if k in gold_outputs and k in active_keys)
    else:
        shared = sorted(k for k in candidate_outputs if k in gold_outputs)

    if not shared:
        raise ContractResolveError("no shared active tokens for logprob smoke")

    lhs_logp = torch.stack([torch.as_tensor(candidate_outputs[k]).float() for k in shared])
    rhs_logp = torch.stack([torch.as_tensor(gold_outputs[k]).float() for k in shared])
    active_mask = torch.ones(len(shared), dtype=torch.bool)

    clip_interval = default_clip_interval(contract)
    roles = resolve_comparison_roles(contract, "forward_accuracy")

    aggregates = compute_logprob_aggregates(
        lhs_logp,
        rhs_logp,
        active_mask,
        contract=contract,
        report_kind="forward_accuracy",
        clip_interval=clip_interval,
        comparison_lhs_role=roles.comparison_lhs_role,
        comparison_rhs_role=roles.comparison_rhs_role,
    )
    verdict = judge_logprob_aggregates(
        aggregates,
        contract,
        execution_dtype="bfloat16",
    )
    return LogprobSmokeResult(
        config_id=config.config_id,
        backend_profile=backend_profile,
        verdict=verdict,
        passed=verdict.passed,
    )


__all__ = [
    "AccuracyReport",
    "ConfigSpec",
    "ForwardInvarianceReport",
    "InvarianceReport",
    "LogprobSmokeResult",
    "TensorComparisonDetail",
    "assert_forward_batch_invariant",
    "build_config_matrix",
]
