# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 C4 (#270): Gradient config-invariance harness.

Training-style VJPs share one comparison semantic across the C2 Batch/Chunk
matrix: same logical sample/token multiset, same upstream grad, same loss
reduction, and the same global active-token denominator.

Accuracy (candidate vs FP32 VJP) and invariance (cross-config) are separate
C1 judgments. This module does not implement the full-model C10 gate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from rl_engine.config.workload import (
    PaddedBatch,
    PhysicalLayout,
    WS1Manifest,
    load_manifest,
    restore_logical_order,
    restore_logical_order_from_padded,
    singleton_aggregate_plan,
)
from rl_engine.contracts.numerical import BackendProvenance, load_contract, normalize_dtype_name
from rl_engine.validation.operators.forward_invariance import (
    AccuracyReport,
    ConfigSpec,
    InvarianceReport,
    TensorComparisonDetail,
    _compare_logical_tensors,
    _validate_provenance,
    build_config_matrix,
)

GradKind = Literal["token", "parameter"]


class MissingBackwardError(RuntimeError):
    """A required differentiable node produced an output with no backward.

    #270 treats a missing backward on a required node as red, so this must
    surface as a categorised verdict rather than an autograd stack trace.
    """

    def __init__(self, op_name: str, detail: str = "") -> None:
        message = (
            f"required differentiable node {op_name!r} produced a non-differentiable "
            "output (no grad_fn); a missing backward is red, not N/A or fallback"
        )
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.op_name = op_name


@dataclass(frozen=True)
class GradientTensorSpec:
    """One named gradient produced by an adapter."""

    name: str
    kind: GradKind
    source_input: str


@dataclass(frozen=True)
class GradientObservation:
    """Runtime facts returned alongside named gradients."""

    grads: Mapping[str, Any]
    actual_backend: str
    kernel_id: str
    output_dtype: str
    device: str


@dataclass(frozen=True)
class GradientInvarianceReport:
    """Suite-level gradient accuracy + invariance report."""

    op_name: str
    backend_profile: str
    accuracy_reports: tuple[AccuracyReport, ...]
    invariance_reports: tuple[InvarianceReport, ...]
    singleton_aggregate_reports: tuple[InvarianceReport, ...]
    backend_provenance: BackendProvenance | None
    candidate_id: str
    device: str
    compute_capability: str | None
    seed: int
    fallback_reason: str | None
    passed: bool
    provenance_valid: bool
    metadata_valid: bool
    loss_reduction: str
    active_token_denominator: int
    grad_tensor_names: tuple[str, ...]
    first_failing_op: str | None
    first_failing_tensor: str | None
    first_failing_config_pair: tuple[str, str] | None
    observed_kernel_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "backend_profile": self.backend_profile,
            "accuracy_reports": [r.to_dict() for r in self.accuracy_reports],
            "invariance_reports": [r.to_dict() for r in self.invariance_reports],
            "singleton_aggregate_reports": [r.to_dict() for r in self.singleton_aggregate_reports],
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
            "loss_reduction": self.loss_reduction,
            "active_token_denominator": self.active_token_denominator,
            "grad_tensor_names": list(self.grad_tensor_names),
            "first_failing_op": self.first_failing_op,
            "first_failing_tensor": self.first_failing_tensor,
            "first_failing_config_pair": self.first_failing_config_pair,
            "observed_kernel_id": self.observed_kernel_id,
        }


def _tensor_specs(grad_tensors: Sequence[GradientTensorSpec]) -> tuple[GradientTensorSpec, ...]:
    specs = tuple(grad_tensors)
    if not specs:
        raise ValueError("grad_tensors must declare at least one gradient")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate gradient names: {names}")
    for spec in specs:
        if spec.kind not in ("token", "parameter"):
            raise ValueError(f"unsupported gradient kind {spec.kind!r} for {spec.name}")
    return specs


def _is_singleton_config(config: ConfigSpec) -> bool:
    return config.config_id.startswith("B1-singleton_aggregate/")


def _singleton_group(config_id: str) -> str | None:
    if config_id.startswith("B1-singleton_aggregate/full/"):
        return "full"
    if config_id.startswith("B1-singleton_aggregate/chunked/"):
        return "chunked"
    return None


def _singleton_sample_id(config_id: str) -> str:
    return config_id.rsplit("/", 1)[-1]


def _token_map_from_physical(value: torch.Tensor, config: ConfigSpec) -> dict[tuple[str, int], Any]:
    layout = config.physical_layout
    if isinstance(layout, PaddedBatch):
        expected = (len(layout.restore_map), layout.padded_len)
        if value.shape[:2] != expected:
            raise ValueError(
                f"padded gradient shape {tuple(value.shape)} does not start with {expected}"
            )
        return restore_logical_order_from_padded(layout, list(value))
    if not isinstance(layout, PhysicalLayout):
        raise TypeError(f"unsupported physical layout {type(layout)!r}")
    n_tokens = len(layout.restore_map)
    if value.shape[0] != n_tokens:
        raise ValueError(
            f"packed/chunked gradient leading dim {value.shape[0]} != {n_tokens} restore rows"
        )
    return restore_logical_order(layout, list(value))


def _coerce_token_grad(value: Any, config: ConfigSpec) -> dict[tuple[str, int], Any]:
    if isinstance(value, Mapping):
        return {(str(sample), int(pos)): tensor for (sample, pos), tensor in value.items()}
    if isinstance(value, torch.Tensor):
        return _token_map_from_physical(value, config)
    raise TypeError(f"token gradient must be a dict or Tensor, got {type(value)!r}")


def _collect_logical_grads(
    op: Callable[..., Any] | Any,
    config: ConfigSpec,
    *,
    specs: Sequence[GradientTensorSpec],
    op_kwargs: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], GradientObservation | None]:
    kwargs = dict(op_kwargs) if op_kwargs else {}
    if hasattr(op, "backward_grads") and callable(op.backward_grads):
        raw = op.backward_grads(config=config, **kwargs)
    elif hasattr(op, "forward") and callable(op.forward):
        raw = op.forward(config=config, **kwargs)
    else:
        raw = op(config=config, **kwargs)

    observation = raw if isinstance(raw, GradientObservation) else None
    grads = dict(observation.grads) if observation is not None else raw
    if not isinstance(grads, Mapping):
        raise TypeError(f"op must return a grad mapping, got {type(grads)!r}")

    missing = [spec.name for spec in specs if spec.name not in grads]
    if missing:
        raise ValueError(f"missing required gradients: {', '.join(missing)}")

    logical: dict[str, Any] = {}
    for spec in specs:
        value = grads[spec.name]
        if spec.kind == "parameter":
            logical[spec.name] = torch.as_tensor(value)
        else:
            logical[spec.name] = _coerce_token_grad(value, config)
    contributions = grads.get("__parameter_contributions__")
    if contributions is not None:
        logical["__parameter_contributions__"] = contributions
    return logical, observation


def _expected_keys(config: ConfigSpec, *, active_only: bool) -> set[tuple[str, int]]:
    return set(config.logical_batch.logical_keys(active_only=active_only))


def _validate_token_keys(
    grads: Mapping[str, Any],
    config: ConfigSpec,
    specs: Sequence[GradientTensorSpec],
    *,
    label: str,
    active_only: bool,
) -> None:
    required = _expected_keys(config, active_only=active_only)
    allowed = set(config.logical_batch.logical_keys(active_only=False))
    for spec in specs:
        if spec.kind != "token":
            continue
        actual = set(grads[spec.name])
        if not required.issubset(actual) or not actual.issubset(allowed):
            raise ValueError(
                f"{label} gradient {spec.name!r} keys for {config.config_id!r} "
                "do not match the C2 logical identity"
            )


def _stack_token_grad(
    grad_map: Mapping[tuple[str, int], Any], keys: Sequence[tuple[str, int]]
) -> torch.Tensor:
    return torch.stack([torch.as_tensor(grad_map[key]) for key in keys])


def _align_token_grad(
    canonical_map: Mapping[tuple[str, int], Any],
    transformed_map: Mapping[tuple[str, int], Any],
    *,
    spec: GradientTensorSpec,
    contract: Mapping[str, Any],
    op_class: str,
    dtype: str | torch.dtype,
    backend_profile: str | None,
    canonical_id: str,
    transformed_id: str,
    expected_keys: set[tuple[str, int]],
) -> TensorComparisonDetail:
    if (
        not expected_keys
        or not expected_keys.issubset(canonical_map)
        or not expected_keys.issubset(transformed_map)
    ):
        return TensorComparisonDetail(
            tensor_name=spec.name,
            config_pair=(canonical_id, transformed_id),
            shape=(0,),
            dtype=normalize_dtype_name(dtype),
            max_abs_error=float("inf"),
            mean_abs_error=float("inf"),
            max_rel_error=float("inf"),
            atol=0.0,
            rtol=0.0,
            passed=False,
            judgment="gradient_invariance",
            comparison_lhs_role="transformed_config",
            comparison_rhs_role="canonical_config",
        )
    ordered = sorted(expected_keys)
    return _compare_logical_tensors(
        _stack_token_grad(canonical_map, ordered),
        _stack_token_grad(transformed_map, ordered),
        judgment="gradient_invariance",
        contract=contract,
        op_class=op_class,
        dtype=dtype,
        backend_profile=backend_profile,
        tensor_name=spec.name,
        config_pair=(canonical_id, transformed_id),
    )


def _compare_parameter_grad(
    canonical: torch.Tensor,
    transformed: torch.Tensor,
    *,
    spec: GradientTensorSpec,
    contract: Mapping[str, Any],
    op_class: str,
    dtype: str | torch.dtype,
    backend_profile: str | None,
    canonical_id: str,
    transformed_id: str,
) -> TensorComparisonDetail:
    return _compare_logical_tensors(
        torch.as_tensor(canonical),
        torch.as_tensor(transformed),
        judgment="gradient_invariance",
        contract=contract,
        op_class=op_class,
        dtype=dtype,
        backend_profile=backend_profile,
        tensor_name=spec.name,
        config_pair=(canonical_id, transformed_id),
    )


def _invariance_report(
    *,
    canonical_id: str,
    transformed_id: str,
    transform_kind: str,
    op_class: str,
    dtype: str | torch.dtype,
    backend_profile: str,
    details: Sequence[TensorComparisonDetail],
) -> InvarianceReport:
    detail_tuple = tuple(details)
    return InvarianceReport(
        canonical_config_id=canonical_id,
        transformed_config_id=transformed_id,
        transform_kind=transform_kind,
        op_class=op_class,
        dtype=normalize_dtype_name(dtype),
        backend_profile=backend_profile,
        details=detail_tuple,
        passed=all(detail.passed for detail in detail_tuple),
    )


def _first_failure(
    op_name: str,
    reports: Sequence[AccuracyReport | InvarianceReport],
) -> tuple[str | None, str | None, tuple[str, str] | None]:
    for report in reports:
        details = report.details
        for detail in details:
            if not detail.passed:
                return op_name, detail.tensor_name, detail.config_pair
    return None, None, None


def _sum_parameter_grads(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("singleton aggregate requires at least one parameter gradient")
    # Parameter grads are aggregated in the C1 accumulation dtype (fp32) using
    # the C2 fixed sample order. Down-casting each B=1 result first would make
    # the singleton sum a different rounding than one B=N reduction.
    total = values[0].float().clone()
    for value in values[1:]:
        total = total + value.float()
    return total


def assert_gradient_batch_invariant(
    op: Callable[..., Any] | Any,
    configs: Sequence[ConfigSpec] | None = None,
    contract: Mapping[str, Any] | None = None,
    *,
    grad_tensors: Sequence[GradientTensorSpec],
    manifest: WS1Manifest | None = None,
    backend_profile: str,
    provenance: BackendProvenance | None = None,
    gold_fn: Callable[..., Any] | None = None,
    op_class: str,
    dtype: torch.dtype = torch.bfloat16,
    op_name: str = "operator",
    op_kwargs: Mapping[str, Any] | None = None,
    active_only: bool = True,
    candidate_id: str = "unspecified",
    device: str = "unspecified",
    compute_capability: str | None = None,
    fallback_reason: str | None = None,
    observed_actual_backend: str | None = None,
    observed_kernel_id: str | None = None,
    observed_output_dtype: str | None = None,
) -> GradientInvarianceReport:
    """Run gradient accuracy and config-invariance checks.

    This is the sole C4 API. C8/C10 must reuse this harness/report schema.
    """

    specs = _tensor_specs(grad_tensors)
    loaded_contract = dict(contract or load_contract())
    m = manifest if manifest is not None else load_manifest()
    config_list = list(configs) if configs is not None else build_config_matrix(m)
    if not config_list:
        raise ValueError("configs must contain at least one configuration")
    if gold_fn is None:
        raise ValueError("gold_fn is required for gradient accuracy")

    canonical_configs = [c for c in config_list if c.is_canonical]
    if not canonical_configs:
        raise ValueError("configs must contain exactly one canonical configuration")
    canonical_config = canonical_configs[0]
    canonical_batch = canonical_config.logical_batch
    plan = singleton_aggregate_plan(canonical_batch)
    if plan.denominator != "active_token_count_across_all_samples":
        raise ValueError(f"unsupported gradient denominator {plan.denominator!r}")
    active_token_denominator = canonical_batch.active_token_count()
    if active_token_denominator <= 0:
        raise ValueError("empty active-token set is a hard fail for gradient checks")
    loss_reduction = str(
        m.chain_semantics.get(
            "loss_reduction",
            "sum_over_active_tokens_then_optional_mean_by_active_count",
        )
    )

    merged_kwargs = dict(op_kwargs) if op_kwargs else {}
    merged_kwargs.setdefault("active_token_denominator", active_token_denominator)
    merged_kwargs.setdefault("loss_reduction", loss_reduction)
    merged_kwargs.setdefault("aggregation_order", plan.aggregation_order)

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

    collected: dict[str, dict[str, Any]] = {}
    observations: dict[str, GradientObservation | None] = {}
    for config in config_list:
        grads, observation = _collect_logical_grads(
            op, config, specs=specs, op_kwargs=merged_kwargs
        )
        _validate_token_keys(grads, config, specs, label="candidate", active_only=active_only)
        collected[config.config_id] = grads
        observations[config.config_id] = observation

    canonical_grads = collected[canonical_config.config_id]
    canonical_observation = observations[canonical_config.config_id]
    if canonical_observation is not None:
        observed_device = str(canonical_observation.device)
        report_device = str(device)
        metadata_valid = metadata_valid and (
            provenance is not None
            and canonical_observation.actual_backend == provenance.actual_backend
            and canonical_observation.actual_backend == observed_actual_backend
            and canonical_observation.kernel_id == observed_kernel_id
            and normalize_dtype_name(canonical_observation.output_dtype)
            == normalize_dtype_name(observed_output_dtype)
            and (
                report_device == observed_device or report_device.startswith(observed_device + ":")
            )
        )

    invariance_reports: list[InvarianceReport] = []
    for config in config_list:
        if config.is_canonical:
            continue
        observation = observations[config.config_id]
        if canonical_observation is not None and observation is not None:
            metadata_valid = metadata_valid and (
                observation.actual_backend == canonical_observation.actual_backend
                and observation.kernel_id == canonical_observation.kernel_id
                and observation.output_dtype == canonical_observation.output_dtype
            )
        details: list[TensorComparisonDetail] = []
        transformed = collected[config.config_id]
        for spec in specs:
            if spec.kind == "parameter" and _is_singleton_config(config):
                continue
            if spec.kind == "token":
                details.append(
                    _align_token_grad(
                        canonical_grads[spec.name],
                        transformed[spec.name],
                        spec=spec,
                        contract=loaded_contract,
                        op_class=op_class,
                        dtype=dtype,
                        backend_profile=backend_profile,
                        canonical_id=canonical_config.config_id,
                        transformed_id=config.config_id,
                        expected_keys=_expected_keys(config, active_only=active_only),
                    )
                )
            else:
                details.append(
                    _compare_parameter_grad(
                        canonical_grads[spec.name],
                        transformed[spec.name],
                        spec=spec,
                        contract=loaded_contract,
                        op_class=op_class,
                        dtype=dtype,
                        backend_profile=backend_profile,
                        canonical_id=canonical_config.config_id,
                        transformed_id=config.config_id,
                    )
                )
        if details:
            invariance_reports.append(
                _invariance_report(
                    canonical_id=canonical_config.config_id,
                    transformed_id=config.config_id,
                    transform_kind=config.transform_kind,
                    op_class=op_class,
                    dtype=dtype,
                    backend_profile=backend_profile,
                    details=details,
                )
            )

    singleton_aggregate_reports: list[InvarianceReport] = []
    parameter_specs = [spec for spec in specs if spec.kind == "parameter"]
    if parameter_specs:
        by_group: dict[str, dict[str, dict[str, Any]]] = {"full": {}, "chunked": {}}
        for config in config_list:
            group = _singleton_group(config.config_id)
            if group is None:
                continue
            by_group[group][_singleton_sample_id(config.config_id)] = collected[config.config_id]
        for group, sample_grads in by_group.items():
            if not sample_grads:
                continue
            missing = [
                sample_id for sample_id in plan.aggregation_order if sample_id not in sample_grads
            ]
            if missing:
                raise ValueError(
                    f"singleton aggregate {group} missing sample grads: {', '.join(missing)}"
                )
            details = []
            for spec in parameter_specs:
                contribution_maps = [
                    sample_grads[sample_id].get("__parameter_contributions__", {}).get(spec.name)
                    for sample_id in plan.aggregation_order
                ]
                if all(value is not None for value in contribution_maps):
                    merged = {
                        key: value
                        for contribution_map in contribution_maps
                        for key, value in contribution_map.items()
                    }
                    ordered_rows = [merged[key] for key in sorted(merged)]
                    aggregated = _sum_parameter_grads(ordered_rows)
                else:
                    ordered = [
                        sample_grads[sample_id][spec.name] for sample_id in plan.aggregation_order
                    ]
                    aggregated = _sum_parameter_grads(ordered)
                details.append(
                    _compare_parameter_grad(
                        canonical_grads[spec.name],
                        aggregated,
                        spec=spec,
                        contract=loaded_contract,
                        op_class=op_class,
                        dtype=dtype,
                        backend_profile=backend_profile,
                        canonical_id=canonical_config.config_id,
                        transformed_id=f"B1-singleton_aggregate/{group}",
                    )
                )
            singleton_aggregate_reports.append(
                _invariance_report(
                    canonical_id=canonical_config.config_id,
                    transformed_id=f"B1-singleton_aggregate/{group}",
                    transform_kind="batch_size",
                    op_class=op_class,
                    dtype=dtype,
                    backend_profile=backend_profile,
                    details=details,
                )
            )

    accuracy_reports: list[AccuracyReport] = []
    for config in config_list:
        candidate_grads = collected[config.config_id]
        gold_grads, _ = _collect_logical_grads(
            gold_fn, config, specs=specs, op_kwargs=merged_kwargs
        )
        _validate_token_keys(gold_grads, config, specs, label="reference", active_only=active_only)
        details = []
        keys = sorted(_expected_keys(config, active_only=active_only))
        for spec in specs:
            if spec.kind == "token":
                candidate_vals = _stack_token_grad(candidate_grads[spec.name], keys)
                gold_vals = _stack_token_grad(gold_grads[spec.name], keys)
            else:
                candidate_vals = torch.as_tensor(candidate_grads[spec.name])
                gold_vals = torch.as_tensor(gold_grads[spec.name])
            details.append(
                _compare_logical_tensors(
                    gold_vals,
                    candidate_vals,
                    judgment="gradient_accuracy",
                    contract=loaded_contract,
                    op_class=op_class,
                    dtype=dtype,
                    backend_profile=backend_profile,
                    tensor_name=spec.name,
                    config_pair=(config.config_id, "fp32_reference"),
                )
            )
        accuracy_reports.append(
            AccuracyReport(
                config_id=config.config_id,
                op_class=op_class,
                dtype=normalize_dtype_name(dtype),
                backend_profile=backend_profile,
                details=tuple(details),
                passed=all(detail.passed for detail in details),
                backend_provenance=provenance,
            )
        )

    first_op, first_tensor, first_pair = _first_failure(
        op_name,
        (*accuracy_reports, *invariance_reports, *singleton_aggregate_reports),
    )
    overall_passed = (
        all(report.passed for report in accuracy_reports)
        and all(report.passed for report in invariance_reports)
        and all(report.passed for report in singleton_aggregate_reports)
        and provenance_valid
        and metadata_valid
    )
    return GradientInvarianceReport(
        op_name=op_name,
        backend_profile=backend_profile,
        accuracy_reports=tuple(accuracy_reports),
        invariance_reports=tuple(invariance_reports),
        singleton_aggregate_reports=tuple(singleton_aggregate_reports),
        backend_provenance=provenance,
        candidate_id=candidate_id,
        device=device,
        compute_capability=compute_capability,
        seed=m.seed,
        fallback_reason=fallback_reason,
        passed=overall_passed,
        provenance_valid=provenance_valid,
        metadata_valid=metadata_valid,
        loss_reduction=loss_reduction,
        active_token_denominator=active_token_denominator,
        grad_tensor_names=tuple(spec.name for spec in specs),
        first_failing_op=first_op,
        first_failing_tensor=first_tensor,
        first_failing_config_pair=first_pair,
        observed_kernel_id=observed_kernel_id,
    )


__all__ = [
    "GradientInvarianceReport",
    "GradientObservation",
    "GradientTensorSpec",
    "MissingBackwardError",
    "assert_gradient_batch_invariant",
]
