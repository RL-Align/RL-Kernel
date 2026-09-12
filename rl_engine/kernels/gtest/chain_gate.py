# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 C10 / #276: full Qwen3-8B Dense model-level train–inference gate.

Batch/Chunk/padding/permutation invariance is bitwise after logical
unpadding. Packing and FP32-gold comparisons use C1 accuracy rows.
Logprob pass/fail uses only max_abs_dlogp / approx_kl0 / clipfrac0.
Gradients use independent C1 rows, reduced in logical-token order.
C9 assembly alone is not EXIT.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from rl_engine.alignment.qwen3_dense import (
    Qwen3DenseBIModel,
    Qwen3DenseSpec,
    Qwen3DenseWeights,
    load_profile_ops,
)
from rl_engine.kernels.gtest.accelerator import (
    ACCELERATOR_TYPES,
    compute_capability,
    device_name,
    device_type_for_profile,
    empty_cache,
    is_available,
    manual_seed_all,
)
from rl_engine.kernels.gtest.chain_gradients import GRADIENT_SCOPE, REQUIRED_GRAD_NAMES
from rl_engine.kernels.gtest.forward_invariance import (
    TensorComparisonDetail,
    _compare_logical_tensors,
)
from rl_engine.kernels.gtest.kv_consistency import make_profile_provenance
from rl_engine.kernels.gtest.tolerance import (
    BackendProvenance,
    LogprobAggregateVerdict,
    compute_logprob_aggregates,
    default_clip_interval,
    judge_logprob_aggregates,
    load_contract,
    resolve_comparison_roles,
    resolve_dtype_policy,
    resolve_tolerance,
)
from rl_engine.kernels.ops.backward_runtime import reset_backward_runtime, snapshot_backward_runtime
from rl_engine.kernels.ops.canonical_backward import active_session, canonical_backward_session
from rl_engine.testing.ws1_workload import (
    LogicalBatch,
    LogicalSample,
    WS1Manifest,
    apply_packing,
    apply_padding,
    batch_permutation_from_manifest,
    build_logical_batch,
    chunk_plan_from_manifest,
    fixture_hash,
    load_manifest,
    permute_batch,
)

PRIMARY_CELLS = (
    "B1-singleton_aggregate/full",
    "BN/full",
    "B1-singleton_aggregate/chunked",
    "BN/chunked",
)
LAYOUT_CELLS = (
    "BN/padded_right",
    "BN/padded_left",
    "BN/permuted",
    "BN/packed",
)
_TRAIN_INFER_KIND = "train_infer_logprob_parity"


@dataclass(frozen=True)
class CellOutput:
    cell_id: str
    selected_logp: dict[tuple[str, int], torch.Tensor]
    loss: torch.Tensor | None
    grads: dict[str, torch.Tensor]
    first_drift_node: str | None
    node_digests: dict[str, str]
    requested_backend: str
    actual_backend: str
    node_token_digests: dict[str, dict[str, str]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "n_logp_tokens": len(self.selected_logp),
            "loss": (None if self.loss is None else float(self.loss.detach().float().cpu())),
            "grad_names": sorted(self.grads),
            "first_drift_node": self.first_drift_node,
            "node_digests": dict(self.node_digests),
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
        }


@dataclass(frozen=True)
class ChainGateReport:
    backend_profile: str
    workload_id: str
    fixture_hash: str
    config_fingerprint: dict[str, Any]
    weight_source: str
    weight_hash: str
    seed: int
    workload_seed: int
    device: str
    gpu_name: str | None
    compute_capability: str | None
    backend_provenance: BackendProvenance
    runtime_backend_observations: dict[str, dict[str, Any]]
    backward_runtime_observations: dict[str, dict[str, Any]]
    cells: dict[str, CellOutput]
    invariance: tuple[TensorComparisonDetail, ...]
    gradient_invariance: tuple[TensorComparisonDetail, ...]
    accuracy: tuple[TensorComparisonDetail, ...]
    gradient_accuracy: tuple[TensorComparisonDetail, ...]
    accuracy_aggregates: tuple[LogprobAggregateVerdict, ...]
    train_infer: LogprobAggregateVerdict | None
    train_infer_bn: LogprobAggregateVerdict | None
    decode_prefill: tuple[tuple[str, LogprobAggregateVerdict], ...]
    first_drift: str | None
    aggregates: LogprobAggregateVerdict | None
    gradient_scope: str
    required_grad_names: tuple[str, ...]
    all_parameter_gradients: bool
    representative_case_ids: tuple[str, ...]
    workflow_url: str | None
    c8_evidence_path: str
    c8_source_commit: str | None
    passed: bool
    backward_executed: bool
    train_infer_executed: bool
    accuracy_executed: bool
    gradient_accuracy_executed: bool
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_profile": self.backend_profile,
            "workload_id": self.workload_id,
            "fixture_hash": self.fixture_hash,
            "config_fingerprint": dict(self.config_fingerprint),
            "weight_source": self.weight_source,
            "weight_hash": self.weight_hash,
            "seed": self.seed,
            "workload_seed": self.workload_seed,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "compute_capability": self.compute_capability,
            "backend_provenance": self.backend_provenance.to_dict(),
            "runtime_backend_observations": {
                key: dict(value) for key, value in self.runtime_backend_observations.items()
            },
            "backward_runtime_observations": {
                key: dict(value) for key, value in self.backward_runtime_observations.items()
            },
            "cells": {key: value.to_dict() for key, value in self.cells.items()},
            "invariance": [item.to_dict() for item in self.invariance],
            "gradient_invariance": [item.to_dict() for item in self.gradient_invariance],
            "accuracy": [item.to_dict() for item in self.accuracy],
            "gradient_accuracy": [item.to_dict() for item in self.gradient_accuracy],
            "accuracy_aggregates": [item.to_dict() for item in self.accuracy_aggregates],
            "train_infer": (None if self.train_infer is None else self.train_infer.to_dict()),
            "train_infer_bn": (
                None if self.train_infer_bn is None else self.train_infer_bn.to_dict()
            ),
            "decode_prefill": [
                {"case_id": case_id, **verdict.to_dict()}
                for case_id, verdict in self.decode_prefill
            ],
            "first_drift": self.first_drift,
            "aggregates": (None if self.aggregates is None else self.aggregates.to_dict()),
            "gradient_scope": self.gradient_scope,
            "required_grad_names": sorted(self.required_grad_names),
            "all_parameter_gradients": self.all_parameter_gradients,
            "representative_case_ids": list(self.representative_case_ids),
            "workflow_url": self.workflow_url,
            "c8_evidence_path": self.c8_evidence_path,
            "c8_source_commit": self.c8_source_commit,
            "passed": self.passed,
            "backward_executed": self.backward_executed,
            "train_infer_executed": self.train_infer_executed,
            "accuracy_executed": self.accuracy_executed,
            "gradient_accuracy_executed": self.gradient_accuracy_executed,
            "disclaimer": self.disclaimer,
        }


def build_model(
    *,
    backend_profile: str,
    weights_mode: str,
    weights_path: str | None,
    device: torch.device,
    dtype: torch.dtype,
    manifest: WS1Manifest | None = None,
    allow_pytorch_gold: bool = False,
) -> Qwen3DenseBIModel:
    m = manifest if manifest is not None else load_manifest()
    spec = Qwen3DenseSpec.from_manifest(m)
    ops = load_profile_ops(backend_profile, m, allow_pytorch_gold=allow_pytorch_gold)
    if weights_mode == "synthetic":
        weights = Qwen3DenseWeights.synthetic(spec, device=device, dtype=dtype, seed=m.seed)
    elif weights_mode in {"hf", "required"}:
        if not weights_path:
            raise RuntimeError("C10/C11 require --weights-path to the pinned Qwen3-8B snapshot")
        weights = Qwen3DenseWeights.from_hf(spec, weights_path, device=device, dtype=dtype)
    else:
        raise ValueError(f"unknown weights_mode {weights_mode!r}")
    return Qwen3DenseBIModel(spec, weights, ops, execution_dtype=dtype)


def run_fp32_reference_cell(
    *,
    backend_profile: str,
    weights_mode: str,
    weights_path: str | None,
    device: torch.device,
    manifest: WS1Manifest | None = None,
    run_backward: bool = True,
) -> CellOutput:
    """Run BN/full on the FP32 gold topology. Separate from the candidate model."""

    m = manifest if manifest is not None else load_manifest()
    # LOCAL-DEVIATION (bf16 reference): 64 GB HBM cannot fit the
    # FP32-reference full-model backward; see the PR description.
    reference = build_model(
        backend_profile=backend_profile,
        weights_mode=weights_mode,
        weights_path=weights_path,
        device=device,
        dtype=torch.bfloat16,
        manifest=m,
        allow_pytorch_gold=True,
    )
    batch = build_logical_batch(m)
    _configure_required_gradients(reference, enabled=run_backward)
    cell = _run_padded_cell(
        reference,
        batch,
        cell_id="BN/full",
        pad_side="right",
        manifest=m,
        run_backward=run_backward,
        active_token_denominator=batch.active_token_count(),
    )
    _configure_required_gradients(reference, enabled=False)
    del reference
    empty_cache(device.type)
    return cell


def run_chain_gate(
    *,
    backend_profile: str,
    model: Qwen3DenseBIModel,
    contract: Mapping[str, Any] | None = None,
    manifest: WS1Manifest | None = None,
    run_backward: bool = True,
    run_train_infer: bool = True,
    padding_sides: Sequence[str] = ("right", "left"),
    execution_seed: int | None = None,
    reference_cell: CellOutput | None = None,
) -> ChainGateReport:
    c = contract if contract is not None else load_contract()
    m = manifest if manifest is not None else load_manifest()
    policy = resolve_dtype_policy(c)
    batch = build_logical_batch(m)
    cells: dict[str, CellOutput] = {}
    resolved_seed = m.seed if execution_seed is None else int(execution_seed)
    manual_seed_all(device_type_for_profile(backend_profile), resolved_seed)

    reset_backward_runtime()
    _configure_required_gradients(model, enabled=run_backward)

    global_active = batch.active_token_count()
    cells["BN/full"] = _run_padded_cell(
        model,
        batch,
        cell_id="BN/full",
        pad_side="right",
        manifest=m,
        run_backward=run_backward,
        active_token_denominator=global_active,
    )
    canonical = cells["BN/full"]
    grad_details: list[TensorComparisonDetail] = []

    cells["B1-singleton_aggregate/full"] = _run_singleton_cell(
        model,
        batch,
        cell_id="B1-singleton_aggregate/full",
        manifest=m,
        run_backward=run_backward,
        active_token_denominator=global_active,
    )
    if run_backward:
        grad_details.extend(
            _compare_required_grads(
                canonical,
                cells["B1-singleton_aggregate/full"],
                contract=c,
                dtype=policy.execution_dtype,
                backend_profile=backend_profile,
            )
        )
        _release_parameter_grads(cells["B1-singleton_aggregate/full"])

    cells["BN/chunked"] = _run_chunked_cell(
        model,
        batch,
        cell_id="BN/chunked",
        manifest=m,
        run_backward=run_backward,
        active_token_denominator=global_active,
    )
    if run_backward:
        grad_details.extend(
            _compare_required_grads(
                canonical,
                cells["BN/chunked"],
                contract=c,
                dtype=policy.execution_dtype,
                backend_profile=backend_profile,
            )
        )
        _release_parameter_grads(cells["BN/chunked"])

    cells["B1-singleton_aggregate/chunked"] = _run_singleton_chunked_cell(
        model,
        batch,
        cell_id="B1-singleton_aggregate/chunked",
        manifest=m,
        run_backward=run_backward,
        active_token_denominator=global_active,
    )
    if run_backward:
        grad_details.extend(
            _compare_required_grads(
                canonical,
                cells["B1-singleton_aggregate/chunked"],
                contract=c,
                dtype=policy.execution_dtype,
                backend_profile=backend_profile,
            )
        )
        _release_parameter_grads(cells["B1-singleton_aggregate/chunked"])

    if run_backward:
        _validate_required_gradient_cells(cells)
    for side in padding_sides:
        cell_id = f"BN/padded_{side}"
        cells[cell_id] = _run_padded_cell(
            model,
            batch,
            cell_id=cell_id,
            pad_side=side,
            manifest=m,
            run_backward=run_backward,
            active_token_denominator=global_active,
        )
        if run_backward:
            grad_details.extend(
                _compare_required_grads(
                    canonical,
                    cells[cell_id],
                    contract=c,
                    dtype=policy.execution_dtype,
                    backend_profile=backend_profile,
                )
            )
            _release_parameter_grads(cells[cell_id])

    perm = batch_permutation_from_manifest(m)
    permuted = permute_batch(batch, perm)
    cells["BN/permuted"] = _run_padded_cell(
        model,
        permuted,
        cell_id="BN/permuted",
        pad_side="right",
        manifest=m,
        run_backward=run_backward,
        active_token_denominator=global_active,
    )
    if run_backward:
        grad_details.extend(
            _compare_required_grads(
                canonical,
                cells["BN/permuted"],
                contract=c,
                dtype=policy.execution_dtype,
                backend_profile=backend_profile,
            )
        )
        _release_parameter_grads(cells["BN/permuted"])

    packing_status = str(m.fixtures.get("packing", {}).get("status", "unsupported"))
    if packing_status == "supported":
        cells["BN/packed"] = _run_packed_cell(
            model,
            batch,
            cell_id="BN/packed",
            run_backward=run_backward,
            active_token_denominator=global_active,
        )
    _configure_required_gradients(model, enabled=False)

    inv_details: list[TensorComparisonDetail] = []
    for cell_id in (
        "B1-singleton_aggregate/full",
        "BN/chunked",
        "B1-singleton_aggregate/chunked",
    ):
        inv_details.append(
            _compare_logp_maps(
                canonical.selected_logp,
                cells[cell_id].selected_logp,
                contract=c,
                judgment="forward_invariance",
                dtype=policy.execution_dtype,
                backend_profile=backend_profile,
                config_pair=("BN/full", cell_id),
            )
        )

    for side in padding_sides:
        cell_id = f"BN/padded_{side}"
        inv_details.append(
            _compare_logp_maps(
                canonical.selected_logp,
                cells[cell_id].selected_logp,
                contract=c,
                judgment="forward_invariance",
                dtype=policy.execution_dtype,
                backend_profile=backend_profile,
                config_pair=("BN/full", cell_id),
            )
        )

    inv_details.append(
        _compare_logp_maps(
            canonical.selected_logp,
            cells["BN/permuted"].selected_logp,
            contract=c,
            judgment="forward_invariance",
            dtype=policy.execution_dtype,
            backend_profile=backend_profile,
            config_pair=("BN/full", "BN/permuted"),
        )
    )
    packing_grad_details: list[TensorComparisonDetail] = []
    if packing_status == "supported":
        # Packing changes attention reduction width vs padded BN, so this
        # layout axis uses C1 forward_accuracy, not the #150 bitwise 2x2.
        inv_details.append(
            _compare_logp_maps(
                canonical.selected_logp,
                cells["BN/packed"].selected_logp,
                contract=c,
                judgment="forward_accuracy",
                dtype=policy.execution_dtype,
                backend_profile=backend_profile,
                config_pair=("BN/full", "BN/packed"),
            )
        )

    acc_details: list[TensorComparisonDetail] = []
    acc_aggregates: list[LogprobAggregateVerdict] = []
    grad_acc_details: list[TensorComparisonDetail] = []
    if reference_cell is not None:
        acc_details.append(
            _compare_logp_maps(
                canonical.selected_logp,
                reference_cell.selected_logp,
                contract=c,
                judgment="forward_accuracy",
                dtype=policy.execution_dtype,
                backend_profile=backend_profile,
                config_pair=("BN/full", "fp32_reference"),
            )
        )
        acc_aggregates.append(
            _logp_aggregate_verdict(
                canonical.selected_logp,
                reference_cell.selected_logp,
                contract=c,
                report_kind="forward_accuracy",
            )
        )
        for cell_id in (
            "B1-singleton_aggregate/full",
            "BN/chunked",
            "B1-singleton_aggregate/chunked",
            "BN/padded_right",
            "BN/packed",
            "BN/padded_left",
            "BN/permuted",
        ):
            if cell_id in cells:
                acc_aggregates.append(
                    _logp_aggregate_verdict(
                        cells[cell_id].selected_logp,
                        reference_cell.selected_logp,
                        contract=c,
                        report_kind="forward_accuracy",
                    )
                )
        if "BN/packed" in cells:
            acc_details.append(
                _compare_logp_maps(
                    cells["BN/packed"].selected_logp,
                    reference_cell.selected_logp,
                    contract=c,
                    judgment="forward_accuracy",
                    dtype=policy.execution_dtype,
                    backend_profile=backend_profile,
                    config_pair=("BN/packed", "fp32_reference"),
                )
            )
        if run_backward:
            grad_acc_details.extend(
                _compare_required_grads(
                    canonical,
                    reference_cell,
                    contract=c,
                    dtype=policy.execution_dtype,
                    backend_profile=backend_profile,
                    judgment="gradient_accuracy",
                    config_pair=("BN/full", "fp32_reference"),
                )
            )
            if "BN/packed" in cells:
                grad_acc_details.extend(
                    _compare_required_grads(
                        cells["BN/packed"],
                        reference_cell,
                        contract=c,
                        dtype=policy.execution_dtype,
                        backend_profile=backend_profile,
                        judgment="gradient_accuracy",
                        config_pair=("BN/packed", "fp32_reference"),
                    )
                )

    # Accuracy comparisons are complete; retain only gradient names in the report.
    for completed_cell in cells.values():
        _release_parameter_grads(completed_cell)
    if reference_cell is not None:
        _release_parameter_grads(reference_cell)

    train_infer = None
    train_infer_bn = None
    decode_prefill: tuple[tuple[str, LogprobAggregateVerdict], ...] = ()
    if run_train_infer:
        train_infer = _train_infer_parity(model, batch, contract=c, manifest=m)
        decode_prefill = _full_model_decode_sweep(model, batch, contract=c, manifest=m)
        train_infer_bn = next(
            (verdict for case_id, verdict in decode_prefill if case_id == "decode-bn-padded-right"),
            None,
        )

    first_drift = _locate_first_drift(
        model,
        cells,
        inv_details,
        grad_details + packing_grad_details + acc_details + grad_acc_details,
        train_infer,
        train_infer_bn,
        decode_prefill,
    )

    family = str(m.backend_profiles[backend_profile]["backend_family"])
    runtime_observations = model.profile_ops.validated_runtime_observations()
    observed_families = {
        _backend_family(str(value["actual_backend"])) for value in runtime_observations.values()
    }
    if observed_families != {family}:
        raise RuntimeError(
            f"profile {backend_profile!r} observed backend families "
            f"{sorted(observed_families)}, expected only {family!r}"
        )
    backward_runtime = snapshot_backward_runtime()
    expected_family = family
    for kind in ("lm_head", "rms_norm", "det_gemm", "embedding"):
        event = backward_runtime.get(kind)
        if event is None or int(event.get("execution_count", 0)) <= 0:
            raise RuntimeError(
                f"profile {backend_profile!r} has no runtime backward record for {kind!r}"
            )
        if str(event.get("family")) != expected_family:
            raise RuntimeError(
                f"profile {backend_profile!r} node {kind!r} backward family "
                f"{event.get('family')!r}, expected {expected_family!r}"
            )
        if not event.get("kernel_id"):
            raise RuntimeError(
                f"profile {backend_profile!r} node {kind!r} missing backward kernel_id"
            )
        if not event.get("kernel_ids"):
            raise RuntimeError(
                f"profile {backend_profile!r} node {kind!r} missing backward kernel_ids"
            )
        if not event.get("implementation_ids"):
            raise RuntimeError(
                f"profile {backend_profile!r} node {kind!r} missing backward implementation_ids"
            )
    provenance = make_profile_provenance(
        backend_profile=backend_profile,
        contract=c,
        requested_backend=family,
        actual_backend=family,
        output_dtype=policy.output_dtype_default,
    )
    device = next(iter(model.weights.tensors.values())).device
    cc = compute_capability(device) if device.type in ACCELERATOR_TYPES else None

    # Cross-cell logprob aggregates (BN vs B1) as the named chain metrics.
    lhs, rhs, mask = _aligned_logp_vectors(
        canonical.selected_logp, cells["B1-singleton_aggregate/full"].selected_logp
    )
    roles = resolve_comparison_roles(c, "forward_invariance")
    # Invariance is bitwise; still report the three aggregates for the logprob outputs.
    train_roles = resolve_comparison_roles(c, _TRAIN_INFER_KIND)
    aggregates = judge_logprob_aggregates(
        compute_logprob_aggregates(
            lhs,
            rhs,
            mask,
            contract=c,
            report_kind=_TRAIN_INFER_KIND,
            clip_interval=default_clip_interval(c),
            comparison_lhs_role=train_roles.comparison_lhs_role,
            comparison_rhs_role=train_roles.comparison_rhs_role,
        ),
        c,
        execution_dtype=policy.execution_dtype,
    )

    passed = (
        run_backward
        and run_train_infer
        and reference_cell is not None
        and all(item.passed for item in inv_details)
        and all(item.passed for item in grad_details)
        and all(item.passed for item in packing_grad_details)
        and all(item.passed for item in acc_details)
        and all(item.passed for item in acc_aggregates)
        and all(item.passed for item in grad_acc_details)
        and train_infer is not None
        and train_infer.passed
        and train_infer_bn is not None
        and train_infer_bn.passed
        and bool(decode_prefill)
        and all(verdict.passed for _case_id, verdict in decode_prefill)
    )
    # Bitwise invariance is the #150 matrix verdict; aggregates are reported
    # but BN-vs-B1 bitwise already covers output identity.
    del roles
    return ChainGateReport(
        backend_profile=backend_profile,
        workload_id=m.workload_id,
        fixture_hash=fixture_hash(m, batch=batch),
        config_fingerprint=dict(m.model_identity["config_fingerprint"]),
        weight_source=model.weights.source,
        weight_hash=model.weights.content_hash,
        seed=resolved_seed,
        workload_seed=m.seed,
        device=str(device),
        gpu_name=_gpu_name(device),
        compute_capability=cc,
        backend_provenance=provenance,
        runtime_backend_observations=runtime_observations,
        backward_runtime_observations=backward_runtime,
        cells=cells,
        invariance=tuple(inv_details),
        gradient_invariance=tuple(grad_details),
        accuracy=tuple(acc_details),
        gradient_accuracy=tuple(grad_acc_details + packing_grad_details),
        accuracy_aggregates=tuple(acc_aggregates),
        train_infer=train_infer,
        train_infer_bn=train_infer_bn,
        decode_prefill=decode_prefill,
        first_drift=first_drift,
        aggregates=aggregates,
        gradient_scope=GRADIENT_SCOPE,
        required_grad_names=tuple(REQUIRED_GRAD_NAMES),
        all_parameter_gradients=True,
        representative_case_ids=_representative_case_ids(m, backend_profile),
        workflow_url=_workflow_url(),
        c8_evidence_path=_c8_evidence_path(),
        c8_source_commit=_c8_source_commit(),
        passed=passed,
        backward_executed=run_backward,
        train_infer_executed=run_train_infer,
        accuracy_executed=reference_cell is not None,
        gradient_accuracy_executed=reference_cell is not None and run_backward,
        disclaimer=(
            "C10 compares tensor.grad from a real training-style backward over "
            "every official Qwen3-8B Dense trainable leaf. Logprob accuracy "
            "uses max_abs_dlogp / approx_kl0 / clipfrac0 against the FP32 gold "
            "cell. Public WS1 EXIT still requires C11 CI and parent #266 A/B/Final."
        ),
    )


def _run_packed_cell(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    cell_id: str,
    run_backward: bool,
    active_token_denominator: int | None = None,
) -> CellOutput:
    layout = apply_packing(batch)
    device = _device(model)
    input_ids = torch.tensor([layout.physical_token_ids], device=device, dtype=torch.long)
    attn = torch.ones_like(input_ids, dtype=torch.bool)
    positions: list[int] = []
    for length in layout.segment_lengths:
        positions.extend(range(int(length)))
    pos = torch.tensor([positions], device=device, dtype=torch.long)
    loss_mask = torch.tensor([layout.physical_loss_mask], device=device, dtype=torch.bool)
    restore = (tuple(layout.restore_map),)
    out = model.forward(
        input_ids,
        attention_mask=attn,
        position_ids=pos,
        capture_nodes=True,
        segment_lengths=layout.segment_lengths,
    )
    node_token_digests = _node_token_fingerprints(model, restore)
    return _finish_cell(
        model,
        out["logits"],
        input_ids,
        loss_mask,
        restore=restore,
        score_logits=out.get("score_logits"),
        restores=(restore,),
        cell_id=cell_id,
        run_backward=run_backward,
        active_token_denominator=active_token_denominator,
        node_token_digests=node_token_digests,
    )


def _run_padded_cell(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    cell_id: str,
    pad_side: str,
    manifest: WS1Manifest,
    run_backward: bool,
    defer_backward: bool = False,
    logical_sample_rank: Mapping[str, int] | None = None,
    active_token_denominator: int | None = None,
) -> CellOutput:
    padded = apply_padding(batch, pad_side=pad_side, manifest=manifest)
    input_ids = torch.tensor(padded.physical_token_ids, device=_device(model), dtype=torch.long)
    attn = torch.tensor(padded.physical_attention_mask, device=_device(model), dtype=torch.bool)
    pos = torch.tensor(padded.physical_position_ids, device=_device(model), dtype=torch.long)
    loss_mask = torch.tensor(padded.physical_loss_mask, device=_device(model), dtype=torch.bool)
    return _forward_cell(
        model,
        input_ids,
        attn,
        pos,
        loss_mask,
        restore=padded.restore_map,
        restores=(padded.restore_map,),
        cell_id=cell_id,
        run_backward=run_backward,
        defer_backward=defer_backward,
        logical_sample_rank=logical_sample_rank,
        active_token_denominator=active_token_denominator,
    )


def _run_singleton_cell(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    cell_id: str,
    manifest: WS1Manifest,
    run_backward: bool,
    active_token_denominator: int | None = None,
) -> CellOutput:
    merged: dict[tuple[str, int], torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {}
    node_token_digests: dict[str, dict[str, str]] = {}
    loss_acc: torch.Tensor | None = None

    def _collect(cell: CellOutput) -> None:
        nonlocal loss_acc
        merged.update(cell.selected_logp)
        if cell.loss is None:
            raise RuntimeError("singleton cell produced no loss")
        loss_acc = cell.loss if loss_acc is None else loss_acc + cell.loss
        _merge_node_token_digests(node_token_digests, cell.node_token_digests)

    sample_rank = {
        sample.sample_id: index
        for index, sample in enumerate(sorted(batch.samples, key=lambda item: item.sample_id))
    }
    if run_backward:
        for name in REQUIRED_GRAD_NAMES:
            model.weights.tensors[name].grad = None
        with canonical_backward_session() as session:
            for sample in batch.samples:
                single = LogicalBatch(
                    workload_id=batch.workload_id,
                    seed=batch.seed,
                    samples=(sample,),
                    cell_id=cell_id,
                )
                _collect(
                    _run_padded_cell(
                        model,
                        single,
                        cell_id=cell_id,
                        pad_side="right",
                        manifest=manifest,
                        run_backward=False,
                        defer_backward=True,
                        logical_sample_rank=sample_rank,
                        active_token_denominator=active_token_denominator,
                    )
                )
            if loss_acc is None:
                raise RuntimeError("singleton aggregate produced no loss")
            loss_acc.backward()
            session.validate_complete()
        grads = _collect_parameter_grads(model, cell_id=cell_id)
    else:
        for sample in batch.samples:
            single = LogicalBatch(
                workload_id=batch.workload_id,
                seed=batch.seed,
                samples=(sample,),
                cell_id=cell_id,
            )
            _collect(
                _run_padded_cell(
                    model,
                    single,
                    cell_id=cell_id,
                    pad_side="right",
                    manifest=manifest,
                    run_backward=False,
                    logical_sample_rank=sample_rank,
                    active_token_denominator=active_token_denominator,
                )
            )

    return CellOutput(
        cell_id=cell_id,
        selected_logp=merged,
        loss=None if loss_acc is None else loss_acc.detach(),
        grads=grads,
        first_drift_node=None,
        node_digests=_combined_node_digests(node_token_digests),
        requested_backend=model.profile_ops.provenance["attention"]["requested_backend"],
        actual_backend=model.profile_ops.provenance["attention"]["actual_backend"],
        node_token_digests=node_token_digests,
    )


def _run_chunked_cell(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    cell_id: str,
    manifest: WS1Manifest,
    run_backward: bool,
    defer_backward: bool = False,
    logical_sample_rank: Mapping[str, int] | None = None,
    active_token_denominator: int | None = None,
) -> CellOutput:
    if not run_backward and not defer_backward:
        return _run_chunked_cell_impl(
            model,
            batch,
            cell_id=cell_id,
            manifest=manifest,
            defer_backward=False,
            logical_sample_rank=logical_sample_rank,
            active_token_denominator=active_token_denominator,
        )
    if active_session() is not None:
        return _run_chunked_cell_impl(
            model,
            batch,
            cell_id=cell_id,
            manifest=manifest,
            defer_backward=True,
            logical_sample_rank=logical_sample_rank,
            active_token_denominator=active_token_denominator,
        )
    if defer_backward:
        raise RuntimeError("deferred chunked backward requires an active canonical session")
    for name in REQUIRED_GRAD_NAMES:
        model.weights.tensors[name].grad = None
    with canonical_backward_session() as session:
        cell = _run_chunked_cell_impl(
            model,
            batch,
            cell_id=cell_id,
            manifest=manifest,
            defer_backward=True,
            logical_sample_rank=logical_sample_rank,
            active_token_denominator=active_token_denominator,
        )
        if cell.loss is None:
            raise RuntimeError("chunked cell produced no loss")
        cell.loss.backward()
        session.validate_complete()
    return CellOutput(
        cell_id=cell.cell_id,
        selected_logp=cell.selected_logp,
        loss=None if cell.loss is None else cell.loss.detach(),
        grads=_collect_parameter_grads(model, cell_id=cell_id),
        first_drift_node=cell.first_drift_node,
        node_digests=cell.node_digests,
        requested_backend=cell.requested_backend,
        actual_backend=cell.actual_backend,
        node_token_digests=cell.node_token_digests,
    )


def _run_chunked_cell_impl(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    cell_id: str,
    manifest: WS1Manifest,
    defer_backward: bool,
    logical_sample_rank: Mapping[str, int] | None,
    active_token_denominator: int | None,
) -> CellOutput:
    plan = chunk_plan_from_manifest(manifest)
    padded = apply_padding(batch, pad_side="right", manifest=manifest)
    input_ids = torch.tensor(padded.physical_token_ids, device=_device(model), dtype=torch.long)
    attn = torch.tensor(padded.physical_attention_mask, device=_device(model), dtype=torch.bool)
    pos = torch.tensor(padded.physical_position_ids, device=_device(model), dtype=torch.long)
    loss_mask = torch.tensor(padded.physical_loss_mask, device=_device(model), dtype=torch.bool)
    cache = model.allocate_cache(input_ids.shape[0], input_ids.shape[1], _device(model))
    logical_keys = _logical_key_tensor(
        padded.restore_map, device=input_ids.device, sample_rank=logical_sample_rank
    )
    logits_parts: list[torch.Tensor] = []
    score_logits_parts: list[torch.Tensor] = []
    node_token_digests: dict[str, dict[str, str]] = {}
    restores: list[tuple[tuple[tuple[str, int] | None, ...], ...]] = []
    seq = input_ids.shape[1]
    forward_context = torch.no_grad() if defer_backward else contextlib.nullcontext()
    with forward_context:
        for start in range(0, seq, plan.chunk_size):
            end = min(start + plan.chunk_size, seq)
            step = model.forward(
                input_ids[:, start:end],
                attention_mask=attn[:, start:end],
                position_ids=pos[:, start:end],
                kv_cache=cache,
                capture_nodes=True,
                logical_keys=(logical_keys[:, start:end] if active_session() is not None else None),
            )
            logits_parts.append(step["logits"])
            score_logits_parts.append(_scoring_logits(step))
            chunk_restore = tuple(tuple(row[start:end]) for row in padded.restore_map)
            restores.append(chunk_restore)
            _merge_node_token_digests(
                node_token_digests, _node_token_fingerprints(model, chunk_restore)
            )
    if defer_backward:
        observed_logits = torch.cat(logits_parts, dim=1)
        observed_score_logits = torch.cat(score_logits_parts, dim=1)
        del cache, logits_parts, score_logits_parts
        chunked = model.forward_chunked_training(
            input_ids,
            chunk_size=plan.chunk_size,
            attention_mask=attn,
            position_ids=pos,
            logical_keys=logical_keys,
        )
        if not torch.equal(observed_logits, chunked["logits"].detach()):
            raise RuntimeError("chunked training logits differ from stateful chunked prefill")
        if not torch.equal(observed_score_logits, chunked["score_logits"].detach()):
            raise RuntimeError("chunked training score logits differ from stateful chunked prefill")
        del observed_logits, observed_score_logits
        return _finish_cell(
            model,
            chunked["logits"],
            input_ids,
            loss_mask,
            restore=padded.restore_map,
            score_logits=chunked.get("score_logits"),
            restores=tuple(restores),
            cell_id=cell_id,
            run_backward=False,
            retain_loss=True,
            active_token_denominator=active_token_denominator,
            node_token_digests=node_token_digests,
        )
    return _finish_cell(
        model,
        torch.cat(logits_parts, dim=1),
        input_ids,
        loss_mask,
        restore=padded.restore_map,
        score_logits=torch.cat(score_logits_parts, dim=1),
        restores=tuple(restores),
        cell_id=cell_id,
        run_backward=False,
        retain_loss=defer_backward,
        active_token_denominator=active_token_denominator,
        node_token_digests=node_token_digests,
    )


def _run_singleton_chunked_cell(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    cell_id: str,
    manifest: WS1Manifest,
    run_backward: bool,
    active_token_denominator: int | None = None,
) -> CellOutput:
    merged: dict[tuple[str, int], torch.Tensor] = {}
    node_token_digests: dict[str, dict[str, str]] = {}
    loss_acc: torch.Tensor | None = None
    sample_rank = {
        sample.sample_id: index
        for index, sample in enumerate(sorted(batch.samples, key=lambda item: item.sample_id))
    }

    def _collect(cell: CellOutput) -> None:
        nonlocal loss_acc
        merged.update(cell.selected_logp)
        if cell.loss is None:
            raise RuntimeError("singleton chunked cell produced no loss")
        loss_acc = cell.loss if loss_acc is None else loss_acc + cell.loss
        _merge_node_token_digests(node_token_digests, cell.node_token_digests)

    if run_backward:
        for name in REQUIRED_GRAD_NAMES:
            model.weights.tensors[name].grad = None
        with canonical_backward_session() as session:
            for sample in batch.samples:
                single = LogicalBatch(
                    workload_id=batch.workload_id,
                    seed=batch.seed,
                    samples=(sample,),
                    cell_id=cell_id,
                )
                _collect(
                    _run_chunked_cell(
                        model,
                        single,
                        cell_id=cell_id,
                        manifest=manifest,
                        run_backward=False,
                        defer_backward=True,
                        logical_sample_rank=sample_rank,
                        active_token_denominator=active_token_denominator,
                    )
                )
            if loss_acc is None:
                raise RuntimeError("singleton chunked aggregate produced no loss")
            loss_acc.backward()
            session.validate_complete()
        grads = _collect_parameter_grads(model, cell_id=cell_id)
    else:
        for sample in batch.samples:
            single = LogicalBatch(
                workload_id=batch.workload_id, seed=batch.seed, samples=(sample,), cell_id=cell_id
            )
            _collect(
                _run_chunked_cell(
                    model,
                    single,
                    cell_id=cell_id,
                    manifest=manifest,
                    run_backward=False,
                    logical_sample_rank=sample_rank,
                    active_token_denominator=active_token_denominator,
                )
            )
        grads = {}
    return CellOutput(
        cell_id=cell_id,
        selected_logp=merged,
        loss=None if loss_acc is None else loss_acc.detach(),
        grads=grads,
        first_drift_node=None,
        node_digests=_combined_node_digests(node_token_digests),
        requested_backend=model.profile_ops.provenance["attention"]["requested_backend"],
        actual_backend=model.profile_ops.provenance["attention"]["actual_backend"],
        node_token_digests=node_token_digests,
    )


def _logical_key_tensor(
    restore: Sequence[Sequence[tuple[str, int] | None]],
    *,
    device: torch.device,
    sample_rank: Mapping[str, int] | None = None,
) -> torch.Tensor:
    if sample_rank is None:
        sample_ids = sorted({key[0] for row in restore for key in row if key is not None})
        sample_rank = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    keys = torch.full(
        (len(restore), len(restore[0]), 2),
        -1,
        device=device,
        dtype=torch.long,
    )
    for batch_index, row in enumerate(restore):
        for physical_index, key in enumerate(row):
            if key is not None:
                keys[batch_index, physical_index, 0] = sample_rank[key[0]]
                keys[batch_index, physical_index, 1] = int(key[1])
    return keys


def _forward_cell(
    model: Qwen3DenseBIModel,
    input_ids: torch.Tensor,
    attn: torch.Tensor,
    pos: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    restore: Sequence[Sequence[tuple[str, int] | None]],
    restores: Sequence[Sequence[Sequence[tuple[str, int] | None]]],
    cell_id: str,
    run_backward: bool,
    defer_backward: bool = False,
    logical_sample_rank: Mapping[str, int] | None = None,
    active_token_denominator: int | None = None,
) -> CellOutput:
    if not run_backward and not defer_backward:
        out = model.forward(
            input_ids,
            attention_mask=attn,
            position_ids=pos,
            capture_nodes=True,
        )
        node_token_digests = _node_token_fingerprints(model, restore)
        return _finish_cell(
            model,
            out["logits"],
            input_ids,
            loss_mask,
            restore=restore,
            restores=restores,
            score_logits=out.get("score_logits"),
            cell_id=cell_id,
            run_backward=False,
            active_token_denominator=active_token_denominator,
            node_token_digests=node_token_digests,
        )

    logical_keys = _logical_key_tensor(
        restore, device=input_ids.device, sample_rank=logical_sample_rank
    )
    session = active_session()
    if session is not None:
        out = model.forward(
            input_ids,
            attention_mask=attn,
            position_ids=pos,
            capture_nodes=True,
            logical_keys=logical_keys,
        )
        node_token_digests = _node_token_fingerprints(model, restore)
        return _finish_cell(
            model,
            out["logits"],
            input_ids,
            loss_mask,
            restore=restore,
            restores=restores,
            score_logits=out.get("score_logits"),
            cell_id=cell_id,
            run_backward=False,
            retain_loss=defer_backward,
            active_token_denominator=active_token_denominator,
            node_token_digests=node_token_digests,
        )
    if defer_backward:
        raise RuntimeError("deferred backward requires an active canonical session")
    with canonical_backward_session() as session:
        out = model.forward(
            input_ids,
            attention_mask=attn,
            position_ids=pos,
            capture_nodes=True,
            logical_keys=logical_keys,
        )
        node_token_digests = _node_token_fingerprints(model, restore)
        cell = _finish_cell(
            model,
            out["logits"],
            input_ids,
            loss_mask,
            restore=restore,
            restores=restores,
            score_logits=out.get("score_logits"),
            cell_id=cell_id,
            run_backward=True,
            active_token_denominator=active_token_denominator,
            node_token_digests=node_token_digests,
        )
        session.validate_complete()
        return cell


def _finish_cell(
    model: Qwen3DenseBIModel,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    restore: Sequence[Sequence[tuple[str, int] | None]],
    restores: Sequence[Sequence[Sequence[tuple[str, int] | None]]],
    cell_id: str,
    run_backward: bool,
    loss: torch.Tensor | None = None,
    retain_loss: bool = False,
    score_logits: torch.Tensor | None = None,
    active_token_denominator: int | None = None,
    node_token_digests: dict[str, dict[str, str]] | None = None,
) -> CellOutput:
    # Standard causal shift: logits[t] predicts token[t+1].
    pred = (logits if score_logits is None else score_logits)[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = model._selected_logp(pred, targets)
    active = loss_mask[:, 1:].to(dtype=torch.bool)
    logp = logp.masked_fill(~active, 0.0)
    selected: dict[tuple[str, int], torch.Tensor] = {}
    for batch_idx, row in enumerate(restore):
        # restore is aligned to physical tokens; selected logp lives on the next token.
        for phys, key in enumerate(row[:-1]):
            nxt = row[phys + 1]
            if key is None or nxt is None:
                continue
            if not bool(active[batch_idx, phys].item()):
                continue
            selected[(nxt[0], nxt[1])] = logp[batch_idx, phys].detach()

    grads: dict[str, torch.Tensor] = {}
    cell_loss = loss
    if cell_loss is None:
        denom = float(
            active_token_denominator if active_token_denominator else int(active.sum().item()) or 1
        )
        cell_loss = -(logp.float() * active.float()).sum() / denom
    if run_backward:
        # Enable only REQUIRED_GRAD_NAMES before the graph is built. Compare
        # the real leaf .grad produced by that backward, not a recomputed VJP.
        for name in REQUIRED_GRAD_NAMES:
            model.weights.tensors[name].grad = None
        cell_loss.backward()
        grads = _collect_parameter_grads(model, cell_id=cell_id)
    del restores

    return CellOutput(
        cell_id=cell_id,
        selected_logp=selected,
        loss=(cell_loss if retain_loss else cell_loss.detach()) if cell_loss is not None else None,
        grads=grads,
        first_drift_node=None,
        node_digests=_combined_node_digests(node_token_digests or {}),
        requested_backend=model.profile_ops.provenance["attention"]["requested_backend"],
        actual_backend=model.profile_ops.provenance["attention"]["actual_backend"],
        node_token_digests=dict(node_token_digests or {}),
    )


def _scoring_logits(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return outputs.get("score_logits", outputs["logits"])


def _train_infer_parity(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    contract: Mapping[str, Any],
    manifest: WS1Manifest,
) -> LogprobAggregateVerdict:
    """Teacher-forcing prefill vs prompt-prefill + stateful decode of completions.

    Tokens are the C2 fixture (no sampling). Each sample is scored at B=1 so
    prompt_len is exact and padding cannot leak into the decode cursor.
    """

    device = _device(model)
    train_parts: list[torch.Tensor] = []
    infer_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    for sample in batch.samples:
        tokens = torch.tensor([sample.token_ids], device=device, dtype=torch.long)
        attn = torch.ones_like(tokens, dtype=torch.bool)
        pos = torch.arange(sample.seq_len, device=device).unsqueeze(0)
        loss_mask = torch.tensor(
            [[int(i >= sample.prompt_len) for i in range(sample.seq_len)]],
            device=device,
            dtype=torch.bool,
        )
        train = model.selected_logprobs(
            tokens, attention_mask=attn, loss_mask=loss_mask, position_ids=pos
        )
        infer = torch.zeros_like(train)
        prompt = sample.prompt_len
        cache = model.allocate_cache(1, sample.seq_len, device)
        prefill = model.forward(
            tokens[:, :prompt],
            attention_mask=attn[:, :prompt],
            position_ids=pos[:, :prompt],
            kv_cache=cache,
        )
        # logits[prompt-1] predicts token[prompt] (first completion).
        first = model._selected_logp(
            _scoring_logits(prefill)[:, -1:, :], tokens[:, prompt : prompt + 1]
        )
        infer[:, prompt - 1] = first[:, 0]
        for phys in range(prompt, sample.seq_len - 1):
            step = model.decode_step(
                tokens[:, phys : phys + 1],
                cache,
                position_ids=pos[:, phys : phys + 1],
                attention_mask=attn[:, phys : phys + 1],
            )
            logp = model._selected_logp(_scoring_logits(step), tokens[:, phys + 1 : phys + 2])
            infer[:, phys] = logp[:, 0]
        active = loss_mask[:, 1:]
        train_parts.append(train.reshape(-1))
        infer_parts.append(infer.reshape(-1))
        mask_parts.append(active.reshape(-1))

    train_cat = torch.cat(train_parts)
    infer_cat = torch.cat(infer_parts)
    mask_cat = torch.cat(mask_parts)
    roles = resolve_comparison_roles(contract, _TRAIN_INFER_KIND)
    aggregates = compute_logprob_aggregates(
        infer_cat,
        train_cat,
        mask_cat,
        contract=contract,
        report_kind=_TRAIN_INFER_KIND,
        clip_interval=default_clip_interval(contract),
        comparison_lhs_role=roles.comparison_lhs_role,
        comparison_rhs_role=roles.comparison_rhs_role,
    )
    policy = resolve_dtype_policy(contract)
    return judge_logprob_aggregates(aggregates, contract, execution_dtype=policy.execution_dtype)


def _full_model_decode_sweep(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    contract: Mapping[str, Any],
    manifest: WS1Manifest,
) -> tuple[tuple[str, LogprobAggregateVerdict], ...]:
    """Full-model decode vs prefill on C2 short/long/varlen and padding axes."""

    cases: list[tuple[str, LogprobAggregateVerdict]] = []
    short = _fixture_sample(manifest, "short_full_model_fixture", sample_id="short")
    long = _fixture_sample(manifest, "long_full_model_fixture", sample_id="long")
    cases.append(
        (
            "decode-b1-short",
            _decode_prefill_batch(
                model,
                LogicalBatch(
                    workload_id=batch.workload_id,
                    seed=batch.seed,
                    samples=(short,),
                    cell_id="decode-b1-short",
                ),
                contract=contract,
                manifest=manifest,
                pad_side=None,
            ),
        )
    )
    cases.append(
        (
            "decode-b1-long",
            _decode_prefill_batch(
                model,
                LogicalBatch(
                    workload_id=batch.workload_id,
                    seed=batch.seed,
                    samples=(long,),
                    cell_id="decode-b1-long",
                ),
                contract=contract,
                manifest=manifest,
                pad_side=None,
            ),
        )
    )
    cases.append(
        (
            "decode-b1-primary-s3",
            _decode_prefill_batch(
                model,
                LogicalBatch(
                    workload_id=batch.workload_id,
                    seed=batch.seed,
                    samples=(batch.samples[-1],),
                    cell_id="decode-b1-primary-s3",
                ),
                contract=contract,
                manifest=manifest,
                pad_side=None,
            ),
        )
    )
    cases.append(
        (
            "decode-bn-varlen",
            _decode_prefill_batch(
                model, batch, contract=contract, manifest=manifest, pad_side="right"
            ),
        )
    )
    cases.append(
        (
            "decode-bn-padded-right",
            _decode_prefill_batch(
                model, batch, contract=contract, manifest=manifest, pad_side="right"
            ),
        )
    )
    cases.append(
        (
            "decode-bn-padded-left",
            _decode_prefill_batch(
                model, batch, contract=contract, manifest=manifest, pad_side="left"
            ),
        )
    )
    return tuple(cases)


def _decode_prefill_batch(
    model: Qwen3DenseBIModel,
    batch: LogicalBatch,
    *,
    contract: Mapping[str, Any],
    manifest: WS1Manifest,
    pad_side: str | None,
) -> LogprobAggregateVerdict:
    device = _device(model)
    if pad_side is None and len(batch.samples) == 1:
        sample = batch.samples[0]
        tokens = torch.tensor([sample.token_ids], device=device, dtype=torch.long)
        attn = torch.ones_like(tokens, dtype=torch.bool)
        pos = torch.arange(sample.seq_len, device=device).unsqueeze(0)
        loss_mask = torch.tensor(
            [[int(i >= sample.prompt_len) for i in range(sample.seq_len)]],
            device=device,
            dtype=torch.bool,
        )
    else:
        side = "right" if pad_side is None else pad_side
        padded = apply_padding(batch, pad_side=side, manifest=manifest)
        tokens = torch.tensor(padded.physical_token_ids, device=device, dtype=torch.long)
        attn = torch.tensor(padded.physical_attention_mask, device=device, dtype=torch.bool)
        pos = torch.tensor(padded.physical_position_ids, device=device, dtype=torch.long)
        loss_mask = torch.tensor(padded.physical_loss_mask, device=device, dtype=torch.bool)
    train = model.selected_logprobs(
        tokens, attention_mask=attn, loss_mask=loss_mask, position_ids=pos
    )
    infer = torch.zeros_like(train)
    cache = model.allocate_cache(tokens.shape[0], tokens.shape[1], device)
    first = model.forward(
        tokens[:, :1],
        attention_mask=attn[:, :1],
        position_ids=pos[:, :1],
        kv_cache=cache,
    )
    infer[:, 0] = model._selected_logp(_scoring_logits(first), tokens[:, 1:2])[:, 0]
    for phys in range(1, tokens.shape[1] - 1):
        step = model.decode_step(
            tokens[:, phys : phys + 1],
            cache,
            position_ids=pos[:, phys : phys + 1],
            attention_mask=attn[:, phys : phys + 1],
        )
        infer[:, phys] = model._selected_logp(
            _scoring_logits(step), tokens[:, phys + 1 : phys + 2]
        )[:, 0]
    active = loss_mask[:, 1:]
    roles = resolve_comparison_roles(contract, _TRAIN_INFER_KIND)
    aggregates = compute_logprob_aggregates(
        infer.reshape(-1),
        train.reshape(-1),
        active.reshape(-1),
        contract=contract,
        report_kind=_TRAIN_INFER_KIND,
        clip_interval=default_clip_interval(contract),
        comparison_lhs_role=roles.comparison_lhs_role,
        comparison_rhs_role=roles.comparison_rhs_role,
    )
    policy = resolve_dtype_policy(contract)
    return judge_logprob_aggregates(aggregates, contract, execution_dtype=policy.execution_dtype)


def _fixture_sample(manifest: WS1Manifest, key: str, *, sample_id: str) -> LogicalSample:
    raw = manifest.fixtures[key]
    return LogicalSample(
        sample_id=sample_id,
        token_ids=tuple(int(x) for x in raw["token_ids"]),
        prompt_len=int(raw["prompt_len"]),
        seq_len=int(raw["seq_len"]),
    )


def _release_parameter_grads(cell: CellOutput) -> None:
    """Release large CPU gradient snapshots after all comparisons finish."""
    for name, value in list(cell.grads.items()):
        cell.grads[name] = torch.empty(0, dtype=value.dtype)


def _collect_parameter_grads(model: Qwen3DenseBIModel, *, cell_id: str) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name in REQUIRED_GRAD_NAMES:
        tensor = model.weights.tensors[name]
        if tensor.grad is None:
            raise RuntimeError(f"required gradient {name!r} is missing in cell {cell_id!r}")
        # A Qwen3-8B gradient is several GiB even in BF16. Keep snapshots
        # off-device and in the candidate dtype; comparisons promote one
        # tensor at a time. Retaining FP32 GPU copies for every layout cell
        # makes the H100/H20 gate unable to fit.
        grads[name] = tensor.grad.detach().to(device="cpu")
    return grads


def _logp_aggregate_verdict(
    lhs: Mapping[tuple[str, int], torch.Tensor],
    rhs: Mapping[tuple[str, int], torch.Tensor],
    *,
    contract: Mapping[str, Any],
    report_kind: str,
) -> LogprobAggregateVerdict:
    a, b, mask = _aligned_logp_vectors(lhs, rhs)
    roles = resolve_comparison_roles(contract, report_kind)
    policy = resolve_dtype_policy(contract)
    return judge_logprob_aggregates(
        compute_logprob_aggregates(
            a,
            b,
            mask,
            contract=contract,
            report_kind=report_kind,
            clip_interval=default_clip_interval(contract),
            comparison_lhs_role=roles.comparison_lhs_role,
            comparison_rhs_role=roles.comparison_rhs_role,
        ),
        contract,
        execution_dtype=policy.execution_dtype,
    )


def _gpu_name(device: torch.device) -> str | None:
    """Accelerator name for evidence: the GPU on CUDA, the NPU on Ascend."""

    if device.type not in ACCELERATOR_TYPES or not is_available(device.type):
        return None
    return device_name(device)


def _workflow_url() -> str | None:
    explicit = os.environ.get("WS1_WORKFLOW_URL")
    if explicit:
        return explicit
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return None


def _c8_evidence_path() -> str:
    return os.environ.get("WS1_C8_EVIDENCE_PATH", "docs/design/ws1-c8-execute.json")


def _c8_source_commit() -> str | None:
    path = Path(_c8_evidence_path())
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    git_meta = payload.get("git", {})
    commit = git_meta.get("commit")
    return str(commit) if commit else None


def _representative_case_ids(manifest: WS1Manifest, backend_profile: str) -> tuple[str, ...]:
    ids: list[str] = []
    for case in manifest.representative_cases:
        profiles = case.get("profile_ids") or ()
        if backend_profile in profiles:
            ids.append(str(case["case_id"]))
    return tuple(ids)


def _compare_required_grads(
    lhs: CellOutput,
    rhs: CellOutput,
    *,
    contract: Mapping[str, Any],
    dtype: str,
    backend_profile: str,
    judgment: str = "gradient_invariance",
    config_pair: tuple[str, str] | None = None,
) -> list[TensorComparisonDetail]:
    pair = config_pair or (lhs.cell_id, rhs.cell_id)
    details: list[TensorComparisonDetail] = []
    for name in REQUIRED_GRAD_NAMES:
        if name not in lhs.grads or name not in rhs.grads:
            spec = resolve_tolerance(
                contract,
                judgment=judgment,
                op_class="reduction",
                dtype=dtype,
                backend_profile=backend_profile,
            )
            details.append(
                TensorComparisonDetail(
                    tensor_name=name,
                    config_pair=pair,
                    shape=(0,),
                    dtype=dtype,
                    max_abs_error=float("inf"),
                    mean_abs_error=float("inf"),
                    max_rel_error=float("inf"),
                    atol=spec.atol,
                    rtol=spec.rtol,
                    passed=False,
                    judgment=judgment,
                    comparison_lhs_role="transformed_config",
                    comparison_rhs_role=(
                        "fp32_reference" if judgment == "gradient_accuracy" else "canonical_config"
                    ),
                )
            )
            continue
        details.append(
            _compare_logical_tensors(
                lhs.grads[name],
                rhs.grads[name],
                judgment=judgment,
                contract=contract,
                op_class="reduction",
                dtype=dtype,
                backend_profile=backend_profile,
                tensor_name=name,
                config_pair=pair,
            )
        )
    return details


def _compare_logp_maps(
    lhs: Mapping[tuple[str, int], torch.Tensor],
    rhs: Mapping[tuple[str, int], torch.Tensor],
    *,
    contract: Mapping[str, Any],
    judgment: str,
    dtype: str,
    backend_profile: str,
    config_pair: tuple[str, str],
) -> TensorComparisonDetail:
    keys = sorted(set(lhs) | set(rhs))
    if not keys or set(lhs) != set(rhs):
        spec = resolve_tolerance(
            contract,
            judgment=judgment,
            op_class="logprob",
            dtype=dtype,
            backend_profile=backend_profile,
        )
        return TensorComparisonDetail(
            tensor_name="selected_logp",
            config_pair=config_pair,
            shape=(len(keys),),
            dtype=dtype,
            max_abs_error=float("inf"),
            mean_abs_error=float("inf"),
            max_rel_error=float("inf"),
            atol=spec.atol,
            rtol=spec.rtol,
            passed=False,
            judgment=judgment,
            comparison_lhs_role="transformed_config",
            comparison_rhs_role="canonical_config",
        )
    a = torch.stack([lhs[key].float().reshape(()) for key in keys])
    b = torch.stack([rhs[key].float().reshape(()) for key in keys])
    return _compare_logical_tensors(
        a,
        b,
        judgment=judgment,
        contract=contract,
        op_class="logprob",
        dtype=dtype,
        backend_profile=backend_profile,
        tensor_name="selected_logp",
        config_pair=config_pair,
    )


def _aligned_logp_vectors(
    lhs: Mapping[tuple[str, int], torch.Tensor],
    rhs: Mapping[tuple[str, int], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = sorted(set(lhs) & set(rhs))
    if not keys:
        raise RuntimeError("no overlapping logical tokens to compare")
    a = torch.stack([lhs[key].float().reshape(()) for key in keys])
    b = torch.stack([rhs[key].float().reshape(()) for key in keys])
    mask = torch.ones_like(a, dtype=torch.bool)
    return a, b, mask


def _device(model: Qwen3DenseBIModel) -> torch.device:
    return next(iter(model.weights.tensors.values())).device


def _configure_required_gradients(model: Qwen3DenseBIModel, *, enabled: bool) -> None:
    missing = sorted(set(REQUIRED_GRAD_NAMES) - set(model.weights.tensors))
    if missing:
        raise RuntimeError(f"model is missing required gradient tensors: {missing}")
    for name, tensor in model.weights.tensors.items():
        if tensor.is_floating_point():
            tensor.requires_grad_(enabled and name in REQUIRED_GRAD_NAMES)
            tensor.grad = None


def _validate_required_gradient_cells(cells: Mapping[str, CellOutput]) -> None:
    required = set(REQUIRED_GRAD_NAMES)
    for cell_id in PRIMARY_CELLS:
        observed = set(cells[cell_id].grads)
        if observed != required:
            raise RuntimeError(
                f"cell {cell_id!r} gradient set {sorted(observed)} != required {sorted(required)}"
            )


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _node_token_fingerprints(
    model: Qwen3DenseBIModel,
    restore: Sequence[Sequence[tuple[str, int] | None]],
) -> dict[str, dict[str, str]]:
    """Hash every logical token at every captured node for exact first-drift."""

    result: dict[str, dict[str, str]] = {}
    batch = len(restore)
    seq = len(restore[0]) if restore else 0
    for node, value in model.captured_node_outputs().items():
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            continue
        if value.dim() < 2 or value.shape[0] != batch:
            continue
        if value.shape[1] == seq:
            token_axis = 1
        elif value.dim() >= 3 and value.shape[2] == seq:
            token_axis = 2
        else:
            continue
        cpu_value = value.detach().contiguous().cpu()
        node_values: dict[str, str] = {}
        for batch_index, row in enumerate(restore):
            for physical_index, logical_key in enumerate(row):
                if logical_key is None:
                    continue
                if token_axis == 1:
                    token_value = cpu_value[batch_index, physical_index]
                else:
                    token_value = cpu_value[batch_index, :, physical_index]
                key = f"{logical_key[0]}:{logical_key[1]}"
                node_values[key] = _tensor_digest(token_value)
        if node_values:
            result[node] = node_values
    return result


def _merge_node_token_digests(
    target: dict[str, dict[str, str]], source: Mapping[str, Mapping[str, str]]
) -> None:
    for node, values in source.items():
        destination = target.setdefault(node, {})
        overlap = set(destination) & set(values)
        for key in overlap:
            if destination[key] != values[key]:
                raise RuntimeError(f"node fingerprint collision for {node!r} logical token {key!r}")
        destination.update(values)


def _combined_node_digests(
    values: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    combined: dict[str, str] = {}
    for node, tokens in values.items():
        digest = hashlib.sha256()
        for key, value in sorted(tokens.items()):
            digest.update(f"{key}\t{value}\n".encode("utf-8"))
        combined[node] = digest.hexdigest()
    return combined


def _backend_family(value: str) -> str:
    if value.startswith("cuda"):
        return "cuda"
    return value


def _locate_first_drift(
    model: Qwen3DenseBIModel,
    cells: Mapping[str, CellOutput],
    inv_details: Sequence[TensorComparisonDetail],
    grad_details: Sequence[TensorComparisonDetail],
    train_infer: LogprobAggregateVerdict | None,
    train_infer_bn: LogprobAggregateVerdict | None = None,
    decode_prefill: Sequence[tuple[str, LogprobAggregateVerdict]] = (),
) -> str | None:
    """First failing layer/op, then tensor/config pair. Used by C10/C11 reports."""

    canonical = cells.get("BN/full")
    for detail in list(inv_details) + list(grad_details):
        if detail.passed:
            continue
        other_id = detail.config_pair[1]
        other = cells.get(other_id)
        if canonical is not None and other is not None:
            for node in model.node_names():
                left = canonical.node_token_digests.get(node)
                right = other.node_token_digests.get(node)
                if left is None or right is None:
                    continue
                if left != right:
                    return f"{node}:{detail.config_pair[0]}->{other_id}"
        return f"{detail.tensor_name}:{detail.config_pair[0]}->{other_id}"
    if train_infer is not None and not train_infer.passed:
        return "train_infer_selected_logp"
    if train_infer_bn is not None and not train_infer_bn.passed:
        return "train_infer_bn_selected_logp"
    for case_id, verdict in decode_prefill:
        if not verdict.passed:
            return f"decode_prefill:{case_id}"
    return None


__all__ = [
    "GRADIENT_SCOPE",
    "LAYOUT_CELLS",
    "PRIMARY_CELLS",
    "REQUIRED_GRAD_NAMES",
    "CellOutput",
    "ChainGateReport",
    "build_model",
    "run_chain_gate",
    "run_fp32_reference_cell",
]
