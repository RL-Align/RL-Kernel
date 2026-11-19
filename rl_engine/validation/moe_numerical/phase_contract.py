# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Compare supplied phase outputs against rules and two independent references.

The caller captures actual implementations. This tool does not execute a model,
attest producer identity, or promote a CPU check to Ascend qualification.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from rl_engine.contracts.operators.moe_numerical import (
    MOE_PHASES,
    MoECombineInputs,
    MoEContractError,
    MoENumericalContract,
    moe_rule_manifest,
)
from rl_engine.reference.moe_numerical import combine_accuracy_reference, combine_fixed_tree


@dataclass(frozen=True)
class MoEPhaseOutput:
    phase: str
    output: torch.Tensor
    contract_fingerprint: str
    mapping_run_id: str
    parameter_version: str
    actual_backend: str
    implementation_fingerprint: str

    def __post_init__(self) -> None:
        if self.phase not in MOE_PHASES:
            raise MoEContractError(f"phase must be one of {MOE_PHASES}")
        for name in (
            "contract_fingerprint",
            "mapping_run_id",
            "parameter_version",
            "actual_backend",
            "implementation_fingerprint",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MoEContractError(f"{name} must be explicit and non-empty")


def _bit_mismatches(expected: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    shape = (*expected.shape, expected.element_size())
    lhs = expected.resolve_neg().contiguous().view(torch.uint8).reshape(shape)
    rhs = actual.resolve_neg().contiguous().view(torch.uint8).reshape(shape)
    return (lhs != rhs).any(dim=-1)


def compare_moe_phases(
    contract: MoENumericalContract,
    inputs: MoECombineInputs,
    observations: Sequence[MoEPhaseOutput],
    *,
    expected_mapping_run_id: str,
    expected_parameter_version: str,
) -> dict:
    """Require all four phases for one logical sample, then check actual tensors.

    Fingerprints/run/version mismatches fail closed before arithmetic comparison.
    Output mismatches have independent accuracy and fixed-tree verdicts. Shared
    WS1 reduction thresholds are reused without private relaxation; FP64 is an
    extra accuracy oracle here, not a replacement for the shared FP32 gold.
    Empty runtime work is valid in the reference but is not acceptance evidence.
    """
    if not isinstance(expected_parameter_version, str) or not expected_parameter_version.strip():
        raise MoEContractError("expected_parameter_version must be explicit and non-empty")
    by_phase = {record.phase: record for record in observations}
    if len(observations) != len(MOE_PHASES) or set(by_phase) != set(MOE_PHASES):
        raise MoEContractError("provide exactly one output for train/prefill/decode/verify")
    if inputs.num_tokens == 0:
        raise MoEContractError("empty samples cannot qualify phase consistency")
    for record in observations:
        if record.contract_fingerprint != contract.fingerprint:
            raise MoEContractError("phase contract_fingerprint mismatch")
        if record.mapping_run_id != expected_mapping_run_id:
            raise MoEContractError("phase mapping_run_id mismatch")
        if record.parameter_version != expected_parameter_version:
            raise MoEContractError("phase parameter_version mismatch; recapture after updates")
    fixed = combine_fixed_tree(contract, inputs, expected_mapping_run_id=expected_mapping_run_id)
    accuracy = combine_accuracy_reference(
        contract, inputs, expected_mapping_run_id=expected_mapping_run_id
    )
    manifest = moe_rule_manifest(contract)
    tolerance = manifest["rules"]["moe_combine"]["forward_accuracy"]
    results = {}
    for phase in MOE_PHASES:
        record = by_phase[phase]
        actual = record.output
        metadata_errors = []
        if not isinstance(actual, torch.Tensor):
            metadata_errors.append("non_tensor_output")
            actual_device = None
            actual_shape = None
            actual_dtype = None
        elif getattr(actual, "is_nested", False):
            metadata_errors.append("nested_tensor_output")
            actual_device = str(actual.device)
            actual_shape = None
            actual_dtype = str(actual.dtype)
        elif actual.layout != torch.strided or actual.device.type == "meta":
            metadata_errors.append("non_materialized_output")
            actual_device = str(actual.device)
            actual_shape = list(actual.shape)
            actual_dtype = str(actual.dtype)
        else:
            actual_device = str(actual.device)
            actual_shape = list(actual.shape)
            actual_dtype = str(actual.dtype)
        if (
            isinstance(actual, torch.Tensor)
            and not getattr(actual, "is_nested", False)
            and actual.layout == torch.strided
            and actual.device.type != "meta"
        ):
            if actual.shape != fixed.shape:
                metadata_errors.append("shape_mismatch")
            if actual.dtype != fixed.dtype:
                metadata_errors.append("dtype_mismatch")
        result: dict[str, Any] = {
            "actual_backend": record.actual_backend,
            "implementation_fingerprint": record.implementation_fingerprint,
            "output_device": actual_device,
            "output_shape": actual_shape,
            "output_dtype": actual_dtype,
            "producer_identity_verified": False,
            "metadata_errors": metadata_errors,
            "compared_count": 0,
            "bit_mismatch_count": 0,
            "accuracy_mismatch_count": 0,
            "fixed_tree_passed": False,
            "accuracy_passed": False,
            "first_bit_mismatch": None,
            "max_absolute_error": None,
        }
        if not metadata_errors:
            cpu = actual.detach().cpu()
            bits = _bit_mismatches(fixed, cpu)
            delta = (cpu.double() - accuracy).abs()
            finite = torch.isfinite(cpu)
            inaccurate = ~finite | (delta > tolerance["atol"] + tolerance["rtol"] * accuracy.abs())
            first = torch.nonzero(bits, as_tuple=False)
            result.update(
                compared_count=fixed.numel(),
                bit_mismatch_count=int(bits.sum()),
                accuracy_mismatch_count=int(inaccurate.sum()),
                fixed_tree_passed=not bool(bits.any()),
                accuracy_passed=not bool(inaccurate.any()),
                first_bit_mismatch=first[0].tolist() if first.numel() else None,
                max_absolute_error=float(delta.max()) if bool(finite.all()) else None,
            )
        result["passed"] = result["fixed_tree_passed"] and result["accuracy_passed"]
        results[phase] = result
    forward_invariance_passed = all(result["fixed_tree_passed"] for result in results.values())
    forward_accuracy_passed = all(result["accuracy_passed"] for result in results.values())
    return {
        "schema_version": "moe.phase_combine_report.v1",
        "evidence_scope": "supplied_combine_outputs",
        "forward_invariance_passed": forward_invariance_passed,
        "forward_accuracy_passed": forward_accuracy_passed,
        "passed_policy": "forward_invariance_and_accuracy",
        "passed": forward_invariance_passed and forward_accuracy_passed,
        "device_execution_verified": False,
        "mapping_run_id": expected_mapping_run_id,
        "mapping_run_id_verified": True,
        "parameter_version": expected_parameter_version,
        "parameter_version_matched": True,
        "parameter_content_verified": False,
        "expert_weight_applied": contract.expert_weight_applied,
        "weight_application_performed": not contract.expert_weight_applied,
        "weight_ownership_verified": False,
        "num_tokens": inputs.num_tokens,
        "rule_manifest": manifest,
        "accuracy_reference": "cpu_fp64_math_fsum",
        "fixed_order_reference": "cpu_fp32_adjacent_pairs_odd_carry",
        "routing_comparison": "not_performed",
        "phases": results,
    }
