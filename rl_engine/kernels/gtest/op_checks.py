# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rl_engine.kernels.gtest.tolerance import load_contract


@dataclass(frozen=True)
class OperatorCase:
    """One deterministic test object for an operator candidate."""

    name: str
    op_class: str
    dtype: torch.dtype
    inputs: Mapping[str, Any]
    gold_fn: Callable[..., Any]


@dataclass(frozen=True)
class CandidateSpec:
    """One implementation to validate against the gold path."""

    name: str
    fn: Callable[..., Any] | Any
    backend: str = "unknown"
    arch_key: str | None = None


@dataclass(frozen=True)
class OutputCheck:
    """Per-output comparison result."""

    output_index: int
    shape: tuple[int, ...]
    candidate_dtype: str
    gold_dtype: str
    atol: float
    rtol: float
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    passed: bool
    message: str = ""


@dataclass(frozen=True)
class CaseCheck:
    """Per-case result for one candidate."""

    case_name: str
    dtype: str
    op_class: str
    passed: bool
    outputs: list[OutputCheck]


@dataclass(frozen=True)
class CandidateReport:
    """Aggregate report for one candidate implementation."""

    candidate_name: str
    backend: str
    total_outputs: int
    passed_outputs: int
    pass_rate: float
    passed: bool
    cases: list[CaseCheck]


@dataclass(frozen=True)
class OperatorCheckReport:
    """Suite-level report across candidates."""

    suite_name: str
    total_candidates: int
    passed_candidates: int
    pass_rate: float
    passed: bool
    candidates: list[CandidateReport]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_operator_suite(
    suite_name: str,
    *,
    candidates: Sequence[CandidateSpec],
    cases: Sequence[OperatorCase],
    contract: Mapping[str, Any] | None = None,
) -> OperatorCheckReport:
    """Run candidates against gold outputs and return a structured report."""

    loaded_contract = dict(contract or load_contract())
    # run all test ops
    # cases : test object
    # camdidate : test instance
    # loaded_contract : tolerance table
    candidate_reports = [
        _run_candidate(candidate, cases, loaded_contract) for candidate in candidates
    ]
    passed_candidates = sum(1 for report in candidate_reports if report.passed)
    total_candidates = len(candidate_reports)
    pass_rate = float(passed_candidates / total_candidates) if total_candidates else 0.0
    return OperatorCheckReport(
        suite_name=suite_name,
        total_candidates=total_candidates,
        passed_candidates=passed_candidates,
        pass_rate=pass_rate,
        passed=passed_candidates == total_candidates,
        candidates=candidate_reports,
    )


def _run_candidate(
    candidate: CandidateSpec,
    cases: Sequence[OperatorCase],
    contract: Mapping[str, Any],
) -> CandidateReport:
    case_checks = [_run_case(candidate, case, contract) for case in cases]
    total_outputs = sum(len(case.outputs) for case in case_checks)
    passed_outputs = sum(
        1 for case in case_checks for output in case.outputs if output.passed
    )
    pass_rate = float(passed_outputs / total_outputs) if total_outputs else 0.0
    return CandidateReport(
        candidate_name=candidate.name,
        backend=candidate.backend,
        total_outputs=total_outputs,
        passed_outputs=passed_outputs,
        pass_rate=pass_rate,
        passed=passed_outputs == total_outputs,
        cases=case_checks,
    )


def _run_case(
    candidate: CandidateSpec,
    case: OperatorCase,
    contract: Mapping[str, Any],
) -> CaseCheck:
    candidate_outputs = _flatten_tensors(_call_candidate(candidate.fn, case.inputs))
    gold_outputs = _flatten_tensors(case.gold_fn(**case.inputs))
    if len(candidate_outputs) != len(gold_outputs):
        raise ValueError(
            f"candidate {candidate.name!r} returned {len(candidate_outputs)} outputs, "
            f"gold returned {len(gold_outputs)}"
        )
    atol, rtol = _resolve_tolerance(
        contract,
        op_class=case.op_class,
        dtype=case.dtype,
        arch_key=candidate.arch_key,
    )
    output_checks = [
        _compare_output(
            candidate_output,
            gold_output,
            output_index=index,
            atol=atol,
            rtol=rtol,
        )
        for index, (candidate_output, gold_output) in enumerate(
            zip(candidate_outputs, gold_outputs, strict=True)
        )
    ]
    return CaseCheck(
        case_name=case.name,
        dtype=str(case.dtype),
        op_class=case.op_class,
        passed=all(output.passed for output in output_checks),
        outputs=output_checks,
    )


# compatibility function or forward
def _call_candidate(candidate: Callable[..., Any] | Any, inputs: Mapping[str, Any]) -> Any:
    if hasattr(candidate, "forward") and callable(candidate.forward):
        return candidate.forward(**inputs)
    return candidate(**inputs)


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        outputs: list[torch.Tensor] = []
        for item in value:
            outputs.extend(_flatten_tensors(item))
        return outputs
    raise TypeError(f"operator output must be Tensor or sequence, got {type(value)!r}")


def _resolve_tolerance(
    contract: Mapping[str, Any],
    *,
    op_class: str,
    dtype: torch.dtype,
    arch_key: str | None = None,
) -> tuple[float, float]:
    dtype_name = _dtype_name(dtype)
    if arch_key is not None:
        arch_values = (
            contract["accuracy"]
            .get("arch_overrides", {})
            .get(arch_key, {})
            .get(op_class, {})
            .get(dtype_name)
        )
        if arch_values is not None:
            return float(arch_values["atol"]), float(arch_values.get("rtol", 0.0))

    values = contract["accuracy"]["default"][op_class][dtype_name]
    return float(values["atol"]), float(values.get("rtol", 0.0))


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float32:
        return "float32"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    raise ValueError(f"unsupported dtype: {dtype}")


def _compare_output(
    candidate: torch.Tensor,
    gold: torch.Tensor,
    *,
    output_index: int,
    atol: float,
    rtol: float,
) -> OutputCheck:
    if candidate.shape != gold.shape:
        return OutputCheck(
            output_index=output_index,
            shape=tuple(candidate.shape),
            candidate_dtype=str(candidate.dtype),
            gold_dtype=str(gold.dtype),
            atol=atol,
            rtol=rtol,
            max_abs_error=float("inf"),
            mean_abs_error=float("inf"),
            max_rel_error=float("inf"),
            passed=False,
            message=f"shape mismatch: candidate={tuple(candidate.shape)} gold={tuple(gold.shape)}",
        )

    candidate_fp32 = candidate.float()
    gold_fp32 = gold.float()
    abs_error = (candidate_fp32 - gold_fp32).abs()
    if abs_error.numel() == 0:
        max_abs_error = 0.0
        mean_abs_error = 0.0
        max_rel_error = 0.0
    else:
        max_abs_error = float(abs_error.max().item())
        mean_abs_error = float(abs_error.mean().item())
        rel_error = abs_error / gold_fp32.abs().clamp_min(1e-12)
        max_rel_error = float(rel_error.max().item())

    return OutputCheck(
        output_index=output_index,
        shape=tuple(candidate.shape),
        candidate_dtype=str(candidate.dtype),
        gold_dtype=str(gold.dtype),
        atol=atol,
        rtol=rtol,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        max_rel_error=max_rel_error,
        passed=bool(torch.allclose(candidate_fp32, gold_fp32, atol=atol, rtol=rtol)),
    )


__all__ = [
    "CandidateReport",
    "CandidateSpec",
    "CaseCheck",
    "OperatorCase",
    "OperatorCheckReport",
    "OutputCheck",
    "run_operator_suite",
]
