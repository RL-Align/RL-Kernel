# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Optional

import torch

from rl_engine.kernels.gtest.tolerance import (
    BackendProvenance,
    ContractResolveError,
    load_contract,
    normalize_dtype_name,
    resolve_tolerance,
    validate_backend_provenance,
)


@dataclass(frozen=True)
class OperatorCase:
    """One deterministic test object for an operator candidate."""

    name: str
    op_class: str
    dtype: torch.dtype
    inputs: Mapping[str, Any]
    gold_fn: Callable[..., Any]
    grad_input_names: tuple[str, ...] = ()
    # Bitwise-strict operators (WS1 frozen-contract tree ops): the pass
    # criterion is exact logical-element bit-pattern equality. Forward legs
    # compare against the gold cast ONCE to the candidate's output dtype;
    # the gradient leg differentiates ``gold_backward_fn`` -- the gold OP
    # itself, whose autograd.Function.backward IS the frozen chain -- with
    # the same upstream bits as the candidate. Tolerances are never
    # consulted for ``passed``; error statistics stay diagnostics.
    bitwise_strict: bool = False
    gold_backward_fn: Optional[Callable[..., Any]] = None


@dataclass(frozen=True)
class CandidateSpec:
    """One implementation to validate against the gold path."""

    name: str
    fn: Callable[..., Any] | Any
    backend: str = "unknown"
    arch_key: str | None = None
    provenance: BackendProvenance | None = None


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
    judgment: str
    comparison_lhs_role: str
    comparison_rhs_role: str
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
    backend_provenance: BackendProvenance | None = None


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
    check_grad: bool = False,
    grad_mode: str = "random",
    grad_seed: int = 123,
) -> OperatorCheckReport:
    """Run candidates against gold outputs and return a structured report."""

    loaded_contract = dict(contract or load_contract())
    # run all test ops
    # cases : test object
    # camdidate : test instance
    # loaded_contract : tolerance table
    candidate_reports = [
        _run_candidate(
            candidate,
            cases,
            loaded_contract,
            check_grad=check_grad,
            grad_mode=grad_mode,
            grad_seed=grad_seed,
        )
        for candidate in candidates
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
    *,
    check_grad: bool,
    grad_mode: str,
    grad_seed: int,
) -> CandidateReport:
    if candidate.provenance is not None:
        validate_backend_provenance(contract, candidate.provenance)
        if candidate.backend != candidate.provenance.actual_backend:
            raise ContractResolveError(
                f"candidate backend {candidate.backend!r} disagrees with reported actual_backend "
                f"{candidate.provenance.actual_backend!r}"
            )
        for case in cases:
            case_dtype = normalize_dtype_name(case.dtype)
            provenance_dtype = normalize_dtype_name(candidate.provenance.execution_dtype)
            if case_dtype != provenance_dtype:
                raise ContractResolveError(
                    f"case {case.name!r} dtype {case.dtype} does not match "
                    f"provenance execution_dtype {candidate.provenance.execution_dtype!r}"
                )
    if check_grad:
        case_checks = [
            _run_case_backward(
                candidate,
                case,
                contract,
                grad_mode=grad_mode,
                grad_seed=grad_seed,
            )
            for case in cases
        ]
    else:
        case_checks = [_run_case(candidate, case, contract) for case in cases]
    total_outputs = sum(len(case.outputs) for case in case_checks)
    passed_outputs = sum(1 for case in case_checks for output in case.outputs if output.passed)
    pass_rate = float(passed_outputs / total_outputs) if total_outputs else 0.0
    return CandidateReport(
        candidate_name=candidate.name,
        backend=candidate.backend,
        total_outputs=total_outputs,
        passed_outputs=passed_outputs,
        pass_rate=pass_rate,
        passed=passed_outputs == total_outputs,
        cases=case_checks,
        backend_provenance=candidate.provenance,
    )


def _run_case(
    candidate: CandidateSpec,
    case: OperatorCase,
    contract: Mapping[str, Any],
) -> CaseCheck:
    candidate_outputs = _flatten_tensors(_call_candidate(candidate.fn, case.inputs))
    gold_outputs = _flatten_tensors(case.gold_fn(**case.inputs))
    return _compare_case_outputs(candidate, case, contract, candidate_outputs, gold_outputs)


def _run_case_backward(
    candidate: CandidateSpec,
    case: OperatorCase,
    contract: Mapping[str, Any],
    *,
    grad_mode: str,
    grad_seed: int,
) -> CaseCheck:
    if not case.grad_input_names:
        raise ValueError(f"case {case.name!r} does not declare gradient inputs")

    candidate_inputs = _clone_inputs_for_backward(case.inputs, case.grad_input_names)
    gold_inputs = _clone_inputs_for_backward(case.inputs, case.grad_input_names)
    candidate_outputs = _flatten_tensors(_call_candidate(candidate.fn, candidate_inputs))
    # Frozen-backward gold: differentiate the gold OP itself (its
    # autograd.Function.backward is the frozen chain) instead of the plain
    # gold_fn graph, so the gradient gold shares the contract's rounding
    # positions and bitwise equality is meaningful. The FORWARD comparison
    # still uses the fp32 gold_fn view (identical bits after the single
    # output cast), keeping provenance dtype checks unchanged.
    gold_forward = case.gold_backward_fn if case.gold_backward_fn is not None else case.gold_fn
    gold_graph_outputs = _flatten_tensors(gold_forward(**gold_inputs))
    gold_outputs = (
        _flatten_tensors(case.gold_fn(**gold_inputs))
        if case.gold_backward_fn is not None
        else gold_graph_outputs
    )
    # Candidate and gold must use the same upstream gradients; otherwise we
    # would compare different vector-Jacobian products.
    # grad_mode="ones" is the old output.sum().backward() smoke path.
    # grad_mode="random" is closer to training, where dL/doutput is non-uniform.
    grad_outputs = _make_grad_outputs(candidate_outputs, grad_mode=grad_mode, seed=grad_seed)
    shared_upstreams = [
        grad.to(device=output.device, dtype=output.dtype)
        for grad, output in zip(grad_outputs, candidate_outputs, strict=True)
    ]
    candidate_grads = _backward_grads(
        candidate_outputs,
        candidate_inputs,
        case.grad_input_names,
        grad_outputs=shared_upstreams,
    )
    gold_grads = _backward_grads(
        gold_graph_outputs,
        gold_inputs,
        case.grad_input_names,
        grad_outputs=_match_grad_outputs(shared_upstreams, gold_graph_outputs),
    )
    output_checks = _compare_case_outputs(
        candidate,
        case,
        contract,
        candidate_outputs,
        gold_outputs,
    ).outputs
    # Gradient thresholds come from the independent gradient_accuracy judgment
    # (#267); they must not silently inherit forward_accuracy rows.
    if "judgments" in contract:
        gradient_spec = resolve_tolerance(
            contract,
            judgment="gradient_accuracy",
            op_class=case.op_class,
            dtype=case.dtype,
            arch_key=candidate.arch_key,
            backend_profile=(
                candidate.provenance.backend_profile if candidate.provenance else None
            ),
        )
        atol, rtol = gradient_spec.atol, gradient_spec.rtol
    else:
        gradient_spec = None
        atol, rtol = _resolve_tolerance(
            contract,
            op_class=case.op_class,
            dtype=case.dtype,
            arch_key=candidate.arch_key,
            backend_profile=(
                candidate.provenance.backend_profile if candidate.provenance else None
            ),
            judgment="gradient_accuracy",
        )
    grad_checks = [
        _compare_output(
            candidate_grad,
            gold_grad,
            output_index=len(output_checks) + index,
            atol=atol,
            rtol=rtol,
            judgment="gradient_accuracy",
            comparison_lhs_role=(
                gradient_spec.comparison_lhs_role if gradient_spec is not None else "bf16_candidate"
            ),
            comparison_rhs_role=(
                gradient_spec.comparison_rhs_role if gradient_spec is not None else "fp32_reference"
            ),
            message=f"gradient:{name}",
        )
        for index, (name, candidate_grad, gold_grad) in enumerate(
            zip(case.grad_input_names, candidate_grads, gold_grads, strict=True)
        )
    ]
    if case.bitwise_strict:
        grad_checks = [
            _bitwise_verdict(check, candidate_grad, gold_grad)
            for check, candidate_grad, gold_grad in zip(
                grad_checks, candidate_grads, gold_grads, strict=True
            )
        ]
    checks = [*output_checks, *grad_checks]
    return CaseCheck(
        case_name=case.name,
        dtype=str(case.dtype),
        op_class=case.op_class,
        passed=all(output.passed for output in checks),
        outputs=checks,
    )


def _compare_case_outputs(
    candidate: CandidateSpec,
    case: OperatorCase,
    contract: Mapping[str, Any],
    candidate_outputs: list[torch.Tensor],
    gold_outputs: list[torch.Tensor],
) -> CaseCheck:
    if len(candidate_outputs) != len(gold_outputs):
        raise ValueError(
            f"candidate {candidate.name!r} returned {len(candidate_outputs)} outputs, "
            f"gold returned {len(gold_outputs)}"
        )
    if "judgments" in contract:
        forward_spec = resolve_tolerance(
            contract,
            judgment="forward_accuracy",
            op_class=case.op_class,
            dtype=case.dtype,
            arch_key=candidate.arch_key,
            backend_profile=(
                candidate.provenance.backend_profile if candidate.provenance else None
            ),
        )
        atol, rtol = forward_spec.atol, forward_spec.rtol
    else:
        forward_spec = None
        atol, rtol = _resolve_tolerance(
            contract,
            op_class=case.op_class,
            dtype=case.dtype,
            arch_key=candidate.arch_key,
            backend_profile=(
                candidate.provenance.backend_profile if candidate.provenance else None
            ),
            judgment="forward_accuracy",
        )
    if candidate.provenance is not None:
        for candidate_output, gold_output in zip(candidate_outputs, gold_outputs, strict=True):
            candidate_dtype = normalize_dtype_name(candidate_output.dtype)
            gold_dtype = normalize_dtype_name(gold_output.dtype)
            provenance_output_dtype = normalize_dtype_name(candidate.provenance.output_dtype)
            provenance_reference_dtype = normalize_dtype_name(candidate.provenance.reference_dtype)
            if candidate_dtype != provenance_output_dtype:
                raise ContractResolveError(
                    f"candidate output dtype {candidate_dtype!r} disagrees with provenance "
                    f"output_dtype {candidate.provenance.output_dtype!r}"
                )
            if gold_dtype != provenance_reference_dtype:
                raise ContractResolveError(
                    f"gold output dtype {gold_dtype!r} disagrees with provenance "
                    f"reference_dtype {candidate.provenance.reference_dtype!r}"
                )
    output_checks = [
        _compare_output(
            candidate_output,
            gold_output,
            output_index=index,
            atol=atol,
            rtol=rtol,
            judgment="forward_accuracy",
            comparison_lhs_role=(
                forward_spec.comparison_lhs_role if forward_spec is not None else "bf16_candidate"
            ),
            comparison_rhs_role=(
                forward_spec.comparison_rhs_role if forward_spec is not None else "fp32_reference"
            ),
        )
        for index, (candidate_output, gold_output) in enumerate(
            zip(candidate_outputs, gold_outputs, strict=True)
        )
    ]
    if case.bitwise_strict:
        output_checks = [
            _bitwise_verdict(check, candidate_output, gold_output)
            for check, candidate_output, gold_output in zip(
                output_checks, candidate_outputs, gold_outputs, strict=True
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


def _clone_inputs_for_backward(
    inputs: Mapping[str, Any],
    grad_input_names: tuple[str, ...],
) -> dict[str, Any]:
    grad_names = set(grad_input_names)
    cloned: dict[str, Any] = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone()
            if name in grad_names:
                if not tensor.is_floating_point():
                    raise TypeError(f"gradient input {name!r} must be floating point")
                tensor.requires_grad_(True)
            cloned[name] = tensor
        else:
            cloned[name] = value
    missing = grad_names.difference(cloned)
    if missing:
        raise ValueError(f"missing gradient inputs: {', '.join(sorted(missing))}")
    return cloned


def _backward_grads(
    outputs: list[torch.Tensor],
    inputs: Mapping[str, Any],
    grad_input_names: tuple[str, ...],
    *,
    grad_outputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    if len(outputs) != len(grad_outputs):
        raise ValueError(f"got {len(grad_outputs)} upstream gradients for {len(outputs)} outputs")
    tensors = [inputs[name] for name in grad_input_names]
    grads = torch.autograd.grad(
        outputs,
        tensors,
        grad_outputs=[
            grad_output.to(device=output.device, dtype=output.dtype)
            for output, grad_output in zip(outputs, grad_outputs, strict=True)
        ],
        allow_unused=False,
    )
    return list(grads)


def _make_grad_outputs(
    outputs: list[torch.Tensor],
    *,
    grad_mode: str,
    seed: int,
) -> list[torch.Tensor]:
    if grad_mode == "ones":
        # All-one upstream gradients make the scalar loss equal output.sum().
        return [torch.ones_like(output, dtype=torch.float32) for output in outputs]
    if grad_mode != "random":
        raise ValueError(f"unsupported grad_mode: {grad_mode}")

    grad_outputs: list[torch.Tensor] = []
    generators: dict[torch.device, torch.Generator] = {}
    for output in outputs:
        if output.device not in generators:
            # Generators are device-local; a CUDA generator cannot draw CPU tensors.
            generator = torch.Generator(device=output.device)
            generator.manual_seed(seed)
            generators[output.device] = generator
        # Random upstream gradients test a non-uniform dL/doutput. The same
        # tensors are later reused for gold so the comparison stays fair.
        grad_outputs.append(
            torch.randn(
                output.shape,
                generator=generators[output.device],
                device=output.device,
                dtype=torch.float32,
            )
        )
    return grad_outputs


def _match_grad_outputs(
    grad_outputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    # Reuse upstream values for gold; only move device when needed.
    return [
        grad_output.to(device=output.device)
        for grad_output, output in zip(grad_outputs, outputs, strict=True)
    ]


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
    backend_profile: str | None = None,
    judgment: str = "forward_accuracy",
) -> tuple[float, float]:
    """Resolve thresholds via the shared four-judgment contract (#267).

    Falls back to the legacy ``accuracy`` mirror only when the four-judgment
    block is absent (older fixture contracts in unit tests).
    """

    if "judgments" in contract:
        spec = resolve_tolerance(
            contract,
            judgment=judgment,
            op_class=op_class,
            dtype=dtype,
            arch_key=arch_key,
            backend_profile=backend_profile,
        )
        return float(spec.atol), float(spec.rtol)

    # Legacy fixtures used by some unit tests that inject a minimal contract.
    # They only mirror forward accuracy thresholds; never apply them to grads.
    if judgment != "forward_accuracy":
        raise ContractResolveError(
            f"legacy accuracy contracts only support judgment='forward_accuracy'; "
            f"got {judgment!r}"
        )
    dtype_name = normalize_dtype_name(dtype)
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


_BITWISE_VIEWS = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


def _bitwise_verdict(
    check: OutputCheck, candidate_output: torch.Tensor, gold_output: torch.Tensor
) -> OutputCheck:
    """Override a tolerance verdict with the exact bit-pattern bar.

    Expected bits are the gold's logical elements after ONE cast to the
    candidate's output dtype (the frozen contract's single RNE output cast);
    equality is asserted on the dtype bitcast, so NaN payloads and signed
    zeros compare by pattern, not by value. Error statistics computed by the
    tolerance comparison are preserved as diagnostics.
    """
    if candidate_output.shape != gold_output.shape:
        passed = False
        note = "bitwise: shape mismatch"
    else:
        if candidate_output.dtype not in _BITWISE_VIEWS:
            raise ValueError(
                f"bitwise_strict supports bf16/fp32 outputs, got {candidate_output.dtype}"
            )
        expected = (
            gold_output
            if gold_output.dtype == candidate_output.dtype
            else gold_output.to(candidate_output.dtype)
        )
        passed = bool(
            torch.equal(
                expected.contiguous().view(_BITWISE_VIEWS[candidate_output.dtype]),
                candidate_output.contiguous().view(_BITWISE_VIEWS[candidate_output.dtype]),
            )
        )
        note = "bitwise" if passed else "bitwise: bit-pattern mismatch"
    message = f"{check.message} [{note}]" if check.message else note
    return replace(check, passed=passed, message=message)


def _compare_output(
    candidate: torch.Tensor,
    gold: torch.Tensor,
    *,
    output_index: int,
    atol: float,
    rtol: float,
    judgment: str = "forward_accuracy",
    comparison_lhs_role: str = "bf16_candidate",
    comparison_rhs_role: str = "fp32_reference",
    message: str = "",
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
            judgment=judgment,
            comparison_lhs_role=comparison_lhs_role,
            comparison_rhs_role=comparison_rhs_role,
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
        judgment=judgment,
        comparison_lhs_role=comparison_lhs_role,
        comparison_rhs_role=comparison_rhs_role,
        message=message,
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
