# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Single-GPU selected-logprob comparison."""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

if __package__ in (None, ""):
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from logprob_drift import LogprobDriftStats, summarize_logprob_drift
else:
    from rl_engine.validation.common.logprob_drift import LogprobDriftStats, summarize_logprob_drift


class LogprobBackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class LogprobComparisonInputs:
    logits: torch.Tensor
    target_ids: torch.Tensor
    active_token_mask: torch.Tensor | None = None
    ignore_index: int = -100


@dataclass(frozen=True)
class LogprobCandidate:
    name: str
    requested_backend: str
    actual_backend: str
    fn: Callable[[torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _LogprobPathDrift:
    candidate_name: str
    lse: LogprobDriftStats
    dlogp: LogprobDriftStats
    bitwise_logp: bool
    provenance: dict[str, Any]


@dataclass(frozen=True)
class LogprobComparisonReport:
    reference_name: str
    drifts: tuple[_LogprobPathDrift, ...]
    input_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_logprob_candidate(backend: str) -> LogprobCandidate:
    normalized = backend.strip().lower().replace("_", "-")
    op: Any
    if normalized in {"pytorch", "native"}:
        from rl_engine.reference.logprob.batch_invariant_logp import NativeBatchInvariantLogpOp

        op = NativeBatchInvariantLogpOp()
        actual = "pytorch"
    elif normalized == "triton":
        try:
            from rl_engine.backends.shared.triton.logprob.batch_invariant_logp import (
                TritonBatchInvariantLogpOp,
            )

            op = TritonBatchInvariantLogpOp()
        except Exception as exc:
            raise LogprobBackendUnavailable(f"triton backend is unavailable: {exc}") from exc
        actual = "triton"
    elif normalized in {"cuda-sm90", "sm90"}:
        try:
            from rl_engine.backends.cuda.logprob.batch_invariant_logp import (
                BatchInvariantLogpSM90Op,
            )

            op = BatchInvariantLogpSM90Op()
        except Exception as exc:
            raise LogprobBackendUnavailable(f"cuda-sm90 backend is unavailable: {exc}") from exc
        actual = "cuda-sm90"
    else:
        raise ValueError(
            f"unsupported logprob comparison backend {backend!r}; "
            "expected pytorch, triton, or cuda-sm90"
        )

    diagnostic = getattr(op, "forward_with_lse", None)
    if not callable(diagnostic):
        raise LogprobBackendUnavailable(
            f"backend {normalized!r} does not expose the required direct LSE diagnostic"
        )

    def run(
        logits: torch.Tensor, target_ids: torch.Tensor, ignore_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return diagnostic(logits, target_ids, ignore_index=ignore_index, validate=True)
        except (RuntimeError, NotImplementedError, OSError) as exc:
            raise LogprobBackendUnavailable(
                f"exact backend {normalized!r} cannot execute this input: {exc}"
            ) from exc

    return LogprobCandidate(
        name=f"{actual}-batch-invariant-logp",
        requested_backend=actual,
        actual_backend=actual,
        fn=run,
        provenance={
            "requested_alias": normalized,
            "implementation": f"{type(op).__module__}.{type(op).__qualname__}",
        },
    )


def compare_single_gpu_logprob(
    inputs: LogprobComparisonInputs,
    *,
    candidates: Sequence[str | LogprobCandidate] = ("pytorch",),
) -> LogprobComparisonReport:
    active_mask, effective_targets = _validate_inputs(inputs)
    reference_logp, reference_lse = _run_ws1_reference(
        inputs.logits, effective_targets, inputs.ignore_index
    )

    drifts = []
    for candidate in candidates:
        if isinstance(candidate, str):
            candidate = make_logprob_candidate(candidate)
        logp, lse = _run_candidate(
            candidate,
            inputs.logits,
            effective_targets,
            inputs.ignore_index,
        )
        drifts.append(
            _LogprobPathDrift(
                candidate_name=candidate.name,
                lse=summarize_logprob_drift(lse, reference_lse),
                dlogp=summarize_logprob_drift(logp, reference_logp, mask=active_mask),
                bitwise_logp=torch.equal(logp, reference_logp),
                provenance=_candidate_provenance(candidate),
            )
        )

    return LogprobComparisonReport(
        reference_name="pytorch-batch-invariant-logp",
        drifts=tuple(drifts),
        input_provenance={
            "device": str(inputs.logits.device),
            "input_dtype": str(inputs.logits.dtype),
            "output_dtype": str(reference_logp.dtype),
            "shape": list(inputs.logits.shape),
            "ignore_index": inputs.ignore_index,
            "active_token_count": int(active_mask.sum().item()),
            "tp_world": 1,
            "communication": "none",
        },
    )


def _run_ws1_reference(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from rl_engine.reference.logprob.batch_invariant_logp import NativeBatchInvariantLogpOp

    op = NativeBatchInvariantLogpOp()
    logp = op(logits, target_ids, ignore_index=ignore_index, validate=True)
    _, lse = op.forward_with_lse(logits, target_ids, ignore_index=ignore_index, validate=True)
    return logp.detach(), lse.detach()


def _run_candidate(
    candidate: LogprobCandidate,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate.requested_backend != candidate.actual_backend:
        raise LogprobBackendUnavailable(
            f"requested backend {candidate.requested_backend!r} materialized as "
            f"{candidate.actual_backend!r}; silent fallback is forbidden"
        )
    logp, lse = candidate.fn(logits, target_ids, ignore_index)
    expected_shape = logits.shape[:-1]
    for name, value in (("logp", logp), ("lse", lse)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"candidate {candidate.name!r} {name} must be a tensor")
        if value.shape != expected_shape:
            raise ValueError(
                f"candidate {candidate.name!r} {name} shape {tuple(value.shape)} "
                f"does not match {tuple(expected_shape)}"
            )
        if value.dtype != torch.float32:
            raise ValueError(f"candidate {candidate.name!r} {name} must be FP32")
    return logp.detach(), lse.detach()


def _candidate_provenance(candidate: LogprobCandidate) -> dict[str, Any]:
    return {
        **candidate.provenance,
        "requested_backend": candidate.requested_backend,
        "actual_backend": candidate.actual_backend,
        "tp_world": 1,
        "communication": "none",
        "lse_source": "direct",
    }


def _validate_inputs(
    inputs: LogprobComparisonInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.logits.dim() < 2:
        raise ValueError("logits must be at least 2-D [*lead, vocab]")
    if inputs.logits.shape[:-1] != inputs.target_ids.shape:
        raise ValueError("target_ids shape must match logits leading shape")
    if not inputs.logits.is_floating_point():
        raise ValueError("logits must be floating point")

    if inputs.active_token_mask is None:
        active = inputs.target_ids != inputs.ignore_index
    else:
        if inputs.active_token_mask.shape != inputs.target_ids.shape:
            raise ValueError("active_token_mask shape must match target_ids")
        if inputs.active_token_mask.dtype != torch.bool:
            raise ValueError("active_token_mask must be bool")
        active = inputs.active_token_mask.to(device=inputs.target_ids.device)
        if bool(((inputs.target_ids == inputs.ignore_index) & active).any().item()):
            raise ValueError("active target_ids cannot equal ignore_index")

    effective = inputs.target_ids.to(device=inputs.logits.device, dtype=torch.long).clone()
    active = active.to(device=inputs.logits.device, dtype=torch.bool)
    effective.masked_fill_(~active, inputs.ignore_index)
    valid = effective[active]
    vocab_size = inputs.logits.size(-1)
    if valid.numel() and ((valid < 0).any() or (valid >= vocab_size).any()):
        raise ValueError(f"active target_ids must be in [0, {vocab_size})")
    return active, effective


def _dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def route_rl_kernel_logs_to_stderr() -> None:
    from rl_engine.utils.logger import logger

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stderr)


def _route_rl_kernel_logs_to_stderr() -> None:
    route_rl_kernel_logs_to_stderr()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the WS2 TP=1 selected-logprob/LSE comparison harness."
    )
    parser.add_argument(
        "--candidate",
        action="append",
        choices=("pytorch", "triton", "cuda-sm90"),
        help="Exact backend to compare. Repeat for multiple backends; defaults to pytorch.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--vocab", type=int, default=257)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    route_rl_kernel_logs_to_stderr()
    args = _parse_args(argv)
    device = _device(args.device)
    if args.batch < 1 or args.seq < 1 or args.vocab < 1:
        raise ValueError("batch, seq, and vocab must be positive")
    if not 0 <= args.prompt_tokens <= args.seq:
        raise ValueError("prompt-tokens must be in [0, seq]")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    logits = torch.randn(
        args.batch,
        args.seq,
        args.vocab,
        generator=generator,
        device=device,
        dtype=_dtype(args.dtype),
    )
    target_ids = torch.randint(
        0,
        args.vocab,
        (args.batch, args.seq),
        generator=generator,
        device=device,
    )
    active_mask = torch.ones((args.batch, args.seq), device=device, dtype=torch.bool)
    active_mask[:, : args.prompt_tokens] = False
    report = compare_single_gpu_logprob(
        LogprobComparisonInputs(
            logits=logits,
            target_ids=target_ids,
            active_token_mask=active_mask,
        ),
        candidates=tuple(args.candidate or ("pytorch",)),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


__all__ = [
    "LogprobBackendUnavailable",
    "LogprobCandidate",
    "LogprobComparisonInputs",
    "LogprobComparisonReport",
    "compare_single_gpu_logprob",
    "make_logprob_candidate",
    "route_rl_kernel_logs_to_stderr",
]


if __name__ == "__main__":
    main()
