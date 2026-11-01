# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fail-closed M10 pre-checks that run before any scoring.

Order mirrors the T02 spec §7.1: sampling gate -> LM Head LoRA -> input
well-formedness (dtype/shape/vocab ownership/padded targets) -> active and
non-finite guards. The identity/discrete profile gate and the P1 GEMM
arithmetic/harness check belong to T01/T03 and are deliberately not duplicated
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from rl_engine.validation.cross_config.schema import SerializableModel
from rl_engine.validation.m10.schema import M10Input, M10Status
from rl_engine.validation.sampling_params.gate import evaluate_sampling_gate


@dataclass(frozen=True)
class M10PrecheckResult(SerializableModel):
    """Typed result of the M10 pre-checks."""

    status: M10Status
    passed: bool
    failing_field: Optional[str] = None
    detail: str = ""


def _fail(status: M10Status, field_name: str, detail: str) -> M10PrecheckResult:
    return M10PrecheckResult(status=status, passed=False, failing_field=field_name, detail=detail)


def evaluate_m10_precheck(
    inp: M10Input,
    *,
    expected_pre_update_weight_lock: Optional[str] = None,
) -> M10PrecheckResult:
    """Run every fail-closed pre-check for one M10 input."""
    if inp.sampling_record is None:
        return _fail(
            M10Status.MISSING_PROVENANCE,
            "sampling_record",
            "M10 input carries no canonical sampling record",
        )
    gate = evaluate_sampling_gate(inp.sampling_record)
    if not gate.passed:
        field_name = "sampling_record"
        if gate.failing_layer:
            field_name = f"sampling_record.{gate.failing_layer}"
        return _fail(M10Status(gate.status.value), field_name, gate.detail)

    if inp.lm_head_lora:
        return _fail(
            M10Status.UNSUPPORTED_CAPABILITY,
            "lm_head_lora",
            "LM Head LoRA is not supported in strict consistency mode",
        )

    if not inp.pre_update_weight_lock:
        return _fail(
            M10Status.MISSING_PROVENANCE,
            "pre_update_weight_lock",
            "missing pre-update weight fingerprint",
        )
    if (
        expected_pre_update_weight_lock is not None
        and inp.pre_update_weight_lock != expected_pre_update_weight_lock
    ):
        return _fail(
            M10Status.PRE_UPDATE_WEIGHT_DRIFT,
            "pre_update_weight_lock",
            f"pre-update weight lock {inp.pre_update_weight_lock!r} "
            f"does not match {expected_pre_update_weight_lock!r}",
        )

    rows = inp.final_decoder_rows
    weight = inp.lm_head_weight_local
    target = inp.target_token_ids
    active = inp.active_token_mask

    if rows.ndim != 2:
        return _fail(M10Status.INVALID_PROFILE, "final_decoder_rows", "rows must be [N, H]")
    if rows.dtype != torch.bfloat16:
        return _fail(
            M10Status.INVALID_PROFILE,
            "final_decoder_rows",
            f"rows must be BF16, got {rows.dtype}",
        )
    if weight.dtype != torch.bfloat16:
        return _fail(
            M10Status.INVALID_PROFILE,
            "lm_head_weight_local",
            f"lm_head weight must be BF16, got {weight.dtype}",
        )
    if weight.ndim != 2 or weight.size(1) != rows.size(1):
        return _fail(
            M10Status.INVALID_PROFILE,
            "lm_head_weight_local",
            "lm_head weight must be [V_local, H] with H matching the rows",
        )
    if target.dtype != torch.int64 or target.shape != (rows.size(0),):
        return _fail(
            M10Status.INVALID_PROFILE,
            "target_token_ids",
            "target ids must be INT64 [N] matching the rows",
        )
    if active.dtype != torch.bool or active.shape != target.shape:
        return _fail(
            M10Status.INVALID_PROFILE,
            "active_token_mask",
            "active mask must be BOOL [N] matching the target ids",
        )

    if not (0 <= inp.vocab_start < inp.vocab_end):
        return _fail(
            M10Status.INVALID_PROFILE,
            "vocab_ownership",
            f"vocab shard [{inp.vocab_start}, {inp.vocab_end}) is not a valid half-open range",
        )
    if inp.vocab_end - inp.vocab_start != inp.local_vocab_size:
        return _fail(
            M10Status.INVALID_PROFILE,
            "vocab_ownership",
            f"vocab shard width {inp.vocab_end - inp.vocab_start} "
            f"does not match the local weight rows {inp.local_vocab_size}",
        )
    if inp.global_vocab_size > 0 and inp.vocab_end > inp.global_vocab_size:
        return _fail(
            M10Status.INVALID_PROFILE,
            "vocab_ownership",
            f"vocab shard end {inp.vocab_end} exceeds the real vocab "
            f"{inp.global_vocab_size} (padded rows must not be owned as real)",
        )

    if not bool(active.any()):
        return _fail(
            M10Status.ZERO_ACTIVE_TOKENS,
            "active_token_mask",
            "no active action tokens to score",
        )

    if inp.global_vocab_size > 0:
        active_targets = target[active]
        if bool((active_targets < 0).any()) or bool(
            (active_targets >= inp.global_vocab_size).any()
        ):
            return _fail(
                M10Status.INVALID_PROFILE,
                "target_token_ids",
                "an active target lands outside the real vocab (padded vocab mask)",
            )

    if not bool(torch.isfinite(rows).all()):
        return _fail(
            M10Status.NON_FINITE,
            "final_decoder_rows",
            "final decoder rows contain non-finite values",
        )

    return M10PrecheckResult(
        status=M10Status.PASS,
        passed=True,
        detail="M10 input is well-formed and identity-sampled",
    )
