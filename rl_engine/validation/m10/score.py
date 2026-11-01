# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""M10 scoring: call the existing strict linear_logp and capture raw FP32 bytes.

T02 never reimplements the LM-head GEMM, the log-softmax or the LSE merge. It
consumes :class:`LinearLogpWrapper` (``rlkernel.linear_logp.bitwise.v1``), the
frozen strict boundary shared by the training side and the rollout side, and
captures the active selected-token FP32 raw bytes that T09 compares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch

from rl_engine.integrations.common.linear_logp import LinearLogpWrapper
from rl_engine.validation.m10.precheck import evaluate_m10_precheck
from rl_engine.validation.m10.schema import M10Input, M10Status

# Bit-relevant vocab reduction tile width fixed by the contract for M10.
M10_VOCAB_REDUCTION_TILE = 64


def fp32_raw_bytes(values: torch.Tensor) -> bytes:
    """Return raw little-endian FP32 bytes of a contiguous tensor."""
    flat = values.detach().to(torch.float32).contiguous().reshape(-1)
    return flat.cpu().view(torch.uint8).numpy().tobytes()


@dataclass(frozen=True)
class M10ScoreResult:
    """Result of one M10 scoring pass."""

    status: M10Status
    passed: bool
    active_token_count: int
    selected_logprob: Optional[torch.Tensor]
    selected_logprob_bytes: bytes
    provenance: Mapping[str, Any] = field(default_factory=dict)
    detail: str = ""
    active_token_mask: Optional[torch.Tensor] = None

    @property
    def selected_logprob_active(self) -> Optional[torch.Tensor]:
        """Active-row slice aligned with ``selected_logprob_bytes``.

        ``selected_logprob`` keeps the full ``[N]`` wrapper output (contract §5.1);
        consumers that pair it with the byte buffer must mask it first.
        """
        if self.selected_logprob is None or self.active_token_mask is None:
            return None
        return self.selected_logprob[self.active_token_mask]

    def selected_logprob_bytes_hex(self) -> str:
        return self.selected_logprob_bytes.hex()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view; the raw bytes are rendered as hex."""
        return {
            "status": self.status.value,
            "passed": self.passed,
            "active_token_count": self.active_token_count,
            "selected_logprob_bytes": self.selected_logprob_bytes.hex(),
            "provenance": dict(self.provenance),
            "detail": self.detail,
        }


def score_m10(
    inp: M10Input,
    *,
    device: Optional[torch.device] = None,
    target: str = "training",
    expected_pre_update_weight_lock: Optional[str] = None,
) -> M10ScoreResult:
    """Pre-check, score with the strict linear_logp, and capture the raw bytes.

    Fails closed: a pre-check failure returns the typed status with no bytes and
    never reaches the kernel. The strict wrapper itself rejects non-CUDA or
    non-BF16 input rather than silently falling back.
    """
    active_count = int(inp.active_token_mask.sum().item())
    precheck = evaluate_m10_precheck(
        inp, expected_pre_update_weight_lock=expected_pre_update_weight_lock
    )
    if not precheck.passed:
        return M10ScoreResult(
            status=precheck.status,
            passed=False,
            active_token_count=active_count,
            selected_logprob=None,
            selected_logprob_bytes=b"",
            detail=precheck.detail,
        )

    target_device = device if device is not None else inp.final_decoder_rows.device
    rows = inp.final_decoder_rows.to(target_device)
    weight = inp.lm_head_weight_local.to(target_device)
    targets = inp.target_token_ids.to(target_device)
    mask = inp.active_token_mask.to(target_device)

    op = LinearLogpWrapper()
    logprob = op(
        rows,
        weight,
        targets,
        None,
        vocab_start_index=inp.vocab_start,
        global_vocab_size=inp.global_vocab_size,
        real_vocab_size=inp.effective_real_vocab_size,
        target=target,
    )
    selected = logprob.detach().to(torch.float32).reshape(-1)
    if not bool(torch.isfinite(selected).all()):
        return M10ScoreResult(
            status=M10Status.NON_FINITE,
            passed=False,
            active_token_count=active_count,
            selected_logprob=None,
            selected_logprob_bytes=b"",
            detail="strict linear_logp returned a non-finite selected logprob",
        )
    raw = fp32_raw_bytes(selected[mask])

    arithmetic = dict(dict(op.provenance).get("arithmetic_contract") or {})
    vocab_tile = int(arithmetic.get("vocab_group_width", M10_VOCAB_REDUCTION_TILE))
    provenance = {
        **dict(op.provenance),
        "m10_schema_version": inp.schema_version,
        "weight_id": inp.weight_id,
        "pre_update_weight_lock": inp.pre_update_weight_lock,
        "pre_update_weight_lock_verified": expected_pre_update_weight_lock is not None,
        "vocab_shard": [int(inp.vocab_start), int(inp.vocab_end)],
        "global_vocab_size": int(inp.global_vocab_size),
        "active_token_count": active_count,
        "sampling_record_fingerprint": (
            inp.sampling_record.canonical_fingerprint() if inp.sampling_record else None
        ),
        "selected_logprob_dtype": "fp32",
        "vocab_reduction_tile": vocab_tile,
    }
    return M10ScoreResult(
        status=M10Status.PASS,
        passed=True,
        active_token_count=active_count,
        selected_logprob=selected,
        selected_logprob_bytes=raw,
        provenance=provenance,
        active_token_mask=mask.detach().to("cpu"),
        detail="strict SM90 linear_logp scored the active selected tokens",
    )
