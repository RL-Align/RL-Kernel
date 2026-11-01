# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Synthetic M10 fixtures and the stub provider for T02.

The contract allows T02–T12 to start on synthetic fixtures before the real
providers exist (``p7-task-contract.v3`` §六). The stub provider exposes the same
read ABI a recorded or live provider will, so swapping it out later keeps the
case id, schema, runner and acceptance command unchanged.

Only synthetic data is produced here; no real checkpoint, token or production
batch may enter a fixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.validation.m10.schema import M10Input
from rl_engine.validation.sampling_params.schema import (
    CanonicalSamplingLayer,
    CanonicalSamplingRecord,
)


def build_m10_stub(
    *,
    tokens: int = 8,
    hidden: int = 64,
    local_vocab: int = 128,
    vocab_start: int = 0,
    global_vocab_size: int | None = None,
    active_ratio: float = 1.0,
    lm_head_lora: bool = False,
    seed: int = 0,
    case_id: str = "stub-case",
) -> M10Input:
    """Build one deterministic, synthetic M10 input."""
    if tokens <= 0 or hidden <= 0 or local_vocab <= 0:
        raise ValueError("stub dimensions must be positive")
    if not 0.0 <= active_ratio <= 1.0:
        raise ValueError("active_ratio must be within [0, 1]")

    torch.manual_seed(seed)
    rows = torch.randn(tokens, hidden, dtype=torch.float32).to(torch.bfloat16)
    weight = torch.randn(local_vocab, hidden, dtype=torch.float32).to(torch.bfloat16)
    vocab_end = vocab_start + local_vocab
    target = torch.randint(vocab_start, vocab_end, (tokens,), dtype=torch.int64)

    active = torch.ones(tokens, dtype=torch.bool)
    if active_ratio < 1.0:
        keep = max(1, int(round(tokens * active_ratio)))
        active[keep:] = False

    identity = CanonicalSamplingLayer()
    sampling_record = CanonicalSamplingRecord(
        case_id=case_id,
        engine="stub",
        requested=identity,
        normalized=identity,
        materialized=identity,
        runtime_actual=identity,
    )

    return M10Input(
        final_decoder_rows=rows,
        target_token_ids=target,
        active_token_mask=active,
        lm_head_weight_local=weight,
        vocab_start=vocab_start,
        vocab_end=vocab_end,
        global_vocab_size=vocab_end if global_vocab_size is None else global_vocab_size,
        pre_update_weight_lock=f"stub-lock-{case_id}",
        sampling_record=sampling_record,
        weight_id=f"stub-weight-{seed}",
        lm_head_lora=lm_head_lora,
    )


@dataclass
class StubM10Provider:
    """Stub provider exposing the M10 read ABI used by recorded/live providers."""

    _fixture_kwargs: dict[str, Any]

    def __init__(self, **fixture_kwargs: Any) -> None:
        self._fixture_kwargs = dict(fixture_kwargs)

    def read(self, case_id: str) -> M10Input:
        return build_m10_stub(case_id=case_id, **self._fixture_kwargs)
