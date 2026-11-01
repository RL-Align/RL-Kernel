# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Versioned M10 boundary input schema and fail-closed statuses.

The field set mirrors ``p7-task-contract.v3`` §二 "共享 M10 输入合同". The final
dtype/shape/layout is frozen by T01's boundary-contract manifest; fields that the
manifest marks as versioned are carried through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from rl_engine.validation.cross_config.schema import SerializableModel
from rl_engine.validation.sampling_params.schema import CanonicalSamplingRecord

M10_INPUT_SCHEMA_VERSION = "p7-m10-input-v1"


class M10Status(str, Enum):
    """Fail-closed statuses for the M10 boundary (contract §七).

    The sampling-parameters statuses are included so the M10 gate can surface a
    sampling failure without a second vocabulary of states.
    """

    PASS = "pass"
    SAMPLING_PARAMS_MISMATCH = "sampling_params_mismatch"
    MISSING_PROVENANCE = "missing_provenance"
    SILENT_FALLBACK = "silent_fallback"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    PRE_UPDATE_WEIGHT_DRIFT = "pre_update_weight_drift"
    ZERO_ACTIVE_TOKENS = "zero_active_tokens"
    NON_FINITE = "non_finite"
    INVALID_PROFILE = "invalid_profile"


@dataclass
class M10Input(SerializableModel):
    """One M10 boundary input, shared verbatim by both engines.

    ``final_decoder_rows`` is the only permitted source of M10 logits; a full
    ``[token, vocab]`` logits materialization is never built.
    """

    final_decoder_rows: torch.Tensor  # BF16 [N, H]
    target_token_ids: torch.Tensor  # INT64 [N]
    active_token_mask: torch.Tensor  # BOOL  [N]
    lm_head_weight_local: torch.Tensor  # BF16 [V_local, H]
    vocab_start: int
    vocab_end: int
    global_vocab_size: int
    pre_update_weight_lock: str
    sampling_record: Optional[CanonicalSamplingRecord] = None
    weight_id: str = ""
    rank: int = 0
    group: str = "single"
    lm_head_lora: bool = False
    schema_version: str = M10_INPUT_SCHEMA_VERSION

    @property
    def local_vocab_size(self) -> int:
        return int(self.lm_head_weight_local.size(0))
