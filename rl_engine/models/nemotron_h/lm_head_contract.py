# SPDX-License-Identifier: Apache-2.0
"""Pinned embedding / LM-head / selected-token logprob contract for Nemotron-H.

Values come from ``config.json`` and ``modeling_nemotron_h.py`` of
``nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`` at the pinned revision. The
geometry gate is fail-closed: strict mode never silently runs an unsupported
shape, dtype, layout, bias, softcap, or tied-embedding configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

CHECKPOINT_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
CHECKPOINT_REVISION = "bf77c31"

LM_HEAD_LOGPROB_CONTRACT_VERSION = "nemotron-h-lm-head-logp-sm90-contract-v1"


@dataclass(frozen=True)
class NemotronHLMHeadSpec:
    """Checkpoint fingerprint relevant to the model input and output boundary."""

    hidden_size: int
    vocab_size: int
    param_dtype: torch.dtype
    tie_word_embeddings: bool
    embedding_scale: Optional[float]
    lm_head_bias: bool
    final_logit_softcap: Optional[float]


NEMOTRON_H_LM_HEAD_SPEC = NemotronHLMHeadSpec(
    hidden_size=2688,
    vocab_size=131072,
    param_dtype=torch.bfloat16,
    tie_word_embeddings=False,
    embedding_scale=None,
    lm_head_bias=False,
    final_logit_softcap=None,
)


class NemotronHGeometryError(ValueError):
    """Raised when inputs fall outside the declared strict geometry."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NemotronHGeometryError(f"Nemotron-H lm_head_logprob strict contract: {message}")


def _require_weight(weight: torch.Tensor, name: str, spec: NemotronHLMHeadSpec) -> None:
    expected = (spec.vocab_size, spec.hidden_size)
    shape = tuple(weight.shape)
    _require(weight.dim() == 2, f"{name} must be 2-D [vocab, hidden], got {shape}")
    _require(shape == expected, f"{name} must be {expected}, got {shape}")
    _require(
        weight.dtype == spec.param_dtype,
        f"{name} must be {spec.param_dtype}, got {weight.dtype}",
    )
    _require(weight.is_contiguous(), f"{name} must be contiguous")


def _require_ids(ids: torch.Tensor, name: str) -> None:
    _require(
        ids.dtype in (torch.int32, torch.int64),
        f"{name} must be int32/int64, got {ids.dtype}",
    )


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a is b:
        return True
    if a.device.type == "meta" or b.device.type == "meta":
        # Meta tensors have no storage; only identity can prove tying.
        return False
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def validate_embedding_geometry(
    token_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    spec: NemotronHLMHeadSpec = NEMOTRON_H_LM_HEAD_SPEC,
) -> None:
    """Fail closed unless the embedding lookup matches the pinned checkpoint."""

    _require_weight(embed_weight, "embed_weight", spec)
    _require_ids(token_ids, "token_ids")
    _require(token_ids.device == embed_weight.device, "token_ids and embed_weight devices differ")


def validate_lm_head_logprob_geometry(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    final_logit_softcap: Optional[float] = None,
    embed_weight: Optional[torch.Tensor] = None,
    temperature: Optional[torch.Tensor] = None,
    spec: NemotronHLMHeadSpec = NEMOTRON_H_LM_HEAD_SPEC,
) -> None:
    """Fail closed unless the LM-head logprob inputs match the pinned checkpoint."""

    _require_weight(lm_head_weight, "lm_head_weight", spec)
    _require(hidden.dim() >= 1, "hidden must have at least one dimension")
    _require(
        hidden.size(-1) == spec.hidden_size,
        f"hidden last dim must be {spec.hidden_size}, got {hidden.size(-1)}",
    )
    _require(
        hidden.dtype == spec.param_dtype,
        f"hidden must be {spec.param_dtype}, got {hidden.dtype}",
    )
    _require(hidden.device == lm_head_weight.device, "hidden and lm_head_weight devices differ")
    _require_ids(target_ids, "target_ids")
    _require(
        target_ids.numel() == hidden.numel() // spec.hidden_size,
        "target_ids must have one id per hidden row",
    )
    _require(bias is None and not spec.lm_head_bias, "the checkpoint LM head has no bias")
    _require(
        final_logit_softcap is None and spec.final_logit_softcap is None,
        "the checkpoint applies no final logit softcap",
    )
    if embed_weight is not None:
        _require(
            not spec.tie_word_embeddings and not _shares_storage(embed_weight, lm_head_weight),
            "embedding and LM-head weights must be untied (tie_word_embeddings=false)",
        )
    if temperature is not None:
        _require(temperature.dtype == torch.float32, "temperature must be float32")
        _require(
            temperature.numel() in (1, target_ids.numel()),
            "temperature must be scalar or one value per token",
        )
