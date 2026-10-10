# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""TEMPORARY TEST SCAFFOLD - NOT A PRODUCTION RL-KERNEL OPERATOR"""

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.runtime.semantic_registry import (
    OperatorBackendDescriptor,
    OperatorFallbackPolicy,
    OperatorLifecycle,
)
from rl_engine.validation.fixtures.smoke_ops import SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID


class SmokeOnlyLogpReference:
    """CPU selected-logprob reference used only to test operator plumbing.

    The semantic inputs are logits shaped ``[..., vocabulary]`` and selected
    token IDs shaped ``[...]``. The result is one float32 log probability per
    selected token. When supplied, ``active_mask`` has the token-ID shape and
    inactive output positions are exactly zero. The cross-configuration bridge
    applies the same masking rule when invoking the two-argument interface.
    """

    op_class = "logprob"

    def __call__(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.apply_fp32(logits, token_ids, active_mask=active_mask)

    def apply_fp32(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute CPU log-softmax/gather output with optional active masking."""

        _validate_inputs(logits, token_ids, active_mask)
        selected_ids = token_ids.to(device=logits.device, dtype=torch.long)
        mask = None
        if active_mask is not None:
            mask = active_mask.to(device=logits.device, dtype=torch.bool)
            selected_ids = selected_ids.masked_fill(~mask, 0)

        log_probs = torch.log_softmax(logits.float(), dim=-1)
        selected = torch.gather(log_probs, dim=-1, index=selected_ids.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            selected = selected.masked_fill(~mask, 0.0)
        return selected


def _validate_inputs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    active_mask: Optional[torch.Tensor],
) -> None:
    if logits.device.type != "cpu":
        raise ValueError("smoke-only logprob operators support CPU tensors only")
    if logits.shape[:-1] != token_ids.shape:
        raise ValueError(
            f"logits leading shape {tuple(logits.shape[:-1])} must match "
            f"token_ids shape {tuple(token_ids.shape)}"
        )
    if active_mask is not None and active_mask.shape != token_ids.shape:
        raise ValueError("active_mask shape must match token_ids shape")


SMOKE_ONLY_LOGP_REFERENCE_DESCRIPTOR = OperatorBackendDescriptor(
    semantic_op="selected_logprob",
    backend_id=SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID,
    supported_targets=frozenset({"rollout", "training"}),
    supported_devices=frozenset({"cpu"}),
    supported_dtypes=frozenset({"bfloat16", "float16", "float32"}),
    supported_topologies={
        "rollout": {
            "world_size": (1,),
            "tensor_parallel_size": (1,),
            "context_parallel_size": (1,),
        },
        "training": {
            "world_size": (1,),
            "sharding": ("unsharded",),
        },
    },
    determinism_or_alignment_properties={
        "algorithm": "pytorch.log_softmax_gather",
        "batch_invariant": True,
        "deterministic": True,
        "strict_observable": True,
    },
    lifecycle=OperatorLifecycle.ENGINE_CONSTRUCTION,
    implementation_class_or_factory=SmokeOnlyLogpReference,
    fallback_policy=OperatorFallbackPolicy.ERROR,
    version_or_build_fingerprint="cross-config-smoke-only-logp-reference-v1",
    is_smoke_only=True,
)


__all__ = [
    "SMOKE_ONLY_LOGP_REFERENCE_DESCRIPTOR",
    "SmokeOnlyLogpReference",
]
