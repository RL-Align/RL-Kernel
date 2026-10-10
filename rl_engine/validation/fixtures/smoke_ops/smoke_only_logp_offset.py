# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""TEMPORARY TEST SCAFFOLD - NOT A PRODUCTION RL-KERNEL OPERATOR"""

from __future__ import annotations

import math
from typing import Optional

import torch

from rl_engine.runtime.semantic_registry import (
    OperatorBackendDescriptor,
    OperatorFallbackPolicy,
    OperatorLifecycle,
)
from rl_engine.validation.fixtures.smoke_ops import SMOKE_ONLY_LOGP_OFFSET_BACKEND_ID
from rl_engine.validation.fixtures.smoke_ops.smoke_only_logp_reference import SmokeOnlyLogpReference


class SmokeOnlyLogpOffset(SmokeOnlyLogpReference):
    """CPU reference plus a deterministic test-only active-token offset.

    Inputs and output follow :class:`SmokeOnlyLogpReference`. ``offset`` is zero
    by default. A non-zero value requires the constructor's explicit
    ``allow_smoke_operators=True`` guard. The cross-configuration bridge masks
    inactive output positions after invocation, so drift applies only to active
    selected tokens.
    """

    def __init__(
        self,
        offset: float = 0.0,
        *,
        allow_smoke_operators: bool = False,
    ) -> None:
        normalized_offset = float(offset)
        if not math.isfinite(normalized_offset):
            raise ValueError("offset must be finite")
        if normalized_offset != 0.0 and allow_smoke_operators is not True:
            raise PermissionError(
                "a non-zero smoke offset requires explicit " "allow_smoke_operators=True"
            )
        self.offset = normalized_offset

    def apply_fp32(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        selected = super().apply_fp32(logits, token_ids, active_mask=active_mask)
        if self.offset == 0.0:
            return selected
        if active_mask is None:
            return selected + self.offset
        mask = active_mask.to(device=selected.device, dtype=torch.bool)
        return selected + mask.to(dtype=selected.dtype) * self.offset


SMOKE_ONLY_LOGP_OFFSET_DESCRIPTOR = OperatorBackendDescriptor(
    semantic_op="selected_logprob",
    backend_id=SMOKE_ONLY_LOGP_OFFSET_BACKEND_ID,
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
        "algorithm": "pytorch.log_softmax_gather_plus_test_offset",
        "batch_invariant": True,
        "deterministic": True,
        "strict_observable": True,
        "test_offset_configurable": True,
    },
    lifecycle=OperatorLifecycle.ENGINE_CONSTRUCTION,
    implementation_class_or_factory=SmokeOnlyLogpOffset,
    fallback_policy=OperatorFallbackPolicy.ERROR,
    version_or_build_fingerprint="cross-config-smoke-only-logp-offset-v1",
    is_smoke_only=True,
)


__all__ = [
    "SMOKE_ONLY_LOGP_OFFSET_DESCRIPTOR",
    "SmokeOnlyLogpOffset",
]
