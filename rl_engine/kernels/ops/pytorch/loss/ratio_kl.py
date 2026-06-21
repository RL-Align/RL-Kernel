# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Tuple

import torch

from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp


class NativeRatioKLOp:
    """PyTorch native fallback for the fused ratio + KL operator."""

    def __init__(self) -> None:
        self._logp = NativeLogpOp()

    def __call__(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        action_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(policy_logits, ref_logits, action_ids, attention_mask, old_logps)

    def _selected_logp(self, logits: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        return self._logp.apply_fp32(logits, action_ids)

    def forward(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        action_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = attention_mask.to(torch.bool)
        vocab_size = policy_logits.size(-1)
        invalid = mask & ((action_ids < 0) | (action_ids >= vocab_size))
        if invalid.any():
            raise ValueError(
                f"action_ids at active (unmasked) positions must be in "
                f"[0, {vocab_size}); found out-of-range ids."
            )
        # Masked positions may hold out-of-range ids; clamp before gather to avoid a CUDA assert.
        safe_action_ids = action_ids.masked_fill(~mask, 0)
        logp_policy = self._selected_logp(policy_logits, safe_action_ids)
        with torch.no_grad():
            logp_ref = self._selected_logp(ref_logits, safe_action_ids)

        delta = (logp_policy - old_logps.float()).masked_fill(~mask, 0.0)
        diff = (logp_ref - logp_policy).masked_fill(~mask, 0.0)
        ratio = torch.exp(delta)
        kl = torch.exp(diff) - diff - 1.0
        return ratio, kl
