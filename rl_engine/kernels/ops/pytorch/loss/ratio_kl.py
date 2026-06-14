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
        # Clamp ids first so masked/pad positions never index out of bounds
        safe_ids = action_ids.clamp(0, logits.size(-1) - 1).long()
        return self._logp.apply_fp32(logits, safe_ids)

    def forward(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        action_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = attention_mask.to(torch.bool)
        logp_policy = self._selected_logp(policy_logits, action_ids)
        with torch.no_grad():
            logp_ref = self._selected_logp(ref_logits, action_ids)

        delta = (logp_policy - old_logps.float()).masked_fill(~mask, 0.0)
        diff = (logp_ref - logp_policy).masked_fill(~mask, 0.0)
        ratio = torch.exp(delta)
        kl = torch.exp(diff) - diff - 1.0
        return ratio, kl
