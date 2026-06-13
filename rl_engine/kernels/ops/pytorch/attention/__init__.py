# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch
import torch.nn.functional as F


class NativeAttentionOp:
    """PyTorch SDPA fallback for FlashAttention-layout tensors."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        q_ref = q.transpose(1, 2)
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_ref,
            k_ref,
            v_ref,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )
        return out.transpose(1, 2)


__all__ = ["NativeAttentionOp"]
