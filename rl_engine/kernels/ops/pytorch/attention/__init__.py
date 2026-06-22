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
        # Convert FlashAttention layout to PyTorch SDPA layout:
        # (batch, seqlen, nheads, headdim) -> (batch, nheads, seqlen, headdim)
        q_ref = q.transpose(1, 2)
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)

        q_head_num = q_ref.shape[1]
        k_head_num = k_ref.shape[1]
        if k_head_num != v_ref.shape[1]:
            raise ValueError("k and v must have the same number of heads")

        if q_head_num != k_head_num:
            if q_head_num % k_head_num != 0:
                raise ValueError("q heads must be divisible by k/v heads for GQA/MQA")
            repeat = q_head_num // k_head_num
            k_ref = k_ref.repeat_interleave(repeat, dim=1)
            v_ref = v_ref.repeat_interleave(repeat, dim=1)

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
