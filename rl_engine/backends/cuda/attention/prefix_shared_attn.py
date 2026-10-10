# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch
import torch.nn.functional as F

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger


class PrefixSharedAttentionOp:
    """
    Prefix-Shared Fused Attention optimized for GRPO workloads.

    In GRPO, multiple generated responses (G) share the exact same prompt prefix.
    This kernel avoids redundant memory reads for the prompt's KV cache by loading
    it into shared memory once and broadcasting it across the G queries.
    """

    def __init__(self):
        self.has_hardware_op = False

        if _EXT_AVAILABLE and hasattr(_C, "prefix_shared_attention"):
            self.op = _C.prefix_shared_attention
            self.has_hardware_op = True
            logger.info("Successfully linked to RL-Kernel _C.prefix_shared_attention.")
        else:
            logger.warning(
                "RL-Kernel _C.prefix_shared_attention is unavailable. "
                "PrefixSharedAttentionOp will fallback to native "
                "F.scaled_dot_product_attention (Slow)."
            )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prefix-shared attention forward pass.

        Args:
            q: Query tensor of shape [bs, G, seq_len_q, head_dim]
            k: Shared Key tensor of shape [bs, seq_len_kv, head_dim]
            v: Shared Value tensor of shape [bs, seq_len_kv, head_dim]

        Returns:
            Output tensor of shape [bs, G, seq_len_q, head_dim]
        """
        assert q.dtype == torch.bfloat16, "Only BF16 is supported for this kernel."
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA device"

        # 1. Hardware-accelerated Path
        if self.has_hardware_op and q.shape[-1] == 128:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            return self.op(q, k, v)

        # 2. Fallback Path
        bs, G, len_q, dim = q.shape
        _, len_kv, _ = k.shape

        k_expanded = k.unsqueeze(1).expand(-1, G, -1, -1).reshape(bs * G, 1, len_kv, dim)
        v_expanded = v.unsqueeze(1).expand(-1, G, -1, -1).reshape(bs * G, 1, len_kv, dim)
        q_reshaped = q.reshape(bs * G, 1, len_q, dim)

        out = F.scaled_dot_product_attention(q_reshaped, k_expanded, v_expanded)
        return out.view(bs, G, len_q, dim)
