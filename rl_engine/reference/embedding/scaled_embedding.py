# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""PyTorch reference for the Gemma scaled token embedding.

``out = weight[token_ids] * s`` where ``s`` is ``embed_scale`` rounded to the weight
dtype first, exactly as HF ``Gemma4TextScaledWordEmbedding`` and vLLM do. For bf16 and
``hidden = 5376`` that makes ``s = 73.5``, not ``sqrt(5376) = 73.32``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from rl_engine.reference.embedding.embedding import NativeEmbeddingOp


def embedding_scale_for_dtype(embed_scale: float, dtype: torch.dtype) -> float:
    """Round the scale the way HF does before multiplying: an fp32 scalar cast to ``dtype``."""
    return torch.tensor(float(embed_scale)).to(dtype).float().item()


class _NativeScaledTokenEmbeddingFn(torch.autograd.Function):
    """Forward is bitwise HF: the fp32 product is exact for bf16/fp16 rows, so the cast is the
    only rounding. Backward follows the Triton reducer's fixed order, not HF autograd.
    """

    @staticmethod
    def forward(ctx, token_ids: Tensor, weight: Tensor, scale: float) -> Tensor:
        ctx.save_for_backward(token_ids)
        ctx.scale = scale
        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        rows = NativeEmbeddingOp._embedding(token_ids, weight, output_dtype=torch.float32)
        return (rows * scale).to(weight.dtype)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        from rl_engine.backends.cuda.embedding.embedding import _deterministic_embedding_grad_weight

        (token_ids,) = ctx.saved_tensors
        grad_weight = None
        if ctx.needs_input_grad[1]:
            hidden = ctx.weight_shape[1]
            grad_weight = _deterministic_embedding_grad_weight(
                token_ids.reshape(-1),
                grad_out.reshape(-1, hidden).float() * ctx.scale,
                weight_shape=ctx.weight_shape,
                weight_dtype=ctx.weight_dtype,
            )
        return None, grad_weight, None


class NativeScaledTokenEmbeddingOp(torch.nn.Module):
    """``out = weight[token_ids] * round_to(weight.dtype, embed_scale)``, as HF Gemma 4."""

    op_class = "elementwise"
    is_batch_invariant = True

    def forward(self, token_ids: Tensor, weight: Tensor, embed_scale: float) -> Tensor:
        scale = embedding_scale_for_dtype(embed_scale, weight.dtype)
        return _NativeScaledTokenEmbeddingFn.apply(token_ids.long(), weight, scale)

    def forward_fp32(self, token_ids: Tensor, weight: Tensor, embed_scale: float) -> Tensor:
        # Upcast a table of the touched rows, not the gathered rows, so autograd reduces
        # fp32 gradients and casts once (the backward contract) without copying the vocab.
        unique_ids, inverse = torch.unique(token_ids.long(), return_inverse=True)
        rows = F.embedding(inverse, weight.index_select(0, unique_ids).float())
        return rows * embedding_scale_for_dtype(embed_scale, weight.dtype)


__all__ = ["NativeScaledTokenEmbeddingOp", "embedding_scale_for_dtype"]
