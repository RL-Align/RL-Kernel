# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Gemma scaled token embedding on the shared Triton embedding kernels."""

from __future__ import annotations

import torch

from rl_engine.backends.shared.triton.embedding.embedding import (
    _require_accelerator_tensors,
    _TritonEmbeddingFunction,
)
from rl_engine.reference.embedding.scaled_embedding import embedding_scale_for_dtype


class TritonScaledTokenEmbeddingOp:
    """``weight[token_ids] * round_to(weight.dtype, embed_scale)`` with the embedding VJP."""

    op_class = "elementwise"
    is_batch_invariant = True

    def __call__(
        self, token_ids: torch.Tensor, weight: torch.Tensor, embed_scale: float
    ) -> torch.Tensor:
        return self.forward(token_ids, weight, embed_scale)

    def forward(
        self, token_ids: torch.Tensor, weight: torch.Tensor, embed_scale: float
    ) -> torch.Tensor:
        _require_accelerator_tensors(token_ids, weight, "TritonScaledTokenEmbeddingOp")
        scale = embedding_scale_for_dtype(embed_scale, weight.dtype)
        return _TritonEmbeddingFunction.apply(token_ids, weight, scale, "scaled_token_embedding")


__all__ = ["TritonScaledTokenEmbeddingOp"]
