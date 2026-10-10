# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch
import torch.nn.functional as F


class NativeEmbeddingOp(torch.nn.Module):
    """
    Pure PyTorch native token-embedding reference.
    out = weight[token_ids]   (a plain row gather, no accumulation)

    Maps integer token ids to their hidden-state rows. For Qwen3-8B the
    weight is the input embedding table ``[vocab=151936, hidden=4096]`` and
    is *independent* from the lm_head weight (``tie_word_embeddings=false``).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Canonical entry: gather in the weight's native dtype, then cast the
        gathered rows to weight.dtype (a no-op here, kept for symmetry).
        This is the dtype-behavior path used as the Axis-B accuracy candidate.
        """
        return self._embedding(token_ids, weight, output_dtype=weight.dtype)

    def forward_fp32(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Ground-truth: native-dtype gather, then upcast the result to fp32."""
        return self._embedding(token_ids, weight, output_dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _embedding(
        token_ids: torch.Tensor,
        weight: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        # Embedding is a lossless row gather (pure indexing, no arithmetic), so
        # gathering in the weight's native dtype and upcasting the small result
        # is bitwise-identical to upcasting the whole table first -- but it never
        # allocates a multi-GB fp32 copy of the full vocab table just for a tiny
        # lookup. Only the gathered rows are upcast.
        out = F.embedding(token_ids.long(), weight)
        return out.to(output_dtype)
