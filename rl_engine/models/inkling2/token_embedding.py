# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini token embedding contract (INK-06)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

PADDED_VOCAB_SIZE = 201_024
VALID_VOCAB_SIZE = 200_058


def token_embedding(
    token_ids: torch.Tensor,
    weight: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Gather padded-vocabulary rows with explicit input and output dtypes.

    Token IDs may arrive as int32 or int64 and are converted to int64 for the
    gather. The embedding table controls the default output dtype. Padded rows
    are valid rows when they exist in ``weight`` and are preserved verbatim.
    """
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("token ids must be int32 or int64")
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("embedding weight must be a floating-point [vocab, hidden] tensor")
    if token_ids.device != weight.device:
        raise ValueError("token ids and embedding weight must share a device")

    ids = token_ids.to(dtype=torch.int64)
    if ids.numel():
        min_id = int(ids.min().item())
        max_id = int(ids.max().item())
        if min_id < 0 or max_id >= weight.shape[0]:
            raise ValueError(
                f"token ids must be in [0, {weight.shape[0] - 1}], got [{min_id}, {max_id}]"
            )

    output = F.embedding(ids, weight)
    if output_dtype is not None:
        output = output.to(dtype=output_dtype)
    return output
