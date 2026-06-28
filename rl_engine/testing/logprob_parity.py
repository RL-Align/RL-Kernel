# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Any, Optional

import torch

from rl_engine.testing.reference_ops import selected_logprobs_reference, summarize_kernel_drift


def make_padded_batch_layout(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    destination_rows: torch.Tensor,
    padded_batch_size: Optional[int] = None,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place completion rows into a larger padded batch layout."""

    if logits.ndim < 2:
        raise ValueError("logits must have at least batch and vocab dimensions")
    if logits.shape[:-1] != token_ids.shape:
        raise ValueError("logits leading shape must match token_ids shape")
    if mask.shape != token_ids.shape:
        raise ValueError("mask shape must match token_ids shape")

    vocab_size = int(logits.shape[-1])
    if not 0 <= int(pad_token_id) < vocab_size:
        raise ValueError("pad_token_id must be within the logits vocabulary range")

    source_batch = int(logits.shape[0])
    rows = destination_rows.to(device=logits.device, dtype=torch.long).reshape(-1)
    if rows.numel() != source_batch:
        raise ValueError("destination_rows must contain one destination per source row")
    if rows.numel() and int(rows.min().item()) < 0:
        raise ValueError("destination_rows must be non-negative")
    if rows.unique().numel() != rows.numel():
        raise ValueError("destination_rows must not contain duplicates")

    resolved_batch = int(padded_batch_size) if padded_batch_size is not None else source_batch
    if rows.numel() and int(rows.max().item()) >= resolved_batch:
        raise ValueError("destination_rows contains a row outside padded_batch_size")
    if resolved_batch < source_batch:
        raise ValueError("padded_batch_size must be at least the source batch size")

    out_shape = (resolved_batch,) + tuple(logits.shape[1:])
    token_shape = (resolved_batch,) + tuple(token_ids.shape[1:])

    padded_logits = torch.zeros(out_shape, device=logits.device, dtype=logits.dtype)
    padded_token_ids = torch.full(
        token_shape,
        int(pad_token_id),
        device=token_ids.device,
        dtype=token_ids.dtype,
    )
    padded_mask = torch.zeros(token_shape, device=mask.device, dtype=torch.bool)

    padded_logits[rows] = logits
    padded_token_ids[rows] = token_ids
    padded_mask[rows] = mask.to(dtype=torch.bool)
    return padded_logits, padded_token_ids, padded_mask


def compare_selected_logprob_layouts(
    reference_logits: torch.Tensor,
    reference_token_ids: torch.Tensor,
    reference_mask: torch.Tensor,
    candidate_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    candidate_rows: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Compare selected logprobs for identical rows under different batch layouts."""

    reference = selected_logprobs_reference(
        reference_logits,
        reference_token_ids,
        mask=reference_mask,
        output_dtype=output_dtype,
    )
    candidate = selected_logprobs_reference(
        candidate_logits,
        candidate_token_ids,
        mask=candidate_mask,
        output_dtype=output_dtype,
    )
    rows = candidate_rows.to(device=candidate.device, dtype=torch.long).reshape(-1)
    if rows.numel() != int(reference.shape[0]):
        raise ValueError("candidate_rows must contain one candidate row per reference row")
    if rows.numel() and int(rows.min().item()) < 0:
        raise ValueError("candidate_rows must be non-negative")
    if rows.unique().numel() != rows.numel():
        raise ValueError("candidate_rows must not contain duplicates")
    if rows.numel() and int(rows.max().item()) >= int(candidate.shape[0]):
        raise ValueError("candidate_rows contains a row outside the candidate batch")

    restored = candidate[rows]
    return summarize_kernel_drift(restored, reference, reference_mask)
