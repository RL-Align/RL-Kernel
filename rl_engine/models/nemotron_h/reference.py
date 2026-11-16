# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 oracle for the Nemotron-H embedding and LM-head logprob.

This module deliberately shares no code with the strict CUDA path: it upcasts
the BF16 inputs to FP64 and evaluates the reference math with plain PyTorch
ops. It is the correctness oracle (``allclose``); bitwise parity is judged
between two strict paths, never against this module.
"""
from __future__ import annotations

from typing import Optional

import torch

_ROW_CHUNK = 256


def embedding_reference(token_ids: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
    """Untied lookup with no embedding scaling; a pure row copy."""

    flat = embed_weight.index_select(0, token_ids.reshape(-1).long())
    return flat.reshape(*token_ids.shape, embed_weight.size(1))


def _per_row_temperature(
    temperature: Optional[torch.Tensor], rows: int, device: torch.device
) -> Optional[torch.Tensor]:
    if temperature is None:
        return None
    temp = temperature.to(device=device, dtype=torch.float64).reshape(-1)
    return temp.expand(rows) if temp.numel() == 1 else temp


def _scaled_logits_fp64(
    hidden_rows: torch.Tensor,
    weight_fp64: torch.Tensor,
    temp_rows: Optional[torch.Tensor],
) -> torch.Tensor:
    logits = hidden_rows.double() @ weight_fp64.t()
    if temp_rows is not None:
        logits = logits / temp_rows[:, None]
    return logits


def selected_logprob_fp64(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    temperature: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP64 ``(logp, lse)`` of ``log_softmax(hidden @ W^T / T)[target]``."""

    lead_shape = hidden.shape[:-1]
    rows = hidden.reshape(-1, hidden.size(-1))
    target = target_ids.reshape(-1).long()
    weight = lm_head_weight.double()
    temp = _per_row_temperature(temperature, rows.size(0), rows.device)
    logp = torch.empty(rows.size(0), dtype=torch.float64, device=rows.device)
    lse = torch.empty_like(logp)
    for start in range(0, rows.size(0), _ROW_CHUNK):
        end = min(start + _ROW_CHUNK, rows.size(0))
        chunk_temp = None if temp is None else temp[start:end]
        logits = _scaled_logits_fp64(rows[start:end], weight, chunk_temp)
        chunk_lse = torch.logsumexp(logits, dim=-1)
        picked = logits.gather(1, target[start:end, None]).squeeze(1)
        lse[start:end] = chunk_lse
        logp[start:end] = picked - chunk_lse
    return logp.reshape(lead_shape), lse.reshape(lead_shape)


def selected_logprob_vjp_fp64(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    grad_logp: torch.Tensor,
    *,
    temperature: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP64 ``(grad_hidden, grad_weight)`` of ``sum(grad_logp * logp)``.

    ``d logp / d logits = (onehot(target) - softmax(logits / T)) / T``.
    """

    rows = hidden.reshape(-1, hidden.size(-1))
    target = target_ids.reshape(-1).long()
    grad = grad_logp.reshape(-1).double()
    weight = lm_head_weight.double()
    temp = _per_row_temperature(temperature, rows.size(0), rows.device)
    grad_hidden = torch.empty(rows.shape, dtype=torch.float64, device=rows.device)
    grad_weight = torch.zeros(weight.shape, dtype=torch.float64, device=weight.device)
    for start in range(0, rows.size(0), _ROW_CHUNK):
        end = min(start + _ROW_CHUNK, rows.size(0))
        chunk_temp = None if temp is None else temp[start:end]
        hidden_chunk = rows[start:end].double()
        logits = _scaled_logits_fp64(hidden_chunk, weight, chunk_temp)
        dlogits = -torch.softmax(logits, dim=-1)
        dlogits[torch.arange(end - start, device=rows.device), target[start:end]] += 1.0
        dlogits = dlogits * grad[start:end, None]
        if chunk_temp is not None:
            dlogits = dlogits / chunk_temp[:, None]
        grad_hidden[start:end] = dlogits @ weight
        grad_weight += dlogits.t() @ hidden_chunk
    return grad_hidden.reshape(hidden.shape), grad_weight
