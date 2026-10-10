# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Any

import torch


def _bool_mask(mask: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    return mask.to(device=device, dtype=torch.bool)


def selected_logprobs_reference(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference selected-token logprobs for RL kernel validation."""

    if temperature <= 0.0:
        raise ValueError("temperature must be greater than zero")
    if logits.shape[:-1] != token_ids.shape:
        raise ValueError(
            f"logits leading shape {tuple(logits.shape[:-1])} must match "
            f"token_ids shape {tuple(token_ids.shape)}"
        )
    if mask is not None and mask.shape != token_ids.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match token_ids shape")

    gather_token_ids = token_ids.long()
    active_mask = None
    if mask is not None:
        active_mask = _bool_mask(mask, device=token_ids.device)
        gather_token_ids = gather_token_ids.masked_fill(~active_mask, 0)

    scaled_logits = logits.float() / float(temperature)
    log_probs = torch.log_softmax(scaled_logits, dim=-1)
    selected = torch.gather(log_probs, dim=-1, index=gather_token_ids.unsqueeze(-1)).squeeze(-1)

    if active_mask is not None:
        selected = selected.masked_fill(~active_mask.to(device=selected.device), 0.0)

    return selected.to(dtype=output_dtype)


def masked_sum(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Sum values while ignoring masked-out entries."""

    values_fp32 = values.float()
    if mask is None:
        return values_fp32.sum()
    return values_fp32.masked_fill(~_bool_mask(mask, device=values.device), 0.0).sum()


def active_token_count(
    mask: torch.Tensor | None, values: torch.Tensor | None = None
) -> torch.Tensor:
    """Return the number of active tokens as an fp32 scalar tensor."""

    if mask is None:
        if values is None:
            raise ValueError("values must be provided when mask is None")
        return torch.tensor(values.numel(), device=values.device, dtype=torch.float32)
    return _bool_mask(mask, device=mask.device).sum().to(dtype=torch.float32)


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean values while ignoring masked-out entries."""

    denom = active_token_count(mask, values).clamp_min(eps)
    return masked_sum(values, mask) / denom


def compute_policy_ratio(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute exp(current - old) with masked entries set to zero."""

    ratio = torch.exp(current_logps.float() - old_logps.float())
    if mask is not None:
        ratio = ratio.masked_fill(~_bool_mask(mask, device=ratio.device), 0.0)
    return ratio


def compute_reference_kl(
    current_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the common GRPO/PPO reference KL approximation."""

    diff = ref_logps.float() - current_logps.float()
    kl = torch.exp(diff) - diff - 1.0
    if mask is not None:
        kl = kl.masked_fill(~_bool_mask(mask, device=kl.device), 0.0)
    return kl


def summarize_kernel_drift(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Summarize candidate-vs-reference drift for benchmark/test output."""

    if candidate.shape != reference.shape:
        raise ValueError(
            f"candidate shape {tuple(candidate.shape)} must match reference shape "
            f"{tuple(reference.shape)}"
        )

    diff = (candidate.float() - reference.float()).abs()
    if mask is not None:
        active = _bool_mask(mask, device=diff.device)
        active_diff = diff[active]
        active_count = int(active.sum().item())
    else:
        active_diff = diff.reshape(-1)
        active_count = int(diff.numel())

    if active_count == 0:
        max_abs = 0.0
        mean_abs = 0.0
    else:
        max_abs = float(active_diff.max().item())
        mean_abs = float(active_diff.mean().item())

    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "active_count": active_count,
    }
