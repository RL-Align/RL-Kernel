# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch


class NativeBatchInvariantLogpOp:
    """Batch-invariant selected-token log-probability.

    ``selected_logprob[t] = logits[t, target_ids[t]] - logsumexp(logits[t, :])``

    All reductions run in FP32. The row-wise max -> subtract -> exp -> sum -> log
    pipeline is fully independent per row, so the result for any row depends
    only on that row's logits and target - never on batch size or layout.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        ignore_index: int = -100,
        *,
        validate: bool = True,
    ) -> torch.Tensor:
        return self.apply(logits, target_ids, ignore_index=ignore_index, validate=validate)

    def apply(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        ignore_index: int = -100,
        *,
        validate: bool = True,
    ) -> torch.Tensor:
        self._validate_shapes(logits, target_ids)

        lead_shape = logits.shape[:-1]
        vocab_size = logits.size(-1)

        logits_2d = logits.reshape(-1, vocab_size).float()
        target_1d = target_ids.reshape(-1).to(logits.device, dtype=torch.long)

        selected_logp, _ = self._row_wise_selected_logprob_with_lse(
            logits_2d, target_1d, ignore_index=ignore_index, validate=validate
        )

        return selected_logp.reshape(lead_shape)

    def forward_with_lse(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        ignore_index: int = -100,
        *,
        validate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return selected logprob and the FP32 vocab-domain LSE for diagnostics."""
        self._validate_shapes(logits, target_ids)
        lead_shape = logits.shape[:-1]
        logits_2d = logits.reshape(-1, logits.size(-1)).float()
        target_1d = target_ids.reshape(-1).to(logits.device, dtype=torch.long)
        logp, lse = self._row_wise_selected_logprob_with_lse(
            logits_2d, target_1d, ignore_index=ignore_index, validate=validate
        )
        return logp.reshape(lead_shape), lse.reshape(lead_shape)

    # ---------------------------------------------------------------------- #
    # Core Computation
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _row_wise_selected_logprob_with_lse(
        logits_2d: torch.Tensor,
        target_1d: torch.Tensor,
        *,
        ignore_index: int,
        validate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row selected logprob and LSE with locked reduction order.

        The three reduction steps (max, sum-exp, gather) operate on each row
        independently.  PyTorch's ``max(dim=-1)`` and ``sum(dim=-1)`` iterate
        the vocab dimension in a fixed, deterministic order for a given row
        length, and that order does **not** change when more rows are added
        to the batch.  This is the property that makes the op batch-invariant.

        Accumulation is done entirely in FP32 to avoid half-precision
        catastrophic cancellation during the ``exp(logit - max)`` step.
        """
        vocab_size = logits_2d.size(1)

        valid_mask = target_1d != ignore_index

        if validate:
            valid_targets = target_1d[valid_mask]
            if valid_targets.numel() > 0 and (
                (valid_targets < 0).any() or (valid_targets >= vocab_size).any()
            ):
                bad = valid_targets[(valid_targets < 0) | (valid_targets >= vocab_size)]
                raise ValueError(
                    f"target_ids contains values outside [0, {vocab_size}): {bad.tolist()}"
                )

        safe_target = target_1d.clone()
        safe_target[~valid_mask] = 0

        # logsumexp(z) = log(sum(exp(z - max(z)))) + max(z)
        row_max = logits_2d.max(dim=-1).values
        shifted = logits_2d - row_max.unsqueeze(-1)
        exp_shifted = shifted.exp()
        sum_exp = exp_shifted.sum(dim=-1)
        log_sum_exp = sum_exp.log() + row_max

        row_indices = torch.arange(logits_2d.size(0), device=logits_2d.device)
        selected_logit = logits_2d[row_indices, safe_target]

        selected_logp = selected_logit - log_sum_exp

        selected_logp = selected_logp.where(valid_mask, torch.zeros_like(selected_logp))

        return selected_logp, log_sum_exp

    # ---------------------------------------------------------------------- #
    # Helper
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _validate_shapes(logits: torch.Tensor, target_ids: torch.Tensor) -> None:
        if logits.dim() < 2:
            raise ValueError(
                f"logits must be at least 2-D ([*lead, V]), got shape {tuple(logits.shape)}"
            )
        if logits.shape[:-1] != target_ids.shape:
            raise ValueError(
                f"logits leading shape {tuple(logits.shape[:-1])} must match "
                f"target_ids shape {tuple(target_ids.shape)}"
            )
