# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Readable FP32 reference for softcapped selected logprob."""

import torch
from torch import nn

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def softcapped_selected_logprob(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Return one FP32 log-probability per row; PyTorch supplies autograd.

    Inputs: logits [M, V] in FP16/BF16/FP32, and int64 token_ids [M] on the
    same device. V must be positive and each token ID must lie in [0, V).
    Output: selected_logprob [M], with softcap fixed at 30.0 for Gemma.

    This reference is exposed by NativeSoftcappedSelectedLogprobOp. Its
    reduction order is not the Triton kernel's fixed tiled reduction order.
    """
    # 1. Apply the same FP32 arithmetic as final_logit_softcap.
    logits_f = logits.to(torch.float32)
    softcapped = 30.0 * torch.tanh(logits_f / 30.0)

    # 2. Sum exponentials across each row, then take the natural logarithm.
    # Softcap bounds finite values to [-30, 30], so these exponentials and
    # their sum fit in FP32 for Gemma's 262144-token vocabulary.
    exp_softcapped = torch.exp(softcapped)
    sum_exp = exp_softcapped.sum(dim=-1)
    log_sum_exp = torch.log(sum_exp)

    # 3. Select one softcapped score per row; [M, 1] becomes [M].
    selected_softcapped = softcapped.gather(dim=-1, index=token_ids[:, None]).squeeze(-1)

    # 4. log(exp(selected) / sum_exp) = selected - log(sum_exp).
    selected_logprob = selected_softcapped - log_sum_exp
    return selected_logprob


class NativeSoftcappedSelectedLogprobOp(nn.Module):
    """PyTorch softcapped selected logprob with FP32 output and native autograd.

    logits has shape [M, V] and token_ids has shape [M]. Valid token IDs lie in
    [0, V). The forward's native operations supply gradients automatically.
    """

    op_class = "logprob"

    def forward(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] == 0:
            raise ValueError("logits must have shape [M, V] with V > 0.")
        if logits.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"logits must have dtype {_SUPPORTED_DTYPES}, got {logits.dtype}.")
        if token_ids.shape != logits.shape[:1]:
            raise ValueError("token_ids must have shape [M], matching the logits rows.")
        if token_ids.dtype != torch.int64 or token_ids.device != logits.device:
            raise TypeError("token_ids must have dtype int64 and be on the logits device.")
        return softcapped_selected_logprob(logits, token_ids)
