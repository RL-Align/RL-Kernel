# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Readable FP32 reference for softcapped selected logprob."""

import math

import torch
from torch import nn

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_softcap(softcap: float, vocab_size: int) -> None:
    """Keep the existing softcap domain for a positive vocabulary width.

    Rely on type annotations. Require a finite positive FP32-normal cap,
    exp(-softcap) >= 2 * float32.tiny and V * exp(softcap) <= float32.max / 2.
    These bounds come from the original unshifted exp/sum path. Keep the same
    accepted inputs when switching to log_softmax; extending the API is separate.
    """
    limits = torch.finfo(torch.float32)
    if not math.isfinite(softcap) or softcap < limits.tiny:
        raise ValueError("softcap must be finite, positive and at least float32.tiny.")
    max_softcap = min(
        -math.log(2.0 * limits.tiny), math.log(limits.max / 2.0) - math.log(vocab_size)
    )
    if softcap > max_softcap:
        raise ValueError(
            f"softcap={softcap} with V={vocab_size} exceeds the FP32 exp/sum range "
            f"(maximum {max_softcap:.6g})."
        )


def _validate_inputs(logits: torch.Tensor, token_ids: torch.Tensor, softcap: float) -> None:
    """Validate tensors and the FP32 row-sum bound for their vocabulary width."""
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("logits must have shape [M, V] with V > 0.")
    _validate_softcap(softcap, logits.shape[1])
    if logits.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"logits must have dtype {_SUPPORTED_DTYPES}, got {logits.dtype}.")
    if token_ids.shape != logits.shape[:1]:
        raise ValueError("token_ids must have shape [M], matching the logits rows.")
    if token_ids.dtype != torch.int64 or token_ids.device != logits.device:
        raise TypeError("token_ids must have dtype int64 and be on the logits device.")


def softcapped_selected_logprob(
    logits: torch.Tensor, token_ids: torch.Tensor, softcap: float
) -> torch.Tensor:
    """Return one FP32 log-probability per row; PyTorch supplies autograd.

    Inputs: logits [M, V] in FP16/BF16/FP32, and int64 token_ids [M] on the
    same device. V must be positive and each token ID must lie in [0, V).
    Output: selected_logprob [M]. Callers must explicitly pass a positive
    softcap within the FP32 exp/sum range. Only the Op's forward supplies the
    default value of 30.0.

    This reference is exposed by NativeSoftcappedSelectedLogprobOp. The FP32
    divisor lives on the input device: CUDA division by a Python scalar uses
    a rounded reciprocal followed by multiplication instead. A device tensor
    avoids that optimization without calling Triton from the reference.
    FP32 log_softmax normalizes with a row-maximum shift before selecting the
    requested token, avoiding a separately rounded unshifted log-sum.
    Native reductions still differ from Triton's fixed tiled reduction order;
    agreement with the reference uses tolerances, not bitwise equality.
    """
    _validate_inputs(logits, token_ids, softcap)
    softcap = float(softcap)

    # 1. Apply softcap in FP32, without the CPU-scalar reciprocal optimization.
    logits_f = logits.to(torch.float32)
    divisor = torch.tensor(softcap, device=logits.device, dtype=torch.float32)
    scaled_logits = torch.div(logits_f, divisor)
    softcapped = softcap * torch.tanh(scaled_logits)

    # 2. Normalize in FP32 with PyTorch's stable log_softmax.
    log_probs = torch.log_softmax(softcapped, dim=-1)

    # 3. Select one log-probability per row; [M, 1] becomes [M].
    selected_logprob = log_probs.gather(dim=-1, index=token_ids[:, None]).squeeze(-1)
    return selected_logprob


class NativeSoftcappedSelectedLogprobOp(nn.Module):
    """PyTorch softcapped selected logprob with FP32 output and native autograd.

    logits has shape [M, V] and token_ids has shape [M]. Valid token IDs lie in
    [0, V). Each forward call accepts softcap, defaulting to 30.0; it is a
    positive scalar without its own gradient. Forward validates its FP32
    exp/sum range for V. The Op does not store a softcap value between calls.
    The forward's native operations supply input gradients automatically.
    """

    op_class = "logprob"

    def forward(
        self, logits: torch.Tensor, token_ids: torch.Tensor, *, softcap: float = 30.0
    ) -> torch.Tensor:
        """Compute with this call's softcap; PyTorch records it in the native graph."""
        return softcapped_selected_logprob(logits, token_ids, softcap)
