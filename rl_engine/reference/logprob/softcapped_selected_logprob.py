# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""PyTorch references for softcapped selected logprob.

NativeSoftcappedSelectedLogprobOp uses native log_softmax and autograd as the
independent accuracy and performance baseline. ordered_forward/ordered_backward
spell out the fixed addition tree and return inspectable intermediate stages
for arithmetic-path checks. Both references live here, but the native Op does
not call the fixed-order implementation.

The ordered reference must not replace independent log_softmax/autograd or
FP64 accuracy checks. Native torch log/exp and Triton intrinsics need not
produce identical bits; the path tests isolate those primitive boundaries.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from rl_engine.contracts.operators.softcapped_selected_logprob import (
    REDUCTION_CONFIG,
    SUPPORTED_DTYPES,
)
from rl_engine.reference.reduction.ordered_reduce import ordered_reduce


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
    if logits.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"logits must have dtype {SUPPORTED_DTYPES}, got {logits.dtype}.")
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
    softcap within the FP32 exp/sum range.

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
    [0, V). Each forward call requires an explicit softcap; it is a
    positive scalar without its own gradient. Forward validates its FP32
    exp/sum range for V. The Op does not store a softcap value between calls.
    The forward's native operations supply input gradients automatically.
    """

    op_class = "logprob"

    def forward(
        self, logits: torch.Tensor, token_ids: torch.Tensor, *, softcap: float
    ) -> torch.Tensor:
        """Compute with this call's softcap; PyTorch records it in the native graph."""
        return softcapped_selected_logprob(logits, token_ids, softcap)


def ordered_exp_sum(
    exponentials: torch.Tensor, *, tile_size: int = REDUCTION_CONFIG.tile_size
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum already-computed FP32 [M, V] exponentials through the common reference.

    Exposing these values separately lets path tests isolate elementary-function
    differences from the fixed reduction. Elementwise math stays in this operator.

    Returns:
        FP32 tile sums [M, ceil(V / tile_size)] and their FP32 ordered total [M].
    """
    if exponentials.ndim != 2 or exponentials.shape[1] == 0:
        raise ValueError("exponentials must have shape [M, V] with V > 0")
    if exponentials.dtype != torch.float32:
        raise TypeError("exponentials must be FP32")
    result = ordered_reduce(
        exponentials,
        block_size=tile_size,
        max_group_size=REDUCTION_CONFIG.group_size,
        section_count=REDUCTION_CONFIG.section_count,
        combine_fn=torch.add,
        identity=0.0,
        initial=0.0,
        compute_dtype=torch.float32,
    )
    return result.partials, result.total


@dataclass(frozen=True)
class OrderedForward:
    """Inspectable stages; tensors are FP32 and do not carry an autograd graph."""

    tanh_scaled: torch.Tensor
    """[M, V] tanh(x / c), after FP32 input conversion and device-tensor division."""
    scores: torch.Tensor
    """[M, V] FP32 softcapped scores."""
    row_max: torch.Tensor
    """[M] softcap(maximum FP32 logit), with zero canonicalized to positive zero."""
    exponentials: torch.Tensor
    """[M, V] native PyTorch exp(scores - row_max)."""
    tile_sums: torch.Tensor
    """[M, ceil(V / tile_size)] fixed-tree partial sums."""
    sum_exp: torch.Tensor
    """[M] left-to-right accumulation of tile sums, including the initial zero."""
    log_sum_exp_shifted: torch.Tensor
    """[M] native PyTorch log(sum_exp); library bits may differ across backends."""
    output: torch.Tensor
    """[M] (selected score - row maximum) - log_sum_exp_shifted, in that order."""


def selected_from_statistics(
    scores: torch.Tensor,
    token_ids: torch.Tensor,
    row_max: torch.Tensor,
    log_sum_exp_shifted: torch.Tensor,
) -> torch.Tensor:
    """Return FP32 [M] selected logprobs using two separately rounded subtractions.

    scores is FP32 [M, V]; token_ids is int64 [M]. Both statistics are FP32 [M].
    Subtract the row maximum before the shifted log-sum, as in the Triton kernel.
    """
    selected = scores.gather(1, token_ids[:, None]).squeeze(1)
    selected_shifted = selected - row_max
    return selected_shifted - log_sum_exp_shifted


@torch.no_grad()
def ordered_forward(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    softcap: float,
    tile_size: int = REDUCTION_CONFIG.tile_size,
) -> OrderedForward:
    """Evaluate the fixed FP32 path on the same already-quantized inputs.

    This arithmetic reference assumes finite, valid operator inputs. It does not call
    the production validator or implementation. The independent accuracy
    suite is responsible for judging this arithmetic against mathematics.
    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        divisor = torch.tensor(softcap, device=logits.device, dtype=torch.float32)
        logits_f = logits.detach().float()
        # Mirror Triton's maximum pass: positive softcap is monotone, so only
        # the maximum logit needs division/tanh here. Native remains independent.
        max_logit = logits_f.amax(dim=-1)
        row_max = softcap * torch.tanh(torch.div(max_logit, divisor))
        row_max = torch.where(row_max == 0.0, torch.zeros_like(row_max), row_max)
        quotient = torch.div(logits_f, divisor)
        tanh_scaled = torch.tanh(quotient)
        scores = softcap * tanh_scaled
        exponentials = torch.exp(scores - row_max[:, None])
        tile_sums, sum_exp = ordered_exp_sum(exponentials, tile_size=tile_size)
        log_sum_exp_shifted = torch.log(sum_exp)
        output = selected_from_statistics(scores, token_ids, row_max, log_sum_exp_shifted)
    return OrderedForward(
        tanh_scaled=tanh_scaled,
        scores=scores,
        row_max=row_max,
        exponentials=exponentials,
        tile_sums=tile_sums,
        sum_exp=sum_exp,
        log_sum_exp_shifted=log_sum_exp_shifted,
        output=output,
    )


def gradient_from_probability(
    tanh_scaled: torch.Tensor,
    probability: torch.Tensor,
    token_ids: torch.Tensor,
    upstream: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Explicit backward arithmetic, with FP32 intermediates and one final cast.

    There is no reduction in this operator's backward. Receiving probability
    explicitly lets the path tests isolate libdevice.exp from the subsequent adds,
    multiplies and dtype conversion. This is not the independent autograd oracle.
    tanh_scaled and probability are FP32 [M, V]; IDs and upstream are [M].
    Returns [M, V] in output_dtype, matching the original logits storage dtype.
    """
    columns = torch.arange(tanh_scaled.shape[1], device=tanh_scaled.device)
    selected = (columns[None, :] == token_ids[:, None]).float()
    grad_softcapped = upstream.float()[:, None] * (selected - probability)
    derivative = 1.0 - tanh_scaled * tanh_scaled
    return (grad_softcapped * derivative).to(output_dtype)


@torch.no_grad()
def ordered_backward(
    forward: OrderedForward,
    token_ids: torch.Tensor,
    upstream: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Return [M, V] gradients using the matching ordered forward's FP32 statistics.

    token_ids and upstream are [M]; output_dtype is the original logits dtype.
    This explicitly written backward checks the arithmetic path. Native autograd
    remains the independent gradient-accuracy reference.
    """
    shifted = forward.scores - forward.row_max[:, None]
    probability = torch.exp(shifted - forward.log_sum_exp_shifted[:, None])
    return gradient_from_probability(
        forward.tanh_scaled, probability, token_ids, upstream, output_dtype=output_dtype
    )
