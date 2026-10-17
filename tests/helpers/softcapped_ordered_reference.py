# SPDX-License-Identifier: Apache-2.0
"""Executable FP32 arithmetic specification, not an independent accuracy oracle.

Only PyTorch is used here. The vocabulary tile, addition pairs, accumulation
dtype and final subtraction/cast points are explicit. Do not use this helper
as the performance baseline or replace the independent log_softmax/autograd
and FP64 checks with it. Native torch log/exp and Triton intrinsics need not
produce identical bits; the path tests isolate those primitive boundaries.
"""

from dataclasses import dataclass

import torch

TILE_SIZE = 1024
"""Arithmetic tile width; a change requires a new reference and GPU qualification."""


def fixed_tile_sum(values: torch.Tensor) -> torch.Tensor:
    """Apply the declared addition tree to FP32 [..., 1024] values.

    First fold eight adjacent elements sequentially, starting with element 0
    rather than an extra zero. Reduce each section of 32 resulting groups by
    lower-half plus upper-half pairing. Merge four sections in the same way.
    Each eager add produces an FP32 tensor, preserving every rounding point.
    """
    if values.dtype != torch.float32 or values.shape[-1] != TILE_SIZE:
        raise ValueError("fixed_tile_sum requires FP32 values with a final dimension of 1024")
    groups = values.reshape(*values.shape[:-1], 128, 8)
    subtotal = groups[..., 0]
    for offset in range(1, 8):
        subtotal = subtotal + groups[..., offset]

    sections = subtotal.reshape(*values.shape[:-1], 4, 32)
    for distance in (16, 8, 4, 2, 1):
        sections = sections[..., :distance] + sections[..., distance:]
    sections = sections[..., 0]
    pairs = sections[..., :2] + sections[..., 2:]
    return pairs[..., 0] + pairs[..., 1]


def ordered_exp_sum(exponentials: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [M, tiles] partials and [M] totals for FP32 [M, V] exponentials.

    Pad only the final tile with positive zeros. Merge tile partials left to
    right in vocabulary order, starting from FP32 positive zero. Never use
    torch.sum, whose reduction tree is chosen by its implementation.
    """
    if exponentials.ndim != 2 or exponentials.shape[1] == 0:
        raise ValueError("exponentials must have shape [M, V] with V > 0")
    if exponentials.dtype != torch.float32:
        raise TypeError("exponentials must be FP32")
    n_rows, vocab = exponentials.shape
    partials = []
    total = torch.zeros(n_rows, device=exponentials.device, dtype=torch.float32)
    for start in range(0, vocab, TILE_SIZE):
        tile = exponentials[:, start : start + TILE_SIZE]
        if tile.shape[1] < TILE_SIZE:
            tile = torch.nn.functional.pad(tile, (0, TILE_SIZE - tile.shape[1]), value=0.0)
        partial = fixed_tile_sum(tile)
        partials.append(partial)
        total = total + partial
    return torch.stack(partials, dim=1), total


@dataclass(frozen=True)
class OrderedForward:
    """Inspectable stages; tensors are FP32 and do not carry an autograd graph."""

    tanh_scaled: torch.Tensor
    """[M, V] tanh(x / c), after FP32 input conversion and device-tensor division."""
    scores: torch.Tensor
    """[M, V] FP32 softcapped scores."""
    row_max: torch.Tensor
    """[M] whole-row maxima, with zero canonicalized to positive zero."""
    exponentials: torch.Tensor
    """[M, V] native PyTorch exp(scores - row_max)."""
    tile_sums: torch.Tensor
    """[M, ceil(V / 1024)] fixed-tree partial sums."""
    sum_exp: torch.Tensor
    """[M] left-to-right accumulation of tile sums, including the initial zero."""
    log_sum_exp_shifted: torch.Tensor
    """[M] native PyTorch log(sum_exp); not a bitwise oracle for Triton's tl.log."""
    output: torch.Tensor
    """[M] (selected score - row maximum) - log_sum_exp_shifted, in that order."""


def selected_from_statistics(scores, token_ids, row_max, log_sum_exp_shifted):
    """Keep two distinct FP32 subtractions instead of forming an unshifted LSE."""
    selected = scores.gather(1, token_ids[:, None]).squeeze(1)
    selected_shifted = selected - row_max
    return selected_shifted - log_sum_exp_shifted


@torch.no_grad()
def ordered_forward(logits, token_ids, *, softcap: float) -> OrderedForward:
    """Evaluate the fixed FP32 path on the same already-quantized inputs.

    This test helper assumes finite, valid operator inputs. It does not call
    the production validator or implementation. The independent accuracy
    suite is responsible for judging this arithmetic against mathematics.
    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        divisor = torch.tensor(softcap, device=logits.device, dtype=torch.float32)
        quotient = torch.div(logits.detach().float(), divisor)
        tanh_scaled = torch.tanh(quotient)
        scores = softcap * tanh_scaled
        row_max = scores.amax(dim=-1)
        row_max = torch.where(row_max == 0.0, torch.zeros_like(row_max), row_max)
        exponentials = torch.exp(scores - row_max[:, None])
        tile_sums, sum_exp = ordered_exp_sum(exponentials)
        log_sum_exp_shifted = torch.log(sum_exp)
        output = selected_from_statistics(scores, token_ids, row_max, log_sum_exp_shifted)
    return OrderedForward(
        tanh_scaled,
        scores,
        row_max,
        exponentials,
        tile_sums,
        sum_exp,
        log_sum_exp_shifted,
        output,
    )


def gradient_from_probability(tanh_scaled, probability, token_ids, upstream, *, output_dtype):
    """Explicit backward arithmetic, with FP32 intermediates and one final cast.

    There is no reduction in this operator's backward. Receiving probability
    explicitly lets the path tests isolate tl.exp from the subsequent adds,
    multiplies and dtype conversion. This is not the independent autograd oracle.
    """
    columns = torch.arange(tanh_scaled.shape[1], device=tanh_scaled.device)
    selected = (columns[None, :] == token_ids[:, None]).float()
    grad_softcapped = upstream.float()[:, None] * (selected - probability)
    derivative = 1.0 - tanh_scaled * tanh_scaled
    return (grad_softcapped * derivative).to(output_dtype)


@torch.no_grad()
def ordered_backward(forward: OrderedForward, token_ids, upstream, *, output_dtype):
    """Analytic PyTorch backward; autograd remains a separate accuracy check."""
    shifted = forward.scores - forward.row_max[:, None]
    probability = torch.exp(shifted - forward.log_sum_exp_shifted[:, None])
    return gradient_from_probability(
        forward.tanh_scaled, probability, token_ids, upstream, output_dtype=output_dtype
    )
