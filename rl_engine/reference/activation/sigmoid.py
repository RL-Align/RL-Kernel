# SPDX-License-Identifier: Apache-2.0
"""Length-invariant sigmoid with an overflow-safe analytical backward."""

import torch


class _LengthInvariantSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # Preserve the reference's scalar/vector-tail-independent forward.
        result = 1.0 / (1.0 + torch.exp(-x))
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (result,) = ctx.saved_tensors
        # Differentiating exp(-x) directly produces 0 * inf for x < -88.7.
        return grad_output * result * (1.0 - result)


def length_invariant_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Evaluate sigmoid without the CPU sigmoid scalar-tail rounding path."""
    return _LengthInvariantSigmoid.apply(x)
