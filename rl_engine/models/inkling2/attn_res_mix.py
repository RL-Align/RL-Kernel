# SPDX-License-Identifier: Apache-2.0
"""INK-08: portable fixed-order Block Attention Residuals arithmetic."""

from collections.abc import Sequence

import torch

from rl_engine.ops.norm.fixed_tree import fixed_sum_last


def block_attention_residual(
    blocks: Sequence[torch.Tensor],
    partial: torch.Tensor | None,
    query: torch.Tensor,
    *,
    partial_addend: torch.Tensor | None = None,
) -> torch.Tensor:
    """Read completed blocks then partial; normalize keys AND values in FP32.

    The reduction graph is explicit on CPU and CUDA/ROCm. The hidden-axis
    dot uses a fixed tree and depth uses a left fold. The output is FP32.
    This is a portable candidate, not a qualified engine binding or fused
    kernel. Parameter VJP across different token partitions is not certified.
    """
    sources = tuple(blocks) + (() if partial is None else (partial,))
    if not sources or sources[0].ndim != 2 or sources[0].shape[-1] == 0:
        raise ValueError("expected nonempty [tokens, hidden] residual sources")
    first = sources[0]
    hidden = first.shape[-1]
    if query.shape != (hidden,) or query.device != first.device:
        raise ValueError("query must be [hidden] on the residual device")
    if any(t.shape != first.shape or t.device != first.device for t in sources):
        raise ValueError("residual sources must share shape and device")
    if partial_addend is not None:
        if partial is None or partial_addend.shape != first.shape:
            raise ValueError("partial addend requires a matching partial residual")
        if partial_addend.device != first.device:
            raise ValueError("partial addend must share the residual device")
    if any(t.dtype not in (torch.float32, torch.bfloat16) for t in (*sources, query)):
        raise TypeError("residual sources and query must be BF16 or FP32")
    if partial_addend is not None and partial_addend.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("partial addend must be BF16 or FP32")

    values = [source.float() for source in sources]
    if partial_addend is not None:
        values[-1] = values[-1] + partial_addend.float()
    inv = [torch.rsqrt(fixed_sum_last(value.square()) / hidden + 1e-6) for value in values]
    normalized = [value * scale for value, scale in zip(values, inv, strict=True)]
    if len(values) == 1:
        return normalized[0]

    q = query.float() * (hidden**-0.5)
    logits = [fixed_sum_last(value * q) * scale for value, scale in zip(values, inv, strict=True)]
    # Stable depth softmax; the same source order is used in the weighted fold.
    peak = logits[0]
    for score in logits[1:]:
        peak = torch.maximum(peak, score)
    exp = [(score - peak).exp() for score in logits]
    denominator = exp[0]
    for term in exp[1:]:
        denominator = denominator + term
    result = (exp[0] / denominator) * normalized[0]
    for term, value in zip(exp[1:], normalized[1:], strict=True):
        result = result + (term / denominator) * value
    return result
