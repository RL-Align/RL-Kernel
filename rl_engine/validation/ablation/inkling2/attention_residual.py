# SPDX-License-Identifier: Apache-2.0
"""INK-48: explicit negative controls for block attention residual mixing."""

import torch

from rl_engine.models.inkling2.attn_res_mix import block_attention_residual
from rl_engine.ops.norm.fixed_tree import fixed_sum_last


def attention_residual_probe(blocks, partial, query, *, fault):
    if fault == "reverse_sources":
        return block_attention_residual([partial, *reversed(blocks)], None, query)
    if fault != "skip_value_norm":
        raise ValueError("unknown attention residual probe")
    values = [source.float() for source in (*blocks, partial)]
    hidden = values[0].shape[-1]
    inv = [torch.rsqrt(fixed_sum_last(value.square()) / hidden + 1e-6) for value in values]
    if len(values) == 1:
        return values[0]
    q = query.float() * (hidden**-0.5)
    logits = [fixed_sum_last(value * q) * scale for value, scale in zip(values, inv, strict=True)]
    peak = logits[0]
    for score in logits[1:]:
        peak = torch.maximum(peak, score)
    exp = [(score - peak).exp() for score in logits]
    denominator = exp[0]
    for term in exp[1:]:
        denominator = denominator + term
    # Same key normalization, softmax and fold; only value normalization is disabled.
    result = (exp[0] / denominator) * values[0]
    for term, value in zip(exp[1:], values[1:], strict=True):
        result = result + (term / denominator) * value
    return result
