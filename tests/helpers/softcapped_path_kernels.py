# SPDX-License-Identifier: Apache-2.0
"""Logprob arithmetic-path probes; never an independent accuracy reference."""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def softcap_before_max_probe(
    logits_ptr,
    maxima_ptr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    """Retain the original per-element-softcap maximum for an optimization A/B check.

    Do not call the production maximum kernels: this intentionally softcaps every
    element before reducing, whereas production now softcaps the maximum logit.
    The probe checks unchanged bits, not independent mathematical accuracy.
    """
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_V)
    maximum = tl.full((), float("-inf"), tl.float32)
    for start in range(0, V, BLOCK_V):
        offsets = start + cols
        mask = offsets < V
        logits = tl.load(logits_ptr + row * V + offsets, mask=mask, other=0.0).to(tl.float32)
        scores = SOFTCAP * libdevice.tanh(tl.div_rn(logits, SOFTCAP))
        maximum = tl.maximum(maximum, tl.max(tl.where(mask, scores, float("-inf")), axis=0))
    maximum = tl.where(maximum == 0.0, 0.0, maximum)
    tl.store(maxima_ptr + row, maximum)


@triton.jit
def intrinsic_boundary_probe(
    scores_ptr,
    sums_ptr,
    maxima_ptr,
    logs_ptr,
    probabilities_ptr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Evaluate only libdevice.log/exp on supplied reference stages.

    This isolates library-primitive rounding from the explicit FP32 arithmetic
    tested in PyTorch. It does not validate the accuracy of those intrinsics;
    full independent reference tests retain that responsibility.
    """
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    cols = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = cols < V
    scores = tl.load(scores_ptr + row * V + cols, mask=mask, other=0.0)
    maximum = tl.load(maxima_ptr + row)
    logarithm = libdevice.log(tl.load(sums_ptr + row))
    if tile == 0:
        tl.store(logs_ptr + row, logarithm)
    probability = libdevice.exp((scores - maximum) - logarithm)
    tl.store(probabilities_ptr + row * V + cols, probability, mask=mask)
