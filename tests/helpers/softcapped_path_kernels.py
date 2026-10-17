# SPDX-License-Identifier: Apache-2.0
"""GPU probes for arithmetic-path tests only; never an accuracy reference.

Import lazily on GPU hosts. The probes expose the production tile-sum helper
and the log/exp intrinsic boundaries, not a second implementation of logprob.
The independent FP32/autograd and FP64 suites do not import this module.
"""

import triton
import triton.language as tl

from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
    _ordered_tile_sum,
)


@triton.jit
def tile_sum_probe(values_ptr, sums_ptr):
    tile = tl.program_id(0).to(tl.int64)
    values = tl.load(values_ptr + tile * 1024 + tl.arange(0, 1024))
    tl.store(sums_ptr + tile, _ordered_tile_sum(values))


@triton.jit
def intrinsic_boundary_probe(
    scores_ptr, sums_ptr, maxima_ptr, logs_ptr, probabilities_ptr, V: tl.constexpr
):
    """Evaluate only tl.log and backward tl.exp on supplied reference stages.

    This isolates library-primitive rounding from the explicit FP32 arithmetic
    tested in PyTorch. It does not validate the accuracy of those intrinsics;
    full independent reference tests retain that responsibility.
    """
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    cols = tile * 1024 + tl.arange(0, 1024)
    mask = cols < V
    scores = tl.load(scores_ptr + row * V + cols, mask=mask, other=0.0)
    maximum = tl.load(maxima_ptr + row)
    logarithm = tl.log(tl.load(sums_ptr + row))
    if tile == 0:
        tl.store(logs_ptr + row, logarithm)
    probability = tl.exp((scores - maximum) - logarithm)
    tl.store(probabilities_ptr + row * V + cols, probability, mask=mask)
