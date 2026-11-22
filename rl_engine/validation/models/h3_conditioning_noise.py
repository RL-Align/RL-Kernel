# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The FL2VA / Ref2VA conditioning noise mix on its row (RFC #420 ``conditioning_noise_mix``).

The pipeline noises each conditioning anchor once, before the denoise loop:
``MiniMaxH3PrepareConditionLatentsStep`` calls ``scheduler.scale_noise(condition, 0.999, noise)``
(``t * x0 + (1 - t) * noise``, ``t`` cast to the condition's dtype, FP32 noise). ``bind`` routes
that call to ``conditioning_noise_mix`` (#476), which rounds each operation in the inputs' dtype
exactly as the eager expression does, so the mix is bitwise the provider's.

The row computes in one dtype. When the condition and the noise differ in dtype the provider's
expression mixes two (the products round in the condition's dtype, the sum is promoted), so
that call stays on the provider; ``bind`` counts which calls ran where.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def bind(scheduler, op):
    """Route ``scheduler.scale_noise`` to ``op`` (a ``conditioning_noise_mix`` op) where bitwise.

    Yields ``{"row": n, "provider": m}``, the calls each path ran. Restores the method on exit.
    """

    if "scale_noise" in scheduler.__dict__:
        raise RuntimeError("scheduler.scale_noise is already wrapped on the instance")
    original = scheduler.scale_noise
    calls = {"row": 0, "provider": 0}

    def scale_noise(sample, timestep, noise):
        same = (
            isinstance(sample, torch.Tensor)
            and isinstance(noise, torch.Tensor)
            and sample.dtype == noise.dtype
            and sample.shape == noise.shape
            and sample.device == noise.device
            and sample.is_contiguous()
            and noise.is_contiguous()
            and not (isinstance(timestep, torch.Tensor) and timestep.numel() != 1)
        )
        if not same:
            calls["provider"] += 1
            return original(sample, timestep, noise)
        calls["row"] += 1
        return op(sample, timestep, noise)

    scheduler.scale_noise = scale_noise
    try:
        yield calls
    finally:
        scheduler.__dict__.pop("scale_noise", None)
