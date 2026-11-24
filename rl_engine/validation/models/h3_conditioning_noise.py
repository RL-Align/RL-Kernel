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

The provider's arithmetic depends on its version (``KNOWN_SCALE_NOISE``, by the sha256 of the
method's source). The pinned provider casts a Python-float ``t`` to the sample's dtype before
``1 - t``, as the row does. Diffusers 0.43.0 keeps a Python-float ``t`` in double precision for
``1 - t`` (``1 - 0.999`` then rounds to a different FP32 value), so there only a tensor ``t``
runs on the row. An unknown ``scale_noise`` runs every call on the provider.
"""

from __future__ import annotations

import hashlib
import inspect
from contextlib import contextmanager

import torch

_ROW_TIMESTEP_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# sha256 of ``inspect.getsource`` of ``scale_noise`` -> how it rounds a Python-float ``t``.
KNOWN_SCALE_NOISE = {
    # diffusers@f53d552 (the pinned provider): ``t`` cast to the sample's dtype
    "2c444def0c4570d787c180c1a10676f9362f79b3ad629b96dd6d0bd7d94181c8": "sample_dtype",
    # diffusers 0.43.0 (the verl-omni run environment): a Python float stays double for ``1 - t``
    "2b612801bf39a22835a23dc586f8f28f0c676e0e39075b4952bd8cf60abf64d4": "python_float_double",
}


def scale_noise_variant(scheduler) -> str:
    try:
        source = inspect.getsource(inspect.unwrap(type(scheduler).scale_noise))
    except (OSError, TypeError):  # not file-backed: every call stays on the provider
        return "unknown"
    return KNOWN_SCALE_NOISE.get(hashlib.sha256(source.encode()).hexdigest(), "unknown")


@contextmanager
def bind(scheduler, op):
    """Route ``scheduler.scale_noise`` to ``op`` (a ``conditioning_noise_mix`` op) where bitwise.

    Yields ``{"row": n, "provider": m, "scale_noise": variant}``: the calls each path ran and the
    provider version (``KNOWN_SCALE_NOISE`` or ``unknown``). Restores the method on exit.
    """

    if "scale_noise" in scheduler.__dict__:
        raise RuntimeError("scheduler.scale_noise is already wrapped on the instance")
    original = scheduler.scale_noise
    variant = scale_noise_variant(scheduler)
    calls = {"row": 0, "provider": 0, "scale_noise": variant}

    def row_takes(sample, timestep) -> bool:
        # what the row's validator accepts, and where its arithmetic is the provider's
        if isinstance(timestep, torch.Tensor):
            return (
                timestep.numel() == 1
                and timestep.device == sample.device  # the provider moves it; the row refuses
                and timestep.dtype in _ROW_TIMESTEP_DTYPES
                and not timestep.requires_grad
                and timestep.is_contiguous()
                # the pinned provider casts t to the sample's dtype, as the row does; Diffusers
                # 0.43.0 is only trusted with a t already in that dtype; an unknown version never
                and (
                    variant == "sample_dtype"
                    or (variant == "python_float_double" and timestep.dtype == sample.dtype)
                )
            )
        # a Python float: the row casts it to the sample's dtype, as only the pinned provider does
        return type(timestep) is float and variant == "sample_dtype"

    def scale_noise(sample, timestep, noise):
        same = (
            isinstance(sample, torch.Tensor)
            and isinstance(noise, torch.Tensor)
            and sample.dtype == noise.dtype
            and sample.shape == noise.shape
            and sample.device == noise.device
            and sample.is_contiguous()
            and noise.is_contiguous()
            and row_takes(sample, timestep)
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
