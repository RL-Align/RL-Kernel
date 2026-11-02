# SPDX-License-Identifier: Apache-2.0
"""Capture full sampling support without sorting the vocabulary twice.

Only the boolean support is reused. Strict selected log probabilities are
still computed independently from the raw RL-Kernel logits.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import torch

_CAPTURE = ContextVar("rl_kernel_sampling_support", default=None)
_MARKER = "_rl_kernel_complete_support_capture"


def install_capture():
    from vllm.v1.sample.ops import topk_topp_sampler

    original = topk_topp_sampler.apply_top_k_top_p
    if getattr(original, _MARKER, False):
        return

    @wraps(original)
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        capture = _CAPTURE.get()
        if capture is not None:
            capture["mask"] = torch.isfinite(result)
        return result

    setattr(wrapped, _MARKER, True)
    topk_topp_sampler.apply_top_k_top_p = wrapped


def supports_capture(sampler):
    operation = getattr(getattr(sampler, "topk_topp_sampler", None), "apply_top_k_top_p", None)
    return bool(getattr(operation, _MARKER, False))


@contextmanager
def capture_support(enabled):
    result = {}
    token = _CAPTURE.set(result if enabled else None)
    try:
        yield result
    finally:
        _CAPTURE.reset(token)
