# SPDX-License-Identifier: Apache-2.0
"""Capture full sampling support without sorting the vocabulary twice.

Only the boolean support is reused. Strict selected log probabilities are
still computed independently from the raw RL-Kernel logits.

Both rely on vLLM applying top-k/top-p with the ascending sort that
``rl_engine.ops.sampling.policy.sampling_keep_mask`` reproduces, which
``install_sort_contract`` enforces.
"""
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import torch

_CAPTURE = ContextVar("rl_kernel_sampling_support", default=None)
_MARKER = "_rl_kernel_complete_support_capture"
_SORT_MARKER = "_rl_kernel_sort_top_k_top_p"


def _modules_binding(function):
    """Loaded modules whose ``apply_top_k_top_p`` global is ``function``.

    vLLM modules import the dispatcher by name (in 0.30: the rejection sampler,
    the GPU worker's ``states`` and ``sampler``, the watermarking sampler and a
    diffusion model), so replacing the attribute on ``topk_topp_sampler`` alone
    does not reach the ones already loaded. Reading ``__dict__`` avoids
    triggering a module-level ``__getattr__``.
    """
    found = []
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if isinstance(namespace, dict) and namespace.get("apply_top_k_top_p") is function:
            found.append(module)
    return found


def install_sort_contract():
    """Make vLLM's top-k/top-p use its ascending-sort implementation.

    From vLLM 0.17, ``apply_top_k_top_p`` dispatches to a pivot-based Triton
    kernel (for 8 or more rows; for every row from 0.30). Its top-p cut differs
    from the sort's (mostly by keeping a few more tokens), and it depends on the
    batch.
    ``sampling_keep_mask``, which scores the rollout and the trainer, reproduces
    the sort, so vLLM has to sample from the support it produces.

    Returns True when the dispatcher now uses the sort.
    """
    from vllm.v1.sample.ops import topk_topp_sampler

    dispatcher = topk_topp_sampler.apply_top_k_top_p
    if getattr(dispatcher, _SORT_MARKER, False):
        return True
    if getattr(dispatcher, _MARKER, False):
        raise RuntimeError("install the sort top-k/top-p contract before the support capture")
    if not hasattr(topk_topp_sampler, "apply_top_k_top_p_pytorch"):
        # vLLM 0.16 and older have no Triton dispatcher: the function is the sort.
        return False

    @wraps(dispatcher)
    def sort_top_k_top_p(logits, k, p):
        if p is None and k is None:
            return logits
        # Look up at call time so later wrappers of the sort itself still apply.
        return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)

    setattr(sort_top_k_top_p, _SORT_MARKER, dispatcher)
    # Every loaded module that imported the dispatcher by name, including
    # topk_topp_sampler itself. Modules loaded later bind the replacement when
    # they import it.
    for module in _modules_binding(dispatcher):
        module.apply_top_k_top_p = sort_top_k_top_p
    return True


def install_capture():
    from vllm.v1.sample.ops import topk_topp_sampler

    if getattr(topk_topp_sampler.apply_top_k_top_p, _MARKER, False):
        return
    # The captured support must be the one sampling_keep_mask reproduces.
    install_sort_contract()
    original = topk_topp_sampler.apply_top_k_top_p

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
