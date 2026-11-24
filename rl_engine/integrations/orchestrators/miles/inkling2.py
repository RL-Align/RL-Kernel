# SPDX-License-Identifier: Apache-2.0
"""INK-56: opt-in Miles/TorchTitan unembedding overlay (partial integration)."""

from contextlib import contextmanager
from types import MethodType

from rl_engine.integrations.common.inkling2_head import raw_head


@contextmanager
def bind_miles_head(model, *, platform: str):
    """Replace the live Fp32OutLinear.forward before compiling/sharding the model.

    The original model retains its valid-vocabulary slicing and FP32 softcap;
    GRPO, optimizer, model parameters and checkpoint logic are not replaced.
    This is a single-device unsharded debug boundary, not a FSDP2/EP overlay.
    """
    from miles.backends.torchtitan_utils.models.inkling2.model import (
        Fp32OutLinear,
        Inkling2Model,
    )

    if type(model) is not Inkling2Model or type(model.lm_head) is not Fp32OutLinear:
        raise TypeError("expected the pinned native Miles Inkling2 model and head")
    if platform not in ("cuda", "rocm"):
        raise ValueError("expected cuda or rocm")
    if model.unpadded_vocab_size != 200058 or model.final_logit_softcapping != 30.0:
        raise RuntimeError("unexpected checkpoint output semantics")
    if model._skip_lm_head:
        raise RuntimeError("head bypass is incompatible with this overlay")
    head = model.lm_head
    if "forward" in vars(head):
        raise RuntimeError("head is already overridden")

    def replacement(self, x):
        flat = x.reshape(-1, x.shape[-1])
        result = raw_head(flat, self.weight, platform=platform)
        return result.reshape(*x.shape[:-1], result.shape[-1])

    head.forward = MethodType(replacement, head)
    try:
        yield model
    finally:
        del head.forward
