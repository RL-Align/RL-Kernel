# SPDX-License-Identifier: Apache-2.0
"""INK-54: opt-in SGLang unsharded BF16 head adapter (partial overlay)."""

from contextlib import contextmanager
from types import MethodType

import torch

from rl_engine.integrations.common.inkling2_head import raw_head


def _compute_head(processor, hidden_states, lm_head, embedding_bias=None, *, out=None, platform):
    if not processor.use_fp32_lm_head or processor.shard_softcap != 30.0:
        raise RuntimeError("expected Inkling FP32 LM head with shard softcap 30")
    if processor.final_logit_softcapping is not None or processor.logit_scale is not None:
        raise RuntimeError("unexpected additional logit transform")
    if processor.do_tensor_parallel_all_gather or processor.do_tensor_parallel_all_gather_dp_attn:
        raise RuntimeError("initial strict head adapter requires TP1 without DP-attention gather")
    if embedding_bias is not None or hasattr(lm_head, "apply_lora"):
        raise RuntimeError("bias and LoRA are not supported by this head profile")
    if getattr(lm_head, "weight", None) is None or lm_head.weight.dtype != torch.bfloat16:
        raise RuntimeError("expected the converted BF16 unembedding weight")
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    if should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None)):
        raise RuntimeError("quantized head is outside the BF16 profile")
    logits = raw_head(hidden_states, lm_head.weight, platform=platform)
    # The original _get_logits still owns padding exclusion and output-buffer handling.
    capped = 30.0 * torch.tanh(logits / 30.0)
    if out is not None and out.shape == capped.shape and out.dtype == torch.float32:
        out.copy_(capped)
        return out
    return capped


@contextmanager
def bind_sglang_head(processor, *, platform: str):
    """Patch the actual processor instance before graph capture; restore on exit.

    This replaces only _compute_lm_head. It does not make attention, MoE, conv,
    KV state, sampling or TP strict. Use an isolated eager worker for debugging.
    """
    from sglang.srt.models.inkling2 import Inkling2LogitsProcessor

    if type(processor) is not Inkling2LogitsProcessor:
        raise TypeError("expected the pinned Inkling2LogitsProcessor")
    if platform not in ("cuda", "rocm"):
        raise ValueError("expected cuda or rocm")
    if "_compute_lm_head" in vars(processor):
        raise RuntimeError("processor head is already overridden")

    def replacement(self, hidden_states, lm_head, embedding_bias=None, *, out=None):
        return _compute_head(
            self, hidden_states, lm_head, embedding_bias, out=out, platform=platform
        )

    processor._compute_lm_head = MethodType(replacement, processor)
    try:
        yield processor
    finally:
        del processor._compute_lm_head
