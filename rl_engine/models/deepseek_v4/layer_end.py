# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSeek-V4 layer end: mHC head -> final RMSNorm -> lm_head -> raw logprob.

Runs the four steps after the last decoder layer in the order of the
DeepSeek-V4-Flash-0731 reference (``ParallelHead`` and ``Transformer.forward``)
and returns every intermediate, so a consistency check can report the first
stage whose bytes differ (DSV4 RFC task 8, #508).

Precision follows the reference: mHC head and the norm return bf16, lm_head
returns fp32 logits, and the logprob is the fp32 value before any sampling
transform. lm_head is the shared batch-invariant linear (task 9) with bf16
inputs, fp32 accumulation and fp32 output. The reference multiplies in fp32,
but head.weight is BF16 in the checkpoint and the norm output holds bf16
values, and the product of two bf16 values is exact in fp32, so only the
accumulation order differs from the reference. A weight given in another
dtype is rounded to bf16.
"""

from __future__ import annotations

import torch

from rl_engine.runtime.registry import kernel_registry

LAYER_END_STAGES = ("head", "final_norm", "logits", "logprob")


def mhc_layer_end(
    residual: torch.Tensor,
    head_fn: torch.Tensor,
    head_scale: torch.Tensor,
    head_base: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Return ``{stage: output}`` for the stages in ``LAYER_END_STAGES``.

    Shapes: residual [T, n, C], head_fn [n, n*C], head_scale [1], head_base [n],
    norm_weight [C], lm_head_weight [V, C], target_ids [T].
    """
    device = residual.device
    head = kernel_registry.get_op("mhc_head", device=device)(
        residual, head_fn, head_scale, head_base, rms_eps=rms_eps, hc_pre_eps=hc_pre_eps
    )
    final_norm = kernel_registry.get_op("rms_norm", device=device)(head, norm_weight, eps=norm_eps)
    logits = kernel_registry.get_op("batch_invariant_linear", device=device)(
        final_norm, lm_head_weight.to(torch.bfloat16), out_dtype=torch.float32
    )
    logprob = kernel_registry.get_op("batch_invariant_logp", device=device)(logits, target_ids)
    return {"head": head, "final_norm": final_norm, "logits": logits, "logprob": logprob}
