# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSeek-V4 layer end: mHC head -> final RMSNorm -> lm_head -> raw logprob.

Runs the four steps after the last decoder layer in the order of the
DeepSeek-V4-Flash-0731 reference (``ParallelHead`` and ``Transformer.forward``)
and returns every intermediate, so a consistency check can report the first
stage whose bytes differ (DSV4 RFC task 8, #508).

The caller names the backend, as in rl_engine/ops/norm/rmsnorm_residual.py; an
unavailable backend raises and nothing falls back:

- ``"rocm"``: RL-Kernel's Triton mHC head (gfx942 only), task 5's Triton
  ``rmsnorm_residual``, the batch-invariant Triton linear (task 9, gfx942 only)
  and the Triton batch-invariant logprob.
- ``"reference"``: the PyTorch mHC head of rl_engine/ops/mhc/training.py (on CPU,
  or on a gfx942 / gfx950 GPU), task 5's serial fp32 reference, an fp32
  ``F.linear`` and the PyTorch batch-invariant logprob.

Precision follows the reference: mHC head and the norm return bf16, lm_head
returns fp32 logits, and the logprob is the fp32 value before any sampling
transform. lm_head multiplies bf16 inputs with fp32 accumulation and fp32
output. The reference multiplies in fp32, but head.weight is BF16 in the
checkpoint and the norm output holds bf16 values, and the product of two bf16
values is exact in fp32, so only the accumulation order differs from the
reference. The final norm takes task 5's contract as it is: an fp32
``norm_weight`` and ``norm_eps`` = 1e-6; its residual copy is not used. Inputs
outside these dtypes raise; nothing is cast. A target id outside the vocabulary
that is not the ignore index (-100) raises.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F

from rl_engine.contracts.operators.rmsnorm_residual import EPS as NORM_EPS
from rl_engine.ops.mhc.training import mhc_head
from rl_engine.ops.norm.rmsnorm_residual import rmsnorm_residual

Backend = Literal["reference", "rocm"]

LAYER_END_STAGES = ("head", "final_norm", "logits", "logprob")


def _reference_linear(x, weight, *, out_dtype):
    return F.linear(x.float(), weight.float()).to(out_dtype)


@lru_cache(maxsize=None)
def _lm_head_and_logprob(backend: Backend):
    """The lm_head linear and the logprob op of ``backend``."""
    if backend == "rocm":
        from rl_engine.backends.rocm.gemm.triton_gemm import TritonBatchInvariantLinearOp
        from rl_engine.backends.shared.triton.logprob.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )

        return TritonBatchInvariantLinearOp(), TritonBatchInvariantLogpOp()
    if backend == "reference":
        from rl_engine.reference.logprob.batch_invariant_logp import NativeBatchInvariantLogpOp

        return _reference_linear, NativeBatchInvariantLogpOp()
    raise ValueError(f"unknown layer-end backend {backend!r}; want 'reference' or 'rocm'")


def mhc_layer_end(
    residual: torch.Tensor,
    head_fn: torch.Tensor,
    head_scale: torch.Tensor,
    head_base: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    backend: Backend,
    norm_eps: float = NORM_EPS,
) -> dict[str, torch.Tensor]:
    """Return ``{stage: output}`` for the stages in ``LAYER_END_STAGES``.

    Shapes and dtypes: residual [T, n, C] bf16; head_fn [n, n*C], head_scale [1],
    head_base [n] fp32; norm_weight [C] fp32; lm_head_weight [V, C] bf16;
    target_ids [T] integer. T may be 0.
    """
    if residual.dim() != 3:
        raise ValueError(f"residual must have shape [T, n, C], got {tuple(residual.shape)}")
    if lm_head_weight.dtype != torch.bfloat16:
        raise TypeError(f"lm_head_weight must be bf16, got {lm_head_weight.dtype}")
    linear, logprob_op = _lm_head_and_logprob(backend)
    if backend == "rocm":
        mhc_backend = "rocm_triton"
    else:
        mhc_backend = "reference" if residual.device.type == "cpu" else "rocm"
    head = mhc_head(residual, head_fn, head_scale, head_base, backend=mhc_backend)
    final_norm, _ = rmsnorm_residual(head, norm_weight, norm_eps, backend=backend)
    logits = linear(final_norm, lm_head_weight, out_dtype=torch.float32)
    logprob = logprob_op(logits, target_ids, validate=True)
    return {"head": head, "final_norm": final_norm, "logits": logits, "logprob": logprob}
