# SPDX-License-Identifier: Apache-2.0
"""INK-32: BF16 unembedding, valid-vocabulary FP32 softcap and selected logprob."""

import torch
import torch.nn.functional as F

PADDED_VOCAB_SIZE = 201024
VALID_VOCAB_SIZE = 200058
LOGIT_SOFTCAP = 30.0
IGNORE_INDEX = -100


def _matmul_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.is_cuda:
        from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm

        return _triton_gemm(a, b, output_dtype=torch.float32)
    with torch.autocast(device_type="cpu", enabled=False):
        return a.float() @ b.float()


class _UnembedFP32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight):
        ctx.save_for_backward(hidden, weight)
        return _matmul_fp32(hidden, weight.t())

    @staticmethod
    def backward(ctx, grad_logits):
        hidden, weight = ctx.saved_tensors
        # Miles' FP32-output linear backward rounds the logit gradient first.
        grad = grad_logits.to(torch.bfloat16)
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = _matmul_fp32(grad, weight).to(hidden.dtype)
        if ctx.needs_input_grad[1]:
            dw = _matmul_fp32(grad.t(), hidden).to(weight.dtype)
        return dx, dw


def softcap_valid_logits(
    raw_logits: torch.Tensor, *, valid_vocab_size: int = VALID_VOCAB_SIZE
) -> torch.Tensor:
    """Exclude padded columns before FP32 cap=30; no BF16 logit cast."""
    if raw_logits.ndim != 2 or not 0 < valid_vocab_size <= raw_logits.shape[-1]:
        raise ValueError("invalid [tokens, vocab] logits or valid vocabulary size")
    if raw_logits.dtype != torch.float32:
        raise TypeError("unembedding logits must remain FP32")
    return LOGIT_SOFTCAP * torch.tanh(raw_logits[:, :valid_vocab_size] / LOGIT_SOFTCAP)


def selected_logprob(capped_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """FP32 selected logprob over already-capped, valid-vocabulary logits.

    GPU dispatches to the existing batch-invariant Triton logprob; CPU uses
    an explicit diagnostic reference. Ignored rows return zero and zero VJP.
    Target validation is eager; production graph metadata is future work.
    """
    if capped_logits.ndim != 2 or min(capped_logits.shape) <= 0:
        raise ValueError("expected nonempty [tokens, valid_vocab] logits")
    if capped_logits.dtype != torch.float32:
        raise TypeError("capped logits must be FP32")
    if targets.shape != capped_logits.shape[:1] or targets.device != capped_logits.device:
        raise ValueError("targets must match token count and device")
    if targets.dtype not in (torch.int32, torch.int64):
        raise TypeError("targets must be integer IDs")
    ignored = targets == IGNORE_INDEX
    invalid = ((targets < 0) | (targets >= capped_logits.shape[-1])) & ~ignored
    if bool(invalid.any()):
        raise ValueError("target outside valid vocabulary")
    if capped_logits.is_cuda:
        from rl_engine.backends.shared.triton.logprob.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )

        return TritonBatchInvariantLogpOp()(capped_logits, targets, ignore_index=IGNORE_INDEX)
    if capped_logits.device.type != "cpu":
        raise RuntimeError("expected CPU reference or CUDA/ROCm Triton path")
    safe = torch.where(ignored, 0, targets).long()
    logp = F.log_softmax(capped_logits, dim=-1).gather(1, safe[:, None]).squeeze(1)
    return torch.where(ignored, 0.0, logp)


def lm_head_logprob(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    valid_vocab_size: int = VALID_VOCAB_SIZE,
) -> torch.Tensor:
    """BF16 compute operands -> FP32 GEMM -> valid-vocab softcap -> logprob.

    Discarding padded weight rows before GEMM avoids computing unused logits.
    The released checkpoint has 201024 stored / 200058 valid rows. A smaller
    explicit valid_vocab_size is allowed for fixtures. FP32 master weights
    are cast to BF16. No fused softcap GEMM or TP gradient parity is claimed.
    """
    if hidden.ndim != 2 or min(hidden.shape) <= 0 or weight.ndim != 2:
        raise ValueError("expected nonempty 2-D hidden and unembedding weight")
    if weight.shape[1] != hidden.shape[1] or not 0 < valid_vocab_size <= weight.shape[0]:
        raise ValueError("invalid unembedding geometry or valid vocabulary size")
    if hidden.dtype != torch.bfloat16 or weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("unembedding requires BF16 hidden and BF16/FP32 weight")
    if hidden.device != weight.device or hidden.device.type not in ("cpu", "cuda"):
        raise ValueError("unembedding tensors must share a CPU or CUDA/ROCm device")
    raw = _UnembedFP32.apply(hidden, weight[:valid_vocab_size].bfloat16())
    return selected_logprob(softcap_valid_logits(raw, valid_vocab_size=valid_vocab_size), targets)
