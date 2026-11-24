# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini token embedding contract (INK-06)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

PADDED_VOCAB_SIZE = 201_024
VALID_VOCAB_SIZE = 200_058


def token_embedding(
    token_ids: torch.Tensor,
    weight: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Gather padded-vocabulary rows with explicit input and output dtypes.

    Token IDs may arrive as int32 or int64 and are converted to int64 for the
    gather. The embedding table controls the default output dtype. Padded rows
    are valid rows when they exist in ``weight`` and are preserved verbatim.
    """
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("token ids must be int32 or int64")
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("embedding weight must be a floating-point [vocab, hidden] tensor")
    if token_ids.device != weight.device:
        raise ValueError("token ids and embedding weight must share a device")

    ids = token_ids.to(dtype=torch.int64)
    if ids.numel():
        min_id = int(ids.min().item())
        max_id = int(ids.max().item())
        if min_id < 0 or max_id >= weight.shape[0]:
            raise ValueError(
                f"token ids must be in [0, {weight.shape[0] - 1}], got [{min_id}, {max_id}]"
            )

    output = F.embedding(ids, weight)
    if output_dtype is not None:
        output = output.to(dtype=output_dtype)
    return output


class _NativeTokenEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids, weight):
        from rl_engine.backends.extension import _C

        output = _C.embedding_sm90_forward(token_ids.contiguous(), weight.contiguous())
        ctx.save_for_backward(token_ids.contiguous())
        ctx.vocab_size = weight.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from rl_engine.backends.extension import _C

        (token_ids,) = ctx.saved_tensors
        grad_weight = _C.embedding_sm90_backward(
            token_ids, grad_output.contiguous(), int(ctx.vocab_size)
        )
        return None, grad_weight


def token_embedding_cuda(
    token_ids: torch.Tensor,
    weight: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run the native SM90 embedding forward and deterministic backward kernel."""
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if output_dtype is not None and output_dtype != weight.dtype:
        raise ValueError("native token embedding output_dtype must match weight dtype")
    if not _EXT_AVAILABLE or not hasattr(_C, "embedding_sm90_forward"):
        raise RuntimeError("SM90 embedding CUDA extension is unavailable")
    if not token_ids.is_cuda or not weight.is_cuda:
        raise ValueError("native token embedding requires CUDA tensors")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("token ids must be int32 or int64")
    if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("native embedding weight must be [vocab, hidden] in fp32, fp16, or bf16")
    if token_ids.device != weight.device:
        raise ValueError("token ids and embedding weight must share a device")
    ids = token_ids.to(dtype=torch.int64)
    if ids.numel():
        min_id = int(ids.min().item())
        max_id = int(ids.max().item())
        if min_id < 0 or max_id >= weight.shape[0]:
            raise ValueError(
                f"token ids must be in [0, {weight.shape[0] - 1}], got [{min_id}, {max_id}]"
            )
    return _NativeTokenEmbeddingFunction.apply(ids, weight)


class _RocmTokenEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids, weight):
        from rl_engine.backends.extension import _C

        ids = token_ids.contiguous()
        ctx.save_for_backward(ids)
        ctx.vocab_size = weight.shape[0]
        return _C.inkling2_token_embedding_rocm(ids, weight.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        from rl_engine.backends.extension import _C

        (ids,) = ctx.saved_tensors
        grad_weight = _C.inkling2_token_embedding_backward_rocm(
            ids, grad_output.contiguous(), int(ctx.vocab_size)
        )
        return None, grad_weight


def token_embedding_rocm(
    token_ids: torch.Tensor,
    weight: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run the ROCm padded-vocabulary gather (byte-exact rows) and its backward.

    Same contract as :func:`token_embedding`: int32/int64 ids, ids outside
    ``[0, weight.shape[0])`` raise ``ValueError`` (padded rows are valid rows),
    the table dtype is the default output dtype, and ``output_dtype`` applies an
    explicit cast after the gather. The backward accumulates duplicate ids in
    FP32 in ascending token order without atomics.
    """
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if getattr(torch.version, "hip", None) is None:
        raise RuntimeError("ROCm token embedding requires a ROCm/HIP PyTorch build")
    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_token_embedding_rocm"):
        raise RuntimeError("ROCm token embedding extension is unavailable")
    if not token_ids.is_cuda or not weight.is_cuda:
        raise ValueError("ROCm token embedding requires HIP device tensors")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("token ids must be int32 or int64")
    if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("ROCm embedding weight must be [vocab, hidden] in fp32, fp16, or bf16")
    if token_ids.device != weight.device:
        raise ValueError("token ids and embedding weight must share a device")
    if token_ids.numel():
        min_id = int(token_ids.min().item())
        max_id = int(token_ids.max().item())
        if min_id < 0 or max_id >= weight.shape[0]:
            raise ValueError(
                f"token ids must be in [0, {weight.shape[0] - 1}], got [{min_id}, {max_id}]"
            )
    output = _RocmTokenEmbeddingFunction.apply(token_ids, weight)
    if output_dtype is not None:
        output = output.to(dtype=output_dtype)
    return output
