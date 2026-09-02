# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic Triton embedding with an atomic-free sorted-segment backward."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.backward_runtime import record_backward

_SUPPORTED_WEIGHT_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
_BACKWARD_BLOCK_H = 128


def _validate_embedding_inputs(token_ids: torch.Tensor, weight: torch.Tensor) -> None:
    if weight.dim() != 2:
        raise ValueError(f"embedding weight must be [vocab, hidden], got {tuple(weight.shape)}")
    if weight.size(0) <= 0 or weight.size(1) <= 0:
        raise ValueError("embedding weight must have positive vocab and hidden dimensions")
    if weight.dtype not in _SUPPORTED_WEIGHT_DTYPES:
        raise TypeError("embedding weight must use fp16, bf16, or fp32")
    if token_ids.is_floating_point() or token_ids.is_complex() or token_ids.dtype == torch.bool:
        raise TypeError("embedding token_ids must use an integer dtype")
    if token_ids.device != weight.device:
        raise RuntimeError("embedding token_ids and weight must be on the same device")


def _assert_valid_token_ids_async(ids: torch.Tensor, vocab_size: int) -> None:
    if ids.numel() == 0:
        return
    valid = ((ids >= 0) & (ids < vocab_size)).all()
    torch._assert_async(valid, f"embedding token_ids must be in [0, {vocab_size})")


def _validate_token_ids_sync(token_ids: torch.Tensor, vocab_size: int) -> None:
    if vocab_size <= 0:
        raise ValueError("embedding vocab_size must be positive")
    if token_ids.is_floating_point() or token_ids.is_complex() or token_ids.dtype == torch.bool:
        raise TypeError("embedding token_ids must use an integer dtype")
    ids = token_ids.reshape(-1).to(dtype=torch.int64)
    if ids.numel() and bool(((ids < 0) | (ids >= vocab_size)).any().item()):
        raise ValueError(f"embedding token_ids must be in [0, {vocab_size})")


def _stable_sort_token_rows(
    ids: torch.Tensor, grad_rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(ids, stable=True)
    return ids.index_select(0, order), grad_rows.index_select(0, order).contiguous()


def _embedding_backward_grid(n_tokens: int, hidden: int) -> tuple[int, int]:
    return n_tokens, (hidden + _BACKWARD_BLOCK_H - 1) // _BACKWARD_BLOCK_H


def _validate_embedding_backward_inputs(
    ids: torch.Tensor,
    grad_rows: torch.Tensor,
    *,
    weight_shape: tuple[int, int],
    weight_dtype: torch.dtype,
) -> None:
    if len(weight_shape) != 2 or weight_shape[0] <= 0 or weight_shape[1] <= 0:
        raise ValueError("embedding weight_shape must contain positive vocab and hidden sizes")
    if ids.ndim != 1:
        raise ValueError("embedding backward ids must be flattened to one dimension")
    if grad_rows.ndim != 2 or grad_rows.shape != (ids.numel(), weight_shape[1]):
        raise ValueError(
            "embedding backward rows must have shape " f"[{ids.numel()}, {weight_shape[1]}]"
        )
    if ids.device != grad_rows.device:
        raise RuntimeError("embedding backward ids and rows must share a device")
    if ids.is_floating_point() or ids.is_complex() or ids.dtype == torch.bool:
        raise TypeError("embedding backward ids must use an integer dtype")
    if weight_dtype not in _SUPPORTED_WEIGHT_DTYPES:
        raise TypeError("embedding backward weight must use fp16, bf16, or fp32")


def _embedding_grad_weight(
    ids: torch.Tensor,
    grad_rows: torch.Tensor,
    *,
    weight_shape: tuple[int, int],
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    """Reduce rows in their supplied order after stable grouping by token id."""

    _validate_embedding_backward_inputs(
        ids,
        grad_rows,
        weight_shape=weight_shape,
        weight_dtype=weight_dtype,
    )
    vocab, hidden = weight_shape
    grad_weight = torch.zeros(
        (vocab, hidden),
        device=grad_rows.device,
        dtype=weight_dtype,
    )
    if ids.numel() == 0:
        return grad_weight

    sorted_ids, sorted_grad_rows = _stable_sort_token_rows(ids, grad_rows)
    _embedding_bwd[_embedding_backward_grid(ids.numel(), hidden)](
        sorted_ids,
        sorted_grad_rows,
        grad_weight,
        ids.numel(),
        vocab,
        hidden=hidden,
        block_h=_BACKWARD_BLOCK_H,
        num_warps=4,
    )
    return grad_weight


@triton.jit
def _embedding_fwd(
    ids,
    weight,
    out,
    n_tokens,
    vocab_size,
    hidden: tl.constexpr,
    block_h: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, block_h)
    token = tl.load(ids + row)
    valid_token = (token >= 0) & (token < vocab_size)
    values = tl.load(
        weight + token * hidden + offs,
        mask=valid_token & (offs < hidden),
        other=0.0,
    )
    tl.store(out + row * hidden + offs, values, mask=offs < hidden)


@triton.jit
def _embedding_bwd(
    sorted_ids,
    sorted_grad_rows,
    grad_weight,
    n_tokens,
    vocab_size,
    hidden: tl.constexpr,
    block_h: tl.constexpr,
):
    sorted_position = tl.program_id(0)
    column_block = tl.program_id(1)
    columns = column_block * block_h + tl.arange(0, block_h)
    token = tl.load(sorted_ids + sorted_position)
    previous_token = tl.load(
        sorted_ids + sorted_position - 1,
        mask=sorted_position > 0,
        other=-1,
    )
    segment_start = (sorted_position == 0) | (token != previous_token)
    valid_token = (token >= 0) & (token < vocab_size)

    accumulator = tl.zeros((block_h,), tl.float32)
    row = sorted_position
    continuing = segment_start & valid_token
    while (row < n_tokens) & continuing:
        row_token = tl.load(sorted_ids + row)
        continuing = row_token == token
        values = tl.load(
            sorted_grad_rows + row * hidden + columns,
            mask=continuing & (columns < hidden),
            other=0.0,
        ).to(tl.float32)
        accumulator += values
        row += 1

    tl.store(
        grad_weight + token * hidden + columns,
        accumulator,
        mask=segment_start & valid_token & (columns < hidden),
    )


class _TritonEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        _validate_embedding_inputs(token_ids, weight)
        ids = token_ids.reshape(-1).to(dtype=torch.int64).contiguous()
        vocab, hidden = weight.shape
        _assert_valid_token_ids_async(ids, vocab)
        out = torch.empty((ids.numel(), hidden), device=weight.device, dtype=weight.dtype)
        if ids.numel():
            _embedding_fwd[(ids.numel(),)](
                ids,
                weight.contiguous(),
                out,
                ids.numel(),
                vocab,
                hidden=hidden,
                block_h=triton.next_power_of_2(hidden),
            )
        ctx.save_for_backward(ids)
        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        ctx.output_shape = tuple(token_ids.shape) + (hidden,)
        return out.reshape(ctx.output_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (ids,) = ctx.saved_tensors
        vocab, hidden = ctx.weight_shape
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_rows = grad_output.reshape(-1, hidden).contiguous()
            grad_weight = _embedding_grad_weight(
                ids,
                grad_rows,
                weight_shape=(vocab, hidden),
                weight_dtype=ctx.weight_dtype,
            )
        record_backward(
            "embedding",
            kernel_id="rl_engine.kernels.ops.triton.linear.embedding._embedding_bwd",
            impl="triton_sorted_segment_bwd",
            family="triton",
        )
        return None, grad_weight


class TritonEmbeddingOp:
    """Table lookup with a stable sorted-segment weight VJP."""

    op_class = "elementwise"
    is_batch_invariant = True

    def __call__(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.forward(token_ids, weight)

    def forward(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        supported_devices = ("cuda", "hip", "xpu", "musa")
        if (
            token_ids.device.type not in supported_devices
            or weight.device.type not in supported_devices
        ):
            raise RuntimeError(
                "TritonEmbeddingOp requires accelerator tensors " "(CUDA / ROCm / XPU / MUSA)"
            )
        return _TritonEmbeddingFunction.apply(token_ids, weight)

    @staticmethod
    def validate_token_ids(token_ids: torch.Tensor, vocab_size: int) -> None:
        """Synchronously validate ids at an explicit input-validation boundary."""

        _validate_token_ids_sync(token_ids, vocab_size)
