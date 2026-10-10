# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton joint-attention softmax with the issue #386 arithmetic contract."""

from __future__ import annotations

from math import prod

import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.joint_attn_softmax_layout import (
    LogicalKeyMapping,
    validate_key_padding_mask,
)

_TILE_K = 256

# ---------------------------------------------------------------------------
# Fixed arithmetic
# ---------------------------------------------------------------------------


@triton.jit
def _mul_rn(left, right):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn(left, right):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn(left, right):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _div_rn(left, right):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _is_nan(values):
    bits = tl.cast(values, tl.int32, bitcast=True)
    return (bits & 0x7FFFFFFF) > 0x7F800000


@triton.jit
def _portable_exp_nonpositive(values):
    is_nan = _is_nan(values)
    safe_values = tl.where(is_nan, 0.0, tl.maximum(values, -104.0))
    scaled = _add_rn(_mul_rn(safe_values, 1.4426950216293335), 0.5)
    exponent = tl.floor(scaled).to(tl.int32)
    exponent_fp32 = exponent.to(tl.float32)
    remainder = _sub_rn(safe_values, _mul_rn(exponent_fp32, 0.693145751953125))
    remainder = _sub_rn(remainder, _mul_rn(exponent_fp32, 1.428606765330187e-6))

    polynomial = tl.full(values.shape, 0.00019841270113829523, tl.float32)
    polynomial = _add_rn(_mul_rn(polynomial, remainder), 0.0013888889225199819)
    polynomial = _add_rn(_mul_rn(polynomial, remainder), 0.008333333767950535)
    polynomial = _add_rn(_mul_rn(polynomial, remainder), 0.0416666679084301)
    polynomial = _add_rn(_mul_rn(polynomial, remainder), 0.1666666716337204)
    polynomial = _add_rn(_mul_rn(polynomial, remainder), 0.5)
    polynomial = _add_rn(_mul_rn(polynomial, remainder), 1.0)
    polynomial = _add_rn(_mul_rn(polynomial, remainder), 1.0)

    regular_bits = (exponent + 127) << 23
    subnormal_bits = (exponent + 64 + 127) << 23
    regular_scale = tl.cast(regular_bits, tl.float32, bitcast=True)
    subnormal_scale = tl.cast(subnormal_bits, tl.float32, bitcast=True)
    regular_result = _mul_rn(polynomial, regular_scale)
    subnormal_result = _mul_rn(polynomial, subnormal_scale)
    subnormal_result = _mul_rn(subnormal_result, 5.421010862427522e-20)
    result = tl.where(exponent >= -126, regular_result, subnormal_result)
    result = tl.where(values < -104.0, 0.0, result)
    return tl.where(is_nan, values, result)


@triton.jit
def _tree_max_256(values):
    maximum = tl.max(tl.reshape(values, (2, 128)), axis=0)
    maximum = tl.max(tl.reshape(maximum, (2, 64)), axis=0)
    maximum = tl.max(tl.reshape(maximum, (2, 32)), axis=0)
    maximum = tl.max(tl.reshape(maximum, (2, 16)), axis=0)
    maximum = tl.max(tl.reshape(maximum, (2, 8)), axis=0)
    maximum = tl.max(tl.reshape(maximum, (2, 4)), axis=0)
    maximum = tl.max(tl.reshape(maximum, (2, 2)), axis=0)
    maximum = tl.max(tl.reshape(maximum, (2, 1)), axis=0)
    return tl.max(maximum, axis=0)


@triton.jit
def _tree_sum_256(values):
    total = tl.sum(tl.reshape(values, (2, 128)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 64)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 32)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 16)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 8)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 4)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 2)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 1)), axis=0)
    return tl.sum(total, axis=0)


# ---------------------------------------------------------------------------
# Key-layout kernel
# ---------------------------------------------------------------------------


@triton.jit
def _joint_attn_softmax_key_mapping_kernel(
    key_padding_mask_ptr,
    logical_to_physical_ptr,
    valid_key_counts_ptr,
    key_length,
    TILE_K: tl.constexpr,
):
    batch = tl.program_id(0)
    lane = tl.arange(0, TILE_K)
    batch_offset = batch.to(tl.int64) * key_length
    next_logical_key = tl.zeros((), tl.int32)

    for tile_start in range(0, key_length, TILE_K):
        physical_keys = tile_start + lane
        in_bounds = physical_keys < key_length
        valid_keys = tl.load(
            key_padding_mask_ptr + batch_offset + physical_keys,
            mask=in_bounds,
            other=0,
        ).to(tl.int32)
        logical_keys = next_logical_key + tl.cumsum(valid_keys, axis=0) - 1
        tl.store(
            logical_to_physical_ptr + batch_offset + logical_keys,
            physical_keys,
            mask=in_bounds & (valid_keys != 0),
        )
        next_logical_key += tl.sum(valid_keys, axis=0)

    tl.store(valid_key_counts_ptr + batch, next_logical_key)


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------


@triton.jit
def _joint_attn_softmax_forward_kernel(
    scores_ptr,
    probabilities_ptr,
    saved_probabilities_ptr,
    logical_to_physical_ptr,
    valid_key_counts_ptr,
    row_count,
    key_length,
    rows_per_batch,
    SAVE_STATE: tl.constexpr,
    HAS_KEY_LAYOUT: tl.constexpr,
    TILE_K: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= row_count:
        return

    lane = tl.arange(0, TILE_K)
    row_offset = row.to(tl.int64) * key_length
    if HAS_KEY_LAYOUT:
        batch = row // rows_per_batch
        key_map_offset = batch.to(tl.int64) * key_length
        valid_key_count = tl.load(valid_key_counts_ptr + batch)
        for tile_start in range(0, key_length, TILE_K):
            physical_columns = tile_start + lane
            physical_valid = physical_columns < key_length
            tl.store(
                probabilities_ptr + row_offset + physical_columns,
                0.0,
                mask=physical_valid,
            )
            if SAVE_STATE:
                tl.store(
                    saved_probabilities_ptr + row_offset + physical_columns,
                    0.0,
                    mask=physical_valid,
                )
    else:
        key_map_offset = 0
        valid_key_count = key_length
    online_max = tl.full((), float("-inf"), tl.float32)
    online_sum = tl.zeros((), tl.float32)

    for tile_start in range(0, key_length, TILE_K):
        columns = tile_start + lane
        valid = columns < valid_key_count
        physical_columns = columns
        if HAS_KEY_LAYOUT:
            physical_columns = tl.load(
                logical_to_physical_ptr + key_map_offset + columns,
                mask=valid,
                other=0,
            )
        scores = tl.load(
            scores_ptr + row_offset + physical_columns,
            mask=valid,
            other=float("-inf"),
        ).to(tl.float32)
        tile_max = _tree_max_256(scores)
        contributes = valid & (scores != float("-inf"))
        exp_values = tl.where(
            contributes,
            _portable_exp_nonpositive(_sub_rn(scores, tile_max)),
            0.0,
        )
        tile_sum = _tree_sum_256(exp_values)
        # Keep the finite max tree unchanged while exposing unsupported NaN/+inf input.
        tile_state_max = tl.where(_is_nan(tile_sum), float("nan"), tile_max)

        if tile_state_max != float("-inf"):
            if online_max == float("-inf"):
                online_max = tile_state_max
                online_sum = tile_sum
            else:
                new_max = tl.maximum(online_max, tile_state_max)
                old_scale = _portable_exp_nonpositive(_sub_rn(online_max, new_max))
                tile_scale = _portable_exp_nonpositive(_sub_rn(tile_state_max, new_max))
                online_sum = _add_rn(
                    _mul_rn(online_sum, old_scale),
                    _mul_rn(tile_sum, tile_scale),
                )
                online_max = new_max

    row_has_state = online_max != float("-inf")
    safe_online_max = tl.where(row_has_state, online_max, 0.0)
    safe_online_sum = tl.where(row_has_state, online_sum, 1.0)

    for tile_start in range(0, key_length, TILE_K):
        columns = tile_start + lane
        valid = columns < valid_key_count
        physical_columns = columns
        if HAS_KEY_LAYOUT:
            physical_columns = tl.load(
                logical_to_physical_ptr + key_map_offset + columns,
                mask=valid,
                other=0,
            )
        scores = tl.load(
            scores_ptr + row_offset + physical_columns,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        exp_values = _portable_exp_nonpositive(_sub_rn(scores, safe_online_max))
        probabilities = tl.where(
            row_has_state,
            _div_rn(exp_values, safe_online_sum),
            0.0,
        )
        tl.store(
            probabilities_ptr + row_offset + physical_columns,
            probabilities,
            mask=valid,
        )
        if SAVE_STATE:
            tl.store(
                saved_probabilities_ptr + row_offset + physical_columns,
                probabilities,
                mask=valid,
            )


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------


@triton.jit
def _joint_attn_softmax_backward_kernel(
    probabilities_ptr,
    grad_probabilities_ptr,
    grad_scores_ptr,
    logical_to_physical_ptr,
    valid_key_counts_ptr,
    row_count,
    key_length,
    rows_per_batch,
    HAS_KEY_LAYOUT: tl.constexpr,
    TILE_K: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= row_count:
        return

    lane = tl.arange(0, TILE_K)
    row_offset = row.to(tl.int64) * key_length
    if HAS_KEY_LAYOUT:
        batch = row // rows_per_batch
        key_map_offset = batch.to(tl.int64) * key_length
        valid_key_count = tl.load(valid_key_counts_ptr + batch)
        for tile_start in range(0, key_length, TILE_K):
            physical_columns = tile_start + lane
            physical_valid = physical_columns < key_length
            tl.store(
                grad_scores_ptr + row_offset + physical_columns,
                0.0,
                mask=physical_valid,
            )
    else:
        key_map_offset = 0
        valid_key_count = key_length

    columns = lane
    valid = columns < valid_key_count
    physical_columns = columns
    if HAS_KEY_LAYOUT:
        physical_columns = tl.load(
            logical_to_physical_ptr + key_map_offset + columns,
            mask=valid,
            other=0,
        )
    probabilities = tl.load(
        probabilities_ptr + row_offset + physical_columns,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    grad_probabilities = tl.load(
        grad_probabilities_ptr + row_offset + physical_columns,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    row_delta = _tree_sum_256(_mul_rn(probabilities, grad_probabilities))

    for tile_start in range(TILE_K, key_length, TILE_K):
        columns = tile_start + lane
        valid = columns < valid_key_count
        physical_columns = columns
        if HAS_KEY_LAYOUT:
            physical_columns = tl.load(
                logical_to_physical_ptr + key_map_offset + columns,
                mask=valid,
                other=0,
            )
        probabilities = tl.load(
            probabilities_ptr + row_offset + physical_columns,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        grad_probabilities = tl.load(
            grad_probabilities_ptr + row_offset + physical_columns,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        tile_delta = _tree_sum_256(_mul_rn(probabilities, grad_probabilities))
        row_delta = _add_rn(row_delta, tile_delta)

    for tile_start in range(0, key_length, TILE_K):
        columns = tile_start + lane
        valid = columns < valid_key_count
        physical_columns = columns
        if HAS_KEY_LAYOUT:
            physical_columns = tl.load(
                logical_to_physical_ptr + key_map_offset + columns,
                mask=valid,
                other=0,
            )
        probabilities = tl.load(
            probabilities_ptr + row_offset + physical_columns,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        grad_probabilities = tl.load(
            grad_probabilities_ptr + row_offset + physical_columns,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        grad_scores = _mul_rn(probabilities, _sub_rn(grad_probabilities, row_delta))
        # Padding can change the sign of an exact zero; always store +0.
        grad_scores = tl.where(grad_scores == 0.0, 0.0, grad_scores)
        tl.store(
            grad_scores_ptr + row_offset + physical_columns,
            grad_scores,
            mask=valid,
        )


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------


def _build_key_mapping(
    scores: torch.Tensor,
    key_padding_mask: torch.Tensor,
) -> LogicalKeyMapping:
    """Build one compact logical-key map per batch item on the GPU."""
    validate_key_padding_mask(scores, key_padding_mask)
    batch_size, key_length = key_padding_mask.shape
    logical_to_physical = torch.empty(
        (batch_size, key_length),
        device=scores.device,
        dtype=torch.int32,
    )
    valid_key_counts = torch.empty(
        batch_size,
        device=scores.device,
        dtype=torch.int32,
    )
    if batch_size > 0:
        with torch.cuda.device(scores.device):
            _joint_attn_softmax_key_mapping_kernel[(batch_size,)](
                key_padding_mask.contiguous(),
                logical_to_physical,
                valid_key_counts,
                key_length,
                TILE_K=_TILE_K,
                num_warps=8,
            )
    return LogicalKeyMapping(
        logical_to_physical=logical_to_physical,
        valid_key_counts=valid_key_counts,
        rows_per_batch=prod(scores.shape[1:-1]),
    )


def _launch_forward(
    scores: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    save_state: bool,
    key_mapping: LogicalKeyMapping | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    key_length = scores.size(-1)
    contiguous_scores = scores.contiguous()
    probabilities = torch.empty(scores.shape, device=scores.device, dtype=output_dtype)
    saved_probabilities = (
        torch.empty(scores.shape, device=scores.device, dtype=torch.float32)
        if save_state and output_dtype != torch.float32
        else None
    )
    state_ptr = probabilities if saved_probabilities is None else saved_probabilities
    row_count = scores.numel() // key_length
    if row_count > 0:
        logical_to_physical = scores if key_mapping is None else key_mapping.logical_to_physical
        valid_key_counts = scores if key_mapping is None else key_mapping.valid_key_counts
        rows_per_batch = 1 if key_mapping is None else key_mapping.rows_per_batch
        with torch.cuda.device(scores.device):
            _joint_attn_softmax_forward_kernel[(row_count,)](
                contiguous_scores,
                probabilities,
                state_ptr,
                logical_to_physical,
                valid_key_counts,
                row_count,
                key_length,
                rows_per_batch,
                SAVE_STATE=saved_probabilities is not None,
                HAS_KEY_LAYOUT=key_mapping is not None,
                TILE_K=_TILE_K,
                num_warps=8,
            )
    backward_state = probabilities if output_dtype == torch.float32 else saved_probabilities
    return probabilities, backward_state


# ---------------------------------------------------------------------------
# Autograd bridge
# ---------------------------------------------------------------------------


class _TritonJointAttnSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        output_fp32: bool,
        logical_to_physical: torch.Tensor | None,
        valid_key_counts: torch.Tensor | None,
        rows_per_batch: int,
    ) -> torch.Tensor:
        key_length = scores.size(-1)
        output_dtype = torch.float32 if output_fp32 else scores.dtype
        key_mapping = (
            None
            if logical_to_physical is None or valid_key_counts is None
            else LogicalKeyMapping(
                logical_to_physical=logical_to_physical,
                valid_key_counts=valid_key_counts,
                rows_per_batch=rows_per_batch,
            )
        )
        probabilities, saved_probabilities_fp32 = _launch_forward(
            scores,
            output_dtype=output_dtype,
            save_state=True,
            key_mapping=key_mapping,
        )
        assert saved_probabilities_fp32 is not None
        if key_mapping is None:
            ctx.save_for_backward(saved_probabilities_fp32)
        else:
            ctx.save_for_backward(
                saved_probabilities_fp32,
                key_mapping.logical_to_physical,
                key_mapping.valid_key_counts,
            )
        ctx.has_key_layout = key_mapping is not None
        ctx.input_dtype = scores.dtype
        ctx.key_length = key_length
        ctx.rows_per_batch = rows_per_batch
        return probabilities

    @staticmethod
    def backward(
        ctx,
        grad_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        probabilities_fp32 = ctx.saved_tensors[0]
        key_mapping = (
            LogicalKeyMapping(
                logical_to_physical=ctx.saved_tensors[1],
                valid_key_counts=ctx.saved_tensors[2],
                rows_per_batch=ctx.rows_per_batch,
            )
            if ctx.has_key_layout
            else None
        )
        grad_scores = torch.empty(
            probabilities_fp32.shape,
            device=probabilities_fp32.device,
            dtype=ctx.input_dtype,
        )
        row_count = probabilities_fp32.numel() // ctx.key_length
        if row_count > 0:
            logical_to_physical = (
                probabilities_fp32 if key_mapping is None else key_mapping.logical_to_physical
            )
            valid_key_counts = (
                probabilities_fp32 if key_mapping is None else key_mapping.valid_key_counts
            )
            rows_per_batch = 1 if key_mapping is None else key_mapping.rows_per_batch
            with torch.cuda.device(probabilities_fp32.device):
                _joint_attn_softmax_backward_kernel[(row_count,)](
                    probabilities_fp32,
                    grad_probabilities.contiguous(),
                    grad_scores,
                    logical_to_physical,
                    valid_key_counts,
                    row_count,
                    ctx.key_length,
                    rows_per_batch,
                    HAS_KEY_LAYOUT=key_mapping is not None,
                    TILE_K=_TILE_K,
                    num_warps=8,
                )
        return grad_scores, None, None, None, None


# ---------------------------------------------------------------------------
# Public operator
# ---------------------------------------------------------------------------


class TritonJointAttnSoftmaxOp:
    """Triton softmax using the shared fixed-order FP32 contract."""

    backend_id = "rlkernel.triton.joint_attn_softmax"
    provenance = {
        "selected_backend": "triton_cuda",
        "reduction_order": "tile256_tree_128_to_1_then_left_to_right",
        "accumulator_precision": "fp32",
        "split_k": False,
        "stream_k": False,
        "tf32": False,
        "kernel_fingerprint": "joint-attn-softmax-v2-logical-mask-tile256-exp7",
        "fallback": False,
    }

    def __init__(self) -> None:
        if torch.version.hip is not None:
            raise RuntimeError(
                "the joint-attention Triton arithmetic contract currently uses CUDA PTX"
            )

    def __call__(
        self,
        scores: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias for :meth:`forward`, matching the other backends."""
        return self.forward(scores, key_padding_mask=key_padding_mask)

    def forward(
        self,
        scores: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute in FP32 and cast once at the final Triton write."""
        return self._forward_impl(scores, False, key_padding_mask)

    def forward_fp32(
        self,
        scores: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return FP32 probabilities over the final key dimension."""
        return self._forward_impl(scores, True, key_padding_mask)

    def _forward_impl(
        self,
        scores: torch.Tensor,
        output_fp32: bool,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        self._validate_scores(scores)
        key_mapping = (
            None if key_padding_mask is None else _build_key_mapping(scores, key_padding_mask)
        )
        return self._forward_core(scores, output_fp32, key_mapping)

    @staticmethod
    def _forward_core(
        scores: torch.Tensor,
        output_fp32: bool,
        key_mapping: LogicalKeyMapping | None,
    ) -> torch.Tensor:
        if not torch.is_grad_enabled() or not scores.requires_grad:
            output_dtype = torch.float32 if output_fp32 else scores.dtype
            probabilities, _ = _launch_forward(
                scores,
                output_dtype=output_dtype,
                save_state=False,
                key_mapping=key_mapping,
            )
            return probabilities
        logical_to_physical = None if key_mapping is None else key_mapping.logical_to_physical
        valid_key_counts = None if key_mapping is None else key_mapping.valid_key_counts
        rows_per_batch = 1 if key_mapping is None else key_mapping.rows_per_batch
        return _TritonJointAttnSoftmaxFunction.apply(
            scores,
            output_fp32,
            logical_to_physical,
            valid_key_counts,
            rows_per_batch,
        )

    @staticmethod
    def _validate_scores(scores: torch.Tensor) -> None:
        if not scores.is_cuda:
            raise RuntimeError("TritonJointAttnSoftmaxOp requires a CUDA tensor")
        if scores.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"scores must use BF16 or FP32, got {scores.dtype}")
        if scores.dim() < 1:
            raise ValueError("scores must be at least 1-D with shape [..., K]")
        if scores.size(-1) == 0:
            raise ValueError("scores key dimension must be non-empty")
