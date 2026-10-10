# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fixed-order PyTorch reference for joint-attention softmax.

Softmax reduces the last dimension of BF16/FP32 ``scores[..., K]`` in FP32.
``forward`` returns the input dtype; ``forward_fp32`` returns FP32. Scores
may use ``-inf`` to mask keys; fully masked rows return zero probabilities.

Rows use 256-key tiles with fixed reduction and left-to-right merge order.
CUDA and Triton follow the same arithmetic contract for bytewise comparison.
Unsupported NaNs and ``+inf`` propagate NaN results and remain outside that
contract.
"""

from __future__ import annotations

import torch

from rl_engine.kernels.ops.joint_attn_softmax_layout import KeyMaskLayout

_TILE_K = 256


class NativeJointAttnSoftmaxOp:
    """Callable PyTorch reference with input-dtype and FP32 output modes."""

    backend_id = "rlkernel.pytorch.joint_attn_softmax.reference"
    # Static implementation trace; registry candidate rejections are separate.
    provenance = {
        "selected_backend": "pytorch_reference",
        "reduction_order": "tile256_tree_128_to_1_then_left_to_right",
        "accumulator_precision": "fp32",
        "split_k": False,
        "stream_k": False,
        "tf32": False,
        "kernel_fingerprint": "joint-attn-softmax-v2-logical-mask-tile256-exp7",
        "fallback": False,
    }

    def __call__(
        self,
        scores: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(scores, key_padding_mask=key_padding_mask)

    def forward(
        self,
        scores: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return probabilities in the input dtype."""
        return _apply_fixed_softmax(scores, False, key_padding_mask)

    def forward_fp32(
        self,
        scores: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return probabilities in FP32."""
        return _apply_fixed_softmax(scores, True, key_padding_mask)


def _apply_fixed_softmax(
    scores: torch.Tensor,
    output_fp32: bool,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Apply optional layout handling around the fixed softmax core."""
    if key_padding_mask is None:
        return _NativeJointAttnSoftmaxFunction.apply(scores, output_fp32)

    _validate_scores(scores)
    mask_layout = KeyMaskLayout.from_mask(scores, key_padding_mask)
    logical_scores = mask_layout.compact(scores, fill_value=float("-inf"))
    logical_probabilities = _NativeJointAttnSoftmaxFunction.apply(
        logical_scores,
        output_fp32,
    )
    return mask_layout.restore(logical_probabilities, fill_value=0.0)


class _NativeJointAttnSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, output_fp32: bool) -> torch.Tensor:
        _validate_scores(scores)
        probabilities_fp32 = _fixed_online_softmax(scores)
        ctx.save_for_backward(probabilities_fp32)
        ctx.input_dtype = scores.dtype
        output_dtype = torch.float32 if output_fp32 else scores.dtype
        return probabilities_fp32.to(output_dtype)

    @staticmethod
    def backward(ctx, grad_probabilities: torch.Tensor) -> tuple[torch.Tensor, None]:
        (probabilities_fp32,) = ctx.saved_tensors
        grad_scores = _fixed_softmax_backward(probabilities_fp32, grad_probabilities)
        return grad_scores.to(ctx.input_dtype), None


def _validate_scores(scores: torch.Tensor) -> None:
    """Check structural inputs before entering the fixed arithmetic."""
    if scores.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"scores must use BF16 or FP32, got {scores.dtype}")
    if scores.dim() < 1:
        raise ValueError("scores must be at least 1-D with shape [..., K]")
    if scores.size(-1) == 0:
        raise ValueError("scores key dimension must be non-empty")


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _fixed_online_softmax(scores: torch.Tensor) -> torch.Tensor:
    """Apply the same per-row arithmetic independently of the batch shape."""
    scores_fp32 = scores.float()
    if scores_fp32.numel() == 0:
        return scores_fp32.clone()
    key_length = scores_fp32.size(-1)
    rows = scores_fp32.reshape(-1, key_length)
    probabilities = _fixed_online_softmax_rows(rows)
    return probabilities.reshape(scores.shape)


def _fixed_online_softmax_rows(rows: torch.Tensor) -> torch.Tensor:
    """Merge fixed tile states for every row in parallel, then normalize."""
    row_count, key_length = rows.shape
    online_max = rows.new_full((row_count,), float("-inf"))
    online_sum = rows.new_zeros(row_count)
    has_online_state = torch.zeros(row_count, device=rows.device, dtype=torch.bool)

    # First pass: fixed tree within each tile, then left-to-right online merge.
    for tile_start in range(0, key_length, _TILE_K):
        tile = rows[:, tile_start : tile_start + _TILE_K]
        max_values = _pad_tile(tile, float("-inf"))
        tile_max = _tree_max_256(max_values)

        # Avoid -inf - -inf while giving an all-masked tile zero mass.
        tile_has_unmasked_score = tile_max != float("-inf")
        safe_tile_max = torch.where(tile_has_unmasked_score, tile_max, torch.zeros_like(tile_max))

        exp_values = _portable_exp_nonpositive(tile - safe_tile_max.unsqueeze(-1))
        sum_values = _pad_tile(exp_values, 0.0)
        tile_sum = _tree_sum_256(sum_values)

        online_max, online_sum, has_online_state = _merge_online_softmax_state(
            online_max,
            online_sum,
            has_online_state,
            tile_max,
            tile_sum,
        )

    # Second pass: use the final row state, without changing the merge order.
    probabilities = torch.empty_like(rows)
    safe_online_max = torch.where(has_online_state, online_max, torch.zeros_like(online_max))
    safe_online_sum = torch.where(has_online_state, online_sum, torch.ones_like(online_sum))
    for tile_start in range(0, key_length, _TILE_K):
        tile = rows[:, tile_start : tile_start + _TILE_K]
        tile_probabilities = _portable_exp_nonpositive(
            tile - safe_online_max.unsqueeze(-1)
        ) / safe_online_sum.unsqueeze(-1)
        probabilities[:, tile_start : tile_start + _TILE_K] = torch.where(
            has_online_state.unsqueeze(-1),
            tile_probabilities,
            torch.zeros_like(tile_probabilities),
        )
    return probabilities


def _merge_online_softmax_state(
    online_max: torch.Tensor,
    online_sum: torch.Tensor,
    has_online_state: torch.Tensor,
    tile_max: torch.Tensor,
    tile_sum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize, merge, or skip each row's fixed online state."""
    tile_has_unmasked_score = tile_max != float("-inf")
    first_valid_rows = ~has_online_state & tile_has_unmasked_score
    merge_rows = has_online_state & tile_has_unmasked_score

    new_max = torch.maximum(online_max, tile_max)
    old_delta = torch.where(merge_rows, online_max - new_max, torch.zeros_like(online_max))
    tile_delta = torch.where(merge_rows, tile_max - new_max, torch.zeros_like(tile_max))
    merged_sum = online_sum * _portable_exp_nonpositive(old_delta)
    merged_sum = merged_sum + tile_sum * _portable_exp_nonpositive(tile_delta)

    next_max = torch.where(tile_has_unmasked_score, new_max, online_max)
    next_sum = torch.where(first_valid_rows, tile_sum, online_sum)
    next_sum = torch.where(merge_rows, merged_sum, next_sum)
    return next_max, next_sum, has_online_state | tile_has_unmasked_score


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


def _fixed_softmax_backward(
    probabilities: torch.Tensor,
    grad_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Apply the fixed backward independently to each flattened score row."""
    if probabilities.numel() == 0:
        return torch.empty_like(probabilities)

    key_length = probabilities.size(-1)
    probability_rows = probabilities.reshape(-1, key_length)
    gradient_rows = grad_probabilities.float().reshape(-1, key_length)
    grad_scores = _fixed_softmax_backward_rows(probability_rows, gradient_rows)
    return grad_scores.reshape(probabilities.shape)


def _fixed_softmax_backward_rows(
    probabilities: torch.Tensor, grad_probabilities: torch.Tensor
) -> torch.Tensor:
    """Apply the fixed softmax derivative to every row in parallel."""
    row_delta: torch.Tensor | None = None
    key_length = probabilities.size(-1)

    for tile_start in range(0, key_length, _TILE_K):
        tile_products = (
            probabilities[:, tile_start : tile_start + _TILE_K]
            * grad_probabilities[:, tile_start : tile_start + _TILE_K]
        )
        tree_values = _pad_tile(tile_products, 0.0)
        tile_delta = _tree_sum_256(tree_values)
        row_delta = tile_delta if row_delta is None else row_delta + tile_delta

    assert row_delta is not None
    grad_scores = torch.empty_like(probabilities)
    # Write one tile at a time to limit peak temporary memory.
    for tile_start in range(0, key_length, _TILE_K):
        probability_tile = probabilities[:, tile_start : tile_start + _TILE_K]
        gradient_tile = grad_probabilities[:, tile_start : tile_start + _TILE_K]
        tile_grad_scores = probability_tile * (gradient_tile - row_delta.unsqueeze(-1))
        # Padding can change the sign of an exact zero; always store +0.
        grad_scores[:, tile_start : tile_start + _TILE_K] = torch.where(
            tile_grad_scores == 0.0,
            torch.zeros_like(tile_grad_scores),
            tile_grad_scores,
        )
    return grad_scores


# ---------------------------------------------------------------------------
# Fixed-order helpers
# ---------------------------------------------------------------------------


def _pad_tile(values: torch.Tensor, fill_value: float) -> torch.Tensor:
    """Pad only a partial final tile along its last dimension."""
    padding = _TILE_K - values.size(-1)
    if padding == 0:
        return values
    padding_values = values.new_full((*values.shape[:-1], padding), fill_value)
    return torch.cat([values, padding_values], dim=-1)


def _tree_max_256(values: torch.Tensor) -> torch.Tensor:
    """Reduce padded tiles along the last dimension with fixed pairings."""
    width = _TILE_K
    while width > 1:
        half = width // 2
        values = torch.maximum(values[..., :half], values[..., half:width])
        width = half
    return values[..., 0]


def _tree_sum_256(values: torch.Tensor) -> torch.Tensor:
    """Reduce padded tiles along the last dimension with fixed pairings."""
    width = _TILE_K
    while width > 1:
        half = width // 2
        values = values[..., :half] + values[..., half:width]
        width = half
    return values[..., 0]


def _portable_exp_nonpositive(values: torch.Tensor) -> torch.Tensor:
    """Approximate ``exp`` with the fixed sequence shared by all three backends.

    Softmax only evaluates exponentials after subtracting a maximum, so every
    finite input is non-positive. Range reduction and a degree-seven polynomial
    avoid depending on platform ``exp`` implementations. The operation order
    below is part of the cross-backend byte-equality contract.
    """
    # Clamp underflow and keep NaNs out of the integer conversion.
    safe_values = torch.where(
        torch.isnan(values), torch.zeros_like(values), torch.clamp_min(values, -104.0)
    )
    # Write x as exponent * ln(2) + remainder.
    exponent = torch.floor(safe_values * 1.4426950408889634 + 0.5).to(torch.int32)
    exponent_fp32 = exponent.to(torch.float32)
    remainder = safe_values - exponent_fp32 * 0.693145751953125
    remainder = remainder - exponent_fp32 * 1.428606765330187e-6

    # Evaluate the polynomial in the fixed multiply-then-add order.
    polynomial = torch.full_like(remainder, 1.0 / 5040.0)
    for coefficient in (1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0):
        polynomial = polynomial * remainder
        polynomial = polynomial + coefficient

    # Restore 2**exponent, including the subnormal range.
    regular_scale = ((exponent + 127) << 23).view(torch.float32)
    subnormal_scale = ((exponent + 64 + 127) << 23).view(torch.float32)
    regular_result = polynomial * regular_scale
    subnormal_result = polynomial * subnormal_scale
    subnormal_result = subnormal_result * (2.0**-64)
    result = torch.where(exponent >= -126, regular_result, subnormal_result)
    result = torch.where(values < -104.0, torch.zeros_like(result), result)
    return torch.where(torch.isnan(values), values, result)
