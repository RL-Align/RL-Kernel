# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fixed-order PyTorch reference for joint-attention softmax.

Softmax reduces the last dimension of BF16/FP32 ``scores[..., K]`` in FP32.
``forward`` returns the input dtype; ``forward_fp32`` returns FP32. Scores
may use ``-inf`` to mask keys, but each row needs at least one finite key.

Rows use 256-key tiles with fixed reduction and left-to-right merge order.
CUDA and Triton follow the same arithmetic contract for bytewise comparison.
NaNs, ``+inf``, and fully masked rows are outside that contract.
"""

from __future__ import annotations

import torch

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
        "kernel_fingerprint": "joint-attn-softmax-v1-tile256-exp7",
        "fallback": False,
    }

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        return self.forward(scores)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Return probabilities in the input dtype."""
        return _NativeJointAttnSoftmaxFunction.apply(scores, False)

    def forward_fp32(self, scores: torch.Tensor) -> torch.Tensor:
        """Return probabilities in FP32."""
        return _NativeJointAttnSoftmaxFunction.apply(scores, True)


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


def _fixed_online_softmax(scores: torch.Tensor) -> torch.Tensor:
    """Apply the same per-row arithmetic independently of the batch shape."""
    scores_fp32 = scores.float()
    if scores_fp32.numel() == 0:
        return scores_fp32.clone()
    key_length = scores_fp32.size(-1)
    rows = scores_fp32.reshape(-1, key_length)
    probabilities = torch.stack([_fixed_online_softmax_row(row) for row in rows])
    return probabilities.reshape(scores.shape)


def _fixed_softmax_backward(
    probabilities: torch.Tensor, grad_probabilities: torch.Tensor
) -> torch.Tensor:
    """Apply the fixed backward independently to each flattened score row."""
    if probabilities.numel() == 0:
        return torch.empty_like(probabilities)

    key_length = probabilities.size(-1)
    probability_rows = probabilities.reshape(-1, key_length)
    gradient_rows = grad_probabilities.float().reshape(-1, key_length)

    grad_scores = torch.stack(
        [
            _fixed_softmax_backward_row(probability_row, gradient_row)
            for probability_row, gradient_row in zip(probability_rows, gradient_rows, strict=True)
        ],
        dim=0,
    )
    return grad_scores.reshape(probabilities.shape)


def _validate_scores(scores: torch.Tensor) -> None:
    """Check structural inputs; the finite-row condition is a caller precondition."""
    if scores.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"scores must use BF16 or FP32, got {scores.dtype}")
    if scores.dim() < 1:
        raise ValueError("scores must be at least 1-D with shape [..., K]")
    if scores.size(-1) == 0:
        raise ValueError("scores key dimension must be non-empty")


def _fixed_online_softmax_row(row: torch.Tensor) -> torch.Tensor:
    """Merge tile (max, sum) states, then normalize one row in FP32."""
    online_max: torch.Tensor | None = None
    online_sum: torch.Tensor | None = None

    # First pass: fixed tree within each tile, then left-to-right online merge.
    for tile_start in range(0, row.numel(), _TILE_K):
        tile = row[tile_start : tile_start + _TILE_K]
        padding = _TILE_K - tile.numel()

        max_values = torch.cat(
            [tile, torch.full((padding,), float("-inf"), device=row.device, dtype=row.dtype)]
        )
        tile_max = _tree_max_256(max_values)

        # An all-masked tile has zero mass. Merging two such tiles would evaluate
        # -inf - -inf, so only finite tiles participate in the online state.
        if torch.isneginf(tile_max):
            continue

        exp_values = torch.where(
            torch.isneginf(tile),
            torch.zeros_like(tile),
            _portable_exp_nonpositive(tile - tile_max),
        )
        sum_values = torch.cat(
            [exp_values, torch.zeros((padding,), device=row.device, dtype=row.dtype)]
        )
        tile_sum = _tree_sum_256(sum_values)

        if online_max is None:
            online_max = tile_max
            online_sum = tile_sum
        else:
            assert online_sum is not None
            new_max = torch.maximum(online_max, tile_max)
            old_scale = _portable_exp_nonpositive(online_max - new_max)
            scaled_old_sum = online_sum * old_scale
            tile_scale = _portable_exp_nonpositive(tile_max - new_max)
            scaled_tile_sum = tile_sum * tile_scale
            online_sum = scaled_old_sum + scaled_tile_sum
            online_max = new_max

    assert online_max is not None and online_sum is not None
    # Second pass: use the final row state, without changing the merge order.
    return _portable_exp_nonpositive(row - online_max) / online_sum


def _fixed_softmax_backward_row(
    probabilities: torch.Tensor, grad_probabilities: torch.Tensor
) -> torch.Tensor:
    """Compute P * (dP - delta) with a fixed reduction for delta = sum(P * dP)."""
    row_delta: torch.Tensor | None = None

    for tile_start in range(0, probabilities.numel(), _TILE_K):
        tile_products = (
            probabilities[tile_start : tile_start + _TILE_K]
            * grad_probabilities[tile_start : tile_start + _TILE_K]
        )

        padding = _TILE_K - tile_products.numel()
        tree_values = torch.cat(
            [
                tile_products,
                torch.zeros((padding,), device=probabilities.device, dtype=probabilities.dtype),
            ]
        )
        tile_delta = _tree_sum_256(tree_values)
        row_delta = tile_delta if row_delta is None else row_delta + tile_delta

    assert row_delta is not None
    return probabilities * (grad_probabilities - row_delta)


def _tree_max_256(values: torch.Tensor) -> torch.Tensor:
    """Reduce one padded tile with pairings 128, 64, ..., 1."""
    width = _TILE_K
    while width > 1:
        half = width // 2
        values = torch.maximum(values[:half], values[half:width])
        width = half
    return values[0]


def _tree_sum_256(values: torch.Tensor) -> torch.Tensor:
    """Reduce one padded tile with pairings 128, 64, ..., 1."""
    width = _TILE_K
    while width > 1:
        half = width // 2
        values = values[:half] + values[half:width]
        width = half
    return values[0]


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
