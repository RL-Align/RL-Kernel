# SPDX-License-Identifier: Apache-2.0
"""INK-28: explicit route-weight boundaries, ordered combine and AttnRes update."""

import torch

TOP_K = 8


def _check_float(tensor: torch.Tensor) -> None:
    if tensor.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("expected FP32 or BF16 tensor")
    if tensor.device.type not in ("cpu", "cuda"):
        raise RuntimeError("expected CPU or CUDA/ROCm tensor")


def weight_before_down(
    activation_fp32: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Multiply FP32 SwiGLU activation by route weights BEFORE the BF16 cast.

    This is the SGLang BF16 placement candidate; callers must provide the
    uncast FP32 activation, not an already-rounded BF16 expert activation.
    """
    _check_float(activation_fp32)
    _check_float(weights)
    if activation_fp32.ndim != 2 or activation_fp32.dtype != torch.float32:
        raise ValueError("expected FP32 [routed_rows, intermediate] activation")
    if weights.shape != (activation_fp32.shape[0],) or weights.device != activation_fp32.device:
        raise ValueError("weights must match routed rows and device")
    return (activation_fp32 * weights.float()[:, None]).bfloat16()


def merge_expert_rows(
    grouped_rows: torch.Tensor,
    source_to_grouped: torch.Tensor,
    *,
    route_weights: torch.Tensor | None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Restore source [token, slot] order and left-fold eight FP32 rows.

    All rows must have arrived from the expert exchange; the mapping is a
    permutation of exactly T*8 rows. Pass route_weights=None ONLY for rows
    already weighted before/downstream of the expert projection. Otherwise
    weights are applied here, after the down projection. Those two profiles
    have different rounding boundaries and are not interchangeable.
    Metadata validation is eager; distributed transport is a separate task.
    """
    _check_float(grouped_rows)
    if grouped_rows.ndim != 2 or not grouped_rows.shape[1]:
        raise ValueError("expected [routed_rows, hidden] expert outputs")
    if source_to_grouped.ndim != 2 or source_to_grouped.shape[1] != TOP_K:
        raise ValueError("expected [tokens, 8] source-to-grouped mapping")
    if source_to_grouped.dtype not in (torch.int32, torch.int64):
        raise TypeError("mapping must contain integer indices")
    if source_to_grouped.device != grouped_rows.device:
        raise ValueError("mapping must share the expert device")
    if out_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("output must be BF16 or FP32")
    flat = source_to_grouped.long().reshape(-1)
    if flat.numel() != grouped_rows.shape[0] or not torch.equal(
        flat.sort().values, torch.arange(flat.numel(), device=flat.device)
    ):
        raise ValueError("mapping must be a complete permutation of routed rows")
    if route_weights is not None:
        _check_float(route_weights)
        if route_weights.shape != source_to_grouped.shape or route_weights.device != flat.device:
            raise ValueError("weights must match source slots and device")
    rows = grouped_rows.index_select(0, flat).reshape(-1, TOP_K, grouped_rows.shape[1])
    acc = torch.zeros_like(rows[:, 0], dtype=torch.float32)
    for slot in range(TOP_K):
        value = rows[:, slot].float()
        if route_weights is not None:
            value = value * route_weights[:, slot : slot + 1].float()
        acc = acc + value
    return acc.to(out_dtype)


def update_block_residual(
    partial: torch.Tensor, attention: torch.Tensor, mlp: torch.Tensor, *, block_boundary: bool
) -> torch.Tensor:
    """Miles block update after attention/MLP short conv; cast exactly once.

    Boundary layers start a new partial (attention + MLP); the caller retains
    the old partial in the completed-block bank. Other layers left-associate
    partial + attention + MLP in FP32. No input tensor is mutated.
    """
    for tensor in (partial, attention, mlp):
        _check_float(tensor)
        if tensor.shape != partial.shape or tensor.device != partial.device:
            raise ValueError("residual tensors must share geometry and device")
    if block_boundary:
        result = attention.float() + mlp.float()
    else:
        result = (partial.float() + attention.float()) + mlp.float()
    return result.to(partial.dtype)
