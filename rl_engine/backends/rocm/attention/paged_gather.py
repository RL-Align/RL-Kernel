# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fused logical paged-KV gather for the strict ROCm Attention runtime."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _paged_kv_gather_bhsd_kernel(
        k_cache,
        v_cache,
        page_rows,
        k_out,
        v_out,
        total_elements,
        tokens,
        heads,
        head_dim: tl.constexpr,
        page_size,
        k_stride_page,
        k_stride_token,
        k_stride_head,
        k_stride_dim,
        v_stride_page,
        v_stride_token,
        v_stride_head,
        v_stride_dim,
        page_stride_row,
        page_stride_col,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_elements
        dim = offsets % head_dim
        rem = offsets // head_dim
        token = rem % tokens
        rem = rem // tokens
        head = rem % heads
        row = rem // heads
        logical_page = token // page_size
        page_offset = token % page_size
        physical_page = tl.load(
            page_rows + row * page_stride_row + logical_page * page_stride_col,
            mask=mask,
            other=0,
        ).to(tl.int64)
        k_offsets = (
            physical_page * k_stride_page
            + page_offset * k_stride_token
            + head * k_stride_head
            + dim * k_stride_dim
        )
        v_offsets = (
            physical_page * v_stride_page
            + page_offset * v_stride_token
            + head * v_stride_head
            + dim * v_stride_dim
        )
        k_value = tl.load(k_cache + k_offsets, mask=mask)
        v_value = tl.load(v_cache + v_offsets, mask=mask)
        tl.store(k_out + offsets, k_value, mask=mask)
        tl.store(v_out + offsets, v_value, mask=mask)


def fused_paged_kv_gather_bhsd(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_rows: torch.Tensor,
    page_count: int,
    *,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K and V together into contiguous ``[B, H, S, D]`` buffers.

    Each KV group is sequence-contiguous. Its ``[B, 1, S, D]`` slice can
    therefore be transposed to AITER's ``[B, S, 1, D]`` as a view, avoiding
    the second layout materialization that a token-major gather would need.
    """

    if not _TRITON_AVAILABLE:
        raise RuntimeError("strict ROCm fused paged gather requires Triton")
    if page_count <= 0 or page_rows.ndim != 2 or page_rows.size(1) < page_count:
        raise ValueError("page_rows must provide a positive page count")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("paged K/V cache must use [pages, page_size, heads, dim]")
    rows = page_rows.size(0)
    tokens = page_count * k_cache.size(1)
    expected = (rows, k_cache.size(2), tokens, k_cache.size(3))
    if k_out.shape != expected or v_out.shape != expected:
        raise ValueError("paged gather output buffers have the wrong BHSD shape")
    if not k_out.is_contiguous() or not v_out.is_contiguous():
        raise ValueError("paged gather output buffers must be contiguous")
    if not (k_cache.device == v_cache.device == page_rows.device == k_out.device == v_out.device):
        raise ValueError("paged gather tensors must share one device")
    if k_out.dtype != k_cache.dtype or v_out.dtype != v_cache.dtype:
        raise ValueError("paged gather output dtype must match the cache")

    total = k_out.numel()
    block = 256
    _paged_kv_gather_bhsd_kernel[(triton.cdiv(total, block),)](
        k_cache,
        v_cache,
        page_rows,
        k_out,
        v_out,
        total_elements=total,
        tokens=tokens,
        heads=k_cache.size(2),
        head_dim=k_cache.size(3),
        page_size=k_cache.size(1),
        k_stride_page=k_cache.stride(0),
        k_stride_token=k_cache.stride(1),
        k_stride_head=k_cache.stride(2),
        k_stride_dim=k_cache.stride(3),
        v_stride_page=v_cache.stride(0),
        v_stride_token=v_cache.stride(1),
        v_stride_head=v_cache.stride(2),
        v_stride_dim=v_cache.stride(3),
        page_stride_row=page_rows.stride(0),
        page_stride_col=page_rows.stride(1),
        BLOCK=block,
        num_warps=4,
    )
    return k_out, v_out


_WARMED_GATHER_VARIANTS: set[tuple[int, torch.dtype, int]] = set()


def warmup_fused_paged_kv_gather(
    *,
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
) -> None:
    """Compile the production gather variant before rollout timing starts."""

    device = torch.device(device)
    if device.type != "cuda" or not _TRITON_AVAILABLE:
        return
    device_index = torch.cuda.current_device() if device.index is None else device.index
    key = (device_index, dtype, head_dim)
    if key in _WARMED_GATHER_VARIANTS:
        return

    page_size = 16
    cache_shape = (1, page_size, 1, head_dim)
    output_shape = (1, 1, page_size, head_dim)
    k_cache = torch.empty(cache_shape, dtype=dtype, device=device)
    v_cache = torch.empty_like(k_cache)
    page_rows = torch.zeros((1, 1), dtype=torch.int32, device=device)
    k_out = torch.empty(output_shape, dtype=dtype, device=device)
    v_out = torch.empty_like(k_out)
    fused_paged_kv_gather_bhsd(
        k_cache,
        v_cache,
        page_rows,
        1,
        k_out=k_out,
        v_out=v_out,
    )
    torch.cuda.synchronize(device)
    _WARMED_GATHER_VARIANTS.add(key)


__all__ = ["fused_paged_kv_gather_bhsd", "warmup_fused_paged_kv_gather"]
