# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Decode RMSNorm with the pinned PyTorch ROCm eager reduction order.

Read strided Q/K directly and round a fused residual addition to BF16 before
normalization. The four-value thread accumulators, 64-lane reductions and
four-warp merge reproduce PyTorch 2.12's vectorized RMSNorm, including its
separate multiply/add instructions. This is an inference-only implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _rmsnorm(
    X,
    R,
    W,
    Y,
    U,
    S0: tl.constexpr,
    S1: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    ADD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    thread = tl.arange(0, 256)
    base = (row // H) * S0 + (row % H) * S1
    acc = tl.full((256,), 0.0, tl.float32)
    for tile in range(tl.cdiv(N, 1024)):
        for element in tl.static_range(4):
            offset = thread * 4 + tile * 1024 + element
            value = tl.load(X + base + offset, offset < N, 0).to(tl.float32)
            if ADD:
                residual = tl.load(R + row * N + offset, offset < N, 0).to(tl.float32)
                value = (value + residual).to(X.dtype.element_ty).to(tl.float32)
            acc = acc + value * value
    # Explicit shuffle-down tree: tl.sum alone can choose a different tree.
    for shift in tl.static_range(5, -1, -1):
        peer = (thread // 64) * 64 + ((thread % 64 + (1 << shift)) % 64)
        acc = acc + tl.gather(acc, peer, 0)
    warps = tl.gather(acc, tl.arange(0, 4) * 64, 0)
    pairs = tl.sum(tl.reshape(warps, (2, 2)), 0)
    variance = tl.sum(pairs, 0) / N  # (warp0 + warp2) + (warp1 + warp3)
    rstd = libdevice.rsqrt(variance + EPS)
    offsets = tl.arange(0, BLOCK)
    value = tl.load(X + base + offsets, offsets < N, 0).to(tl.float32)
    if ADD:
        residual = tl.load(R + row * N + offsets, offsets < N, 0).to(tl.float32)
        value = (value + residual).to(X.dtype.element_ty).to(tl.float32)
        tl.store(U + row * N + offsets, value, offsets < N)
    weight = tl.load(W + offsets, offsets < N, 0).to(tl.float32)
    tl.store(Y + row * N + offsets, weight * (rstd * value), offsets < N)


def supports(x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor | None) -> bool:
    """Restrict the shortcut to the eager contract and layouts validated here."""
    return (
        torch.version.hip is not None
        and torch.__version__.split("+")[0].startswith("2.12.")
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim in (2, 3)
        and x.stride(-1) == 1
        and 4 <= x.shape[-1] <= 8192
        and x.shape[-1] & (x.shape[-1] - 1) == 0
        and 0 < x.numel() // x.shape[-1] <= 1024
        and weight.shape == (x.shape[-1],)
        and weight.is_contiguous()
        and weight.device == x.device
        and weight.dtype == x.dtype
        and (
            residual is None
            or (
                residual.shape == x.shape
                and residual.is_contiguous()
                and residual.device == x.device
                and residual.dtype == x.dtype
            )
        )
        and str(torch.cuda.get_device_properties(x.device).gcnArchName).startswith("gfx942")
    )


def forward(x, weight, eps, residual=None):
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    updated = torch.empty_like(output) if residual is not None else output
    heads = x.shape[-2] if x.ndim == 3 else 1
    stride0 = x.stride(-3) if x.ndim == 3 else x.stride(0)
    _rmsnorm[(x.numel() // x.shape[-1],)](
        x,
        residual,
        weight,
        output,
        updated,
        stride0,
        x.stride(-2),
        heads,
        x.shape[-1],
        float(eps),
        residual is not None,
        triton.next_power_of_2(x.shape[-1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return (output, updated) if residual is not None else output
