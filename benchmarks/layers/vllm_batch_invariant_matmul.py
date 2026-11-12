# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone vLLM batch-invariant 2D persistent matmul benchmark kernel.

Vendored from vLLM PR https://github.com/vllm-project/vllm/pull/53247 at
merge commit 7797b6022c129b862e45ae6aed08822e65d1bccb:
https://github.com/vllm-project/vllm/blob/7797b6022c129b862e45ae6aed08822e65d1bccb/vllm/model_executor/layers/batch_invariant.py

Standalone modifications are intentionally narrow: direct Triton imports
replace ``vllm.triton_utils``, the SM count comes from
``torch.cuda.get_device_properties``, and configuration metadata is exposed so
the benchmark can report whether a tuned table entry or the upstream default
was selected.  The persistent kernel, accumulation order, launch grid, and
matmul configuration behavior are preserved from upstream.
"""

from collections.abc import Callable
from typing import Any

import torch
import triton
import triton.language as tl

from benchmarks.layers.vllm_batch_invariant_configs import (
    _get_matmul_config,
    _get_tuned_matmul_arch_family,
)

VLLM_BATCH_INVARIANT_PR_URL = "https://github.com/vllm-project/vllm/pull/53247"
VLLM_BATCH_INVARIANT_SOURCE_SHA = "7797b6022c129b862e45ae6aed08822e65d1bccb"
VLLM_BATCH_INVARIANT_SOURCE_URL = (
    "https://github.com/vllm-project/vllm/blob/"
    f"{VLLM_BATCH_INVARIANT_SOURCE_SHA}/"
    "vllm/model_executor/layers/batch_invariant.py"
)

_FP16_BLOCK_SIZE_N = 256
_FP32_BLOCK_SIZE_N = 128
_FP32_NUM_STAGES = 3

__all__ = [
    "VLLM_BATCH_INVARIANT_PR_URL",
    "VLLM_BATCH_INVARIANT_SOURCE_SHA",
    "VLLM_BATCH_INVARIANT_SOURCE_URL",
    "matmul_config_metadata",
    "matmul_kernel_persistent",
    "matmul_persistent",
]


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: dict[str, Any]
) -> dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"

    bytes_per_elem = args["c_ptr"].element_size()
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


def _default_matmul_configs(N: int) -> dict[torch.dtype, dict[str, int]]:
    return {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": _FP16_BLOCK_SIZE_N,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": _FP32_BLOCK_SIZE_N if N == 1 else 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": _FP32_NUM_STAGES if N == 1 else 3,
            "num_warps": 8,
        },
    }


def _select_matmul_config(
    M: int, N: int, K: int, dtype: torch.dtype
) -> tuple[dict[str, int], bool]:
    configs = _default_matmul_configs(N)
    default = configs[dtype]
    selected = _get_matmul_config(M, N, K, dtype, default)
    return selected, selected is not default


def matmul_config_metadata(M: int, N: int, K: int, dtype: torch.dtype) -> dict[str, Any]:
    """Return the exact launch config and whether it came from the tuned table."""

    config, is_tuned = _select_matmul_config(M, N, K, dtype)
    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    return {
        "shape": {"M": M, "N": N, "K": K},
        "dtype": str(dtype).removeprefix("torch."),
        "config": dict(config),
        "selection": "tuned" if is_tuned else "default",
        "is_tuned": is_tuned,
        "device_capability": list(capability) if capability is not None else None,
        "arch_family": _get_tuned_matmul_arch_family(capability),
    }


def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert (
        bias is None or bias.dim() == 1
    ), "Currently assuming bias is 1D, let Horace know if you run into this"
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    config, _ = _select_matmul_config(M, N, K, dtype)
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        bias,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **config,
    )
    return c
