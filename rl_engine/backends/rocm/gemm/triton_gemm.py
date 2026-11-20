# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Modifications copyright (c) 2026 RL-Kernel Contributors
"""Batch-invariant linear on ROCm, y = x @ w^T + bias: aiter's gemm_a16w16 Triton kernel.

Copied from aiter 326cffa: ``_gemm_a16_w16_kernel`` from
aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py, ``remap_xcd`` and
``pid_grid`` from aiter/ops/triton/utils/_triton/pid_preprocessing.py, and the
Triton launcher of aiter/ops/triton/gemm/basic/gemm_a16w16.py. aiter's
arithmetic is kept; what changes:

1. Batch invariance by construction. K is never split (aiter's split-K path and
   its reduce kernel are removed) and the MFMA shape is fixed
   (``matrix_instr_nonkdim=16``, ``kpack=1``), so the order in which an output
   element is accumulated does not depend on M or on a launch config. Inside a
   tile there is only ``tl.dot``, no ``tl.sum``. The K tile is a launch
   parameter: on gfx942 it was measured not to change the bits.
2. Dynamic shape. M and N are runtime arguments and are not specialized
   (``do_not_specialize``, as in aiter); aiter's compile-time ``EVEN_MN`` switch
   is removed (always the masked path), and ``EVEN_K`` depends on K only. The
   launch config is chosen by M from CONFIGS; each entry compiles once per
   process, and ``warmup`` compiles them all.
3. Activations, ``skip_reduce`` and the persistent and gluon paths are removed.

Backward: dx and dw use det_gemm's fixed-K-order Triton GEMM in fp32; dbias is a
PyTorch sum over the rows. Each backward is recorded with ``record_backward``
("batch_invariant_linear").

The op runs on gfx942 only: its constructor raises on any other architecture.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.utils.logger import logger

# The MFMA shape and kpack are part of the arithmetic: changing them changes the bits.
# They, BLOCK_K and CONFIGS are validated on gfx942 only; the op's constructor raises on
# any other architecture.
VALIDATED_ARCH = "gfx942"
DOT_OPTIONS = {"matrix_instr_nonkdim": 16, "kpack": 1}
# The K tile of a launch config without BLOCK_SIZE_K. On gfx942, K tiles from 16 to
# 256 give the same bits (docs/operators/dsv4-batch-invariant-linear.md).
BLOCK_K = 64


def _config(bm, bn, bk, group, warps, stages, waves, cache):
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": group,
        "num_warps": warps,
        "num_stages": stages,
        "waves_per_eu": waves,
        "cache_modifier": cache,
    }


# ((largest M, launch config), ...): the op uses the first entry whose largest M is at
# least its M. A launch config sets the tile shape, the grouping of output tiles into
# workgroups and the K tile, not the bits. The fastest of a set of candidate configs
# per M range for the DeepSeek-V4-Flash lm_head (N = 129280, K = 4096, fp32 output) on
# gfx942; benchmarks/operators/gemm/benchmark_batch_invariant_linear.py times the op
# against aiter's gemm_a16w16.
CONFIGS = (
    (16, _config(16, 64, 128, 1, 2, 1, 4, ".cg")),
    (32, _config(32, 64, 128, 1, 2, 1, 0, ".cg")),
    (64, _config(64, 128, 128, 1, 8, 1, 0, None)),
    (256, _config(128, 128, 64, 8, 8, 2, 1, None)),
    (512, _config(128, 128, 64, 1, 8, 2, 1, None)),
    (None, _config(128, 256, 32, 4, 4, 2, 2, None)),
)

SUPPORTED_OUT_DTYPES = (torch.bfloat16, torch.float32)


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    # pid remapping on XCDs
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    if tall_xcds == 0:
        tall_xcds = tl.cast(NUM_XCDS, tall_xcds.type)
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    return pid


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: tl.constexpr: default is 1
    """
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@triton.jit(do_not_specialize=["M", "N"])
def _gemm_a16_w16_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    cache_modifier: tl.constexpr,
    ADD_BIAS: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid = remap_xcd(pid, num_pid_m * num_pid_n, NUM_XCDS=8)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Create pointers for first block of A and B input matrices. Rows past M and
    # columns past N wrap around; their results are not stored.
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if ADD_BIAS:
        accumulator = tl.load(bias_ptr + offs_bn * stride_bias).to(dtype=tl.float32)
        accumulator = tl.broadcast_to(accumulator[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, cache_modifier=cache_modifier)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(
                b_ptrs,
                mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                other=0.0,
                cache_modifier=cache_modifier,
            )
        accumulator = tl.dot(a, b, acc=accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Write back the block of the output matrix C with masks.
    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def distinct_configs(configs=CONFIGS) -> list[dict]:
    """The different launch configs of a table; each compiles once per process and shape."""
    result = []
    for _, config in configs:
        if config not in result:
            result.append(config)
    return result


def select_config(m: int, configs=CONFIGS) -> dict:
    """The launch config for M rows from a table."""
    for m_max, config in configs:
        if m_max is None or m <= m_max:
            return config
    raise AssertionError("a config table must end with an entry for every M")


def gemm_a16w16(x, w, bias=None, dtype=torch.bfloat16, config=None):
    """y = x @ w^T (+ bias): x [M, K] and w [N, K] bf16, bias [N]; y [M, N] in ``dtype``.

    ``config`` replaces the config CONFIGS gives for M (launch parameters only; the
    bits do not change)."""
    M, K = x.shape
    N, _ = w.shape
    w = w.T
    cfg = dict(select_config(M) if config is None else config)
    block_k = cfg.pop("BLOCK_SIZE_K", BLOCK_K)
    y = torch.empty((M, N), dtype=dtype, device=x.device)
    if M == 0:
        return y
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),)
    _gemm_a16_w16_kernel[grid](
        x,
        w,
        bias,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        0 if bias is None else bias.stride(0),
        BLOCK_SIZE_K=block_k,
        EVEN_K=K % block_k == 0,
        ADD_BIAS=bias is not None,
        **cfg,
        **DOT_OPTIONS,
    )
    return y


def check_linear_inputs(x, weight, bias, out_dtype):
    """Raise for an input this op does not support, instead of running it unvalidated:
    x [..., K] and weight [N, K] in bf16, bias [N] in out_dtype, out_dtype bf16 or fp32."""
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError(f"x and weight must be bf16, got {x.dtype} and {weight.dtype}")
    if out_dtype not in SUPPORTED_OUT_DTYPES:
        raise TypeError(f"out_dtype must be one of {SUPPORTED_OUT_DTYPES}, got {out_dtype}")
    if x.dim() < 1 or weight.dim() != 2 or weight.shape[1] != x.shape[-1]:
        raise ValueError(
            f"x [..., K] and weight [N, K] do not match: {tuple(x.shape)}, {tuple(weight.shape)}"
        )
    if bias is not None and (bias.dtype != out_dtype or tuple(bias.shape) != (weight.shape[0],)):
        raise ValueError(
            f"bias must be [{weight.shape[0]}] in {out_dtype}, "
            f"got {tuple(bias.shape)} in {bias.dtype}"
        )
    devices = {t.device for t in (x, weight, bias) if t is not None}
    if len(devices) != 1:
        raise ValueError(f"x, weight and bias must be on one device, got {sorted(map(str, devices))}")


class _LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, weight, bias, out_dtype):
        ctx.save_for_backward(x2d, weight)
        ctx.has_bias = bias is not None
        return gemm_a16w16(x2d, weight, bias, dtype=out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm

        x2d, weight = ctx.saved_tensors
        dx = dw = dbias = None
        if x2d.shape[0] == 0:  # no rows: no launch, zero parameter gradients
            dx = torch.zeros_like(x2d) if ctx.needs_input_grad[0] else None
            dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
            if ctx.has_bias and ctx.needs_input_grad[2]:
                dbias = grad_out.new_zeros(weight.shape[0])
            return dx, dw, dbias, None
        # The fp32 det_gemm kernel converts its bf16 or fp32 inputs to fp32 itself.
        if ctx.needs_input_grad[0]:
            dx = _triton_gemm(grad_out, weight, output_dtype=torch.float32)
            dx = dx.to(x2d.dtype)
        if ctx.needs_input_grad[1]:
            dw = _triton_gemm(grad_out.t(), x2d, output_dtype=torch.float32)
            dw = dw.to(weight.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            dbias = grad_out.sum(0, dtype=torch.float32).to(grad_out.dtype)
        record_backward(
            "batch_invariant_linear",
            kernel_id=f"{_triton_gemm.__module__}.{_triton_gemm.__name__}",
            impl="triton_det_gemm_fp32",
            family="rocm-triton",
        )
        return dx, dw, dbias, None


class TritonBatchInvariantLinearOp:
    """y = x @ weight^T (+ bias), batch invariant; see the module docstring."""

    def __init__(self):
        if torch.version.hip is None or not torch.cuda.is_available():
            raise RuntimeError("The Triton batch-invariant linear requires a ROCm GPU.")
        device = torch.cuda.current_device()
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        if arch != VALIDATED_ARCH:
            raise RuntimeError(
                f"The Triton batch-invariant linear is validated on {VALIDATED_ARCH} only; "
                f"this GPU is {arch}."
            )
        logger.info("Linked RL-Kernel's batch-invariant Triton linear (from aiter gemm_a16w16).")

    def __call__(self, x, weight, bias=None, *, out_dtype=torch.bfloat16):
        """x [..., K] and weight [N, K] in bf16; bias [N]; returns [..., N] in out_dtype."""
        check_linear_inputs(x, weight, bias, out_dtype)
        k = x.shape[-1]
        lead = x.shape[:-1]
        x2d = x.reshape(-1, k)  # the kernel reads x, weight and bias through their strides
        y = _LinearFn.apply(x2d, weight, bias, out_dtype)
        return y.reshape(*lead, weight.shape[0])

    @staticmethod
    def warmup(weight, *, out_dtype=torch.bfloat16, bias=None):
        """Compile the kernel for every entry of CONFIGS with ``weight``'s N and K."""
        k = weight.shape[1]
        for m_max, _ in CONFIGS:
            m = m_max or CONFIGS[-2][0] + 1
            x = torch.zeros((m, k), dtype=weight.dtype, device=weight.device)
            gemm_a16w16(x, weight, bias, dtype=out_dtype)


KERNELS = (_gemm_a16_w16_kernel,)
