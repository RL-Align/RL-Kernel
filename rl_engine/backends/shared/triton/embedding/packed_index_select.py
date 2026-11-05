# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 packed row selection and atomic-free, batch-local gradient scatter-add."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.reference.minimax_h3.packed_index_select import validate_packed_index_select

MIN_CUDA_SM = 80
BLOCK_SIZE = 256
KERNEL_ID = __name__ + "._select_forward"
BACKWARD_KERNEL_ID = "torch.argsort[stable]+" + __name__ + "._select_backward"
BACKWARD_IMPL = "stable_segment_sequential_fp32_cast_once"


@triton.jit
def _select_forward(X, Index, Y, N, K, D: tl.constexpr, XB, XS, XD, IS, BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    batch, position, channel = offset // (K * D), (offset // D) % K, offset % D
    row = tl.load(Index + position * IS, offset < N, other=0).to(tl.int64)
    value = tl.load(X + batch * XB + row * XS + channel * XD, offset < N, other=0)
    tl.store(Y + offset, value, offset < N)


@triton.jit
def _select_backward(
    SortedIndex,
    Order,
    Grad,
    DX,
    K,
    S,
    D: tl.constexpr,
    GB,
    GK,
    GD,
    BLOCK_D: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    position, batch = program % K, program // K
    channel = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    row = tl.load(SortedIndex + position).to(tl.int64)
    previous = tl.load(SortedIndex + position - 1, position > 0, other=-1)
    # Only the first occurrence owns the destination. Other CTAs do no work.
    if row != previous:
        accumulator = tl.full((BLOCK_D,), 0, tl.float32)
        cursor = position
        while cursor < K:
            current = tl.load(SortedIndex + cursor)
            if current != row:
                break
            source = tl.load(Order + cursor)
            value = tl.load(
                Grad + batch * GB + source * GK + channel * GD, channel < D, other=0
            ).to(tl.float32)
            accumulator = accumulator + value
            cursor += 1
        tl.store(DX + (batch * S + row) * D + channel, accumulator, channel < D)


class _PackedIndexSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index):
        batch, _, width = x.shape
        count = index.numel()
        output = torch.empty((batch, count, width), device=x.device, dtype=x.dtype)
        with torch.cuda.device(x.device):
            if output.numel():
                _select_forward[(triton.cdiv(output.numel(), BLOCK_SIZE),)](
                    x,
                    index,
                    output,
                    output.numel(),
                    count,
                    width,
                    *x.stride(),
                    index.stride(0),
                    BLOCK=BLOCK_SIZE,
                )
        ctx.save_for_backward(index)
        ctx.input_shape = x.shape
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        batch, sequence, width = ctx.input_shape
        with torch.cuda.device(grad.device):
            result = torch.zeros(ctx.input_shape, device=grad.device, dtype=grad.dtype)
            if grad.numel():
                # Sort only integer metadata; read gradients directly at original strides.
                # Stable ties preserve j order independently of batch/launch geometry.
                order = torch.argsort(index, stable=True)
                sorted_index = index.index_select(0, order)
                block_d = min(triton.next_power_of_2(width), BLOCK_SIZE)
                _select_backward[(index.numel() * batch, triton.cdiv(width, block_d))](
                    sorted_index,
                    order,
                    grad,
                    result,
                    index.numel(),
                    sequence,
                    width,
                    *grad.stride(),
                    BLOCK_D=block_d,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        record_backward(
            "packed_index_select",
            kernel_id=BACKWARD_KERNEL_ID if grad.numel() else "torch.zeros",
            impl=BACKWARD_IMPL if grad.numel() else "empty_zero_gradient",
            family="triton",
        )
        return result, None


class TritonPackedIndexSelectOp:
    """``x [B,S,D], index [K] -> contiguous y [B,K,D]`` on NVIDIA CUDA.

    Strided inputs, indices and upstream gradients are supported. FP32, FP16
    and BF16 values are copied without arithmetic. Backward zero-initializes
    dX and stably groups equal indices; one CTA per (batch, row, channel tile)
    folds their gradients in increasing output position j using FP32 additions,
    starting at +0, and casts once on store. No floating-point atomics, split
    reduction, cross-batch sums or reordered gradient buffer are used.

    This guarantees repeatability and batch invariance for a fixed per-row
    occurrence order. Reordering duplicate occurrences or splitting their sum
    into separate backward calls can change rounding. First-order autograd only.
    CUDA SM80+ is eligible; ROCm is not qualified and fails closed.
    """

    op_class = "reduction"
    is_batch_invariant = True
    kernel_id = KERNEL_ID
    backward_impl = BACKWARD_IMPL

    def __call__(self, x, index):
        return self.forward(x, index)

    def forward(self, x, index):
        if x.device.type != "cuda" or torch.version.hip is not None:
            raise RuntimeError("packed_index_select requires NVIDIA CUDA; no fallback")
        major, minor = torch.cuda.get_device_capability(x.device)
        if 10 * major + minor < MIN_CUDA_SM:
            raise RuntimeError(f"packed_index_select requires CUDA SM{MIN_CUDA_SM}+")
        validate_packed_index_select(x, index)
        return _PackedIndexSelect.apply(x, index)
