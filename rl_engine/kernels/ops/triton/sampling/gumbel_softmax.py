# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.pytorch.sampling.gumbel_softmax import _validate_gumbel_softmax_inputs

_MAX_BLOCK_V = 131072
_CHUNK_V = 65536
_MAX_RAND_OFFSET = 2**31 - 1


def _launch_config(vocab_size: int) -> tuple[int, int]:
    block_v = triton.next_power_of_2(vocab_size)
    if block_v > _MAX_BLOCK_V:
        raise ValueError(
            f"vocab size {vocab_size} exceeds TritonGumbelSoftmaxOp limit {_MAX_BLOCK_V}"
        )
    num_warps = 32 if block_v >= 65536 else 16 if block_v >= 32768 else 8
    return block_v, num_warps


def _needs_precomputed_gumbels(logits: torch.Tensor) -> bool:
    return (
        triton.next_power_of_2(logits.shape[-1]) > _MAX_BLOCK_V or logits.numel() > _MAX_RAND_OFFSET
    )


def _sample_gumbels_like(logits: torch.Tensor, seed: Optional[int]) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(int(seed))
    return (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_(generator=generator)
        .log()
    )


def _num_chunks(vocab_size: int) -> int:
    return triton.cdiv(vocab_size, _CHUNK_V)


@triton.jit
def _gumbel_softmax_fwd_kernel(
    logits_ptr,
    gumbels_ptr,
    y_soft_ptr,
    out_ptr,
    seed,
    n_rows,
    V: tl.constexpr,
    tau: tl.constexpr,
    HAS_GUMBELS: tl.constexpr,
    HARD: tl.constexpr,
    STORE_OUT: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_V)
    mask = cols < V
    row_off = row.to(tl.int64) * V

    logits = tl.load(logits_ptr + row_off + cols, mask=mask, other=-float("inf")).to(tl.float32)
    if HAS_GUMBELS:
        gumbels = tl.load(gumbels_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    else:
        u = tl.rand(seed, row_off + cols)
        u = tl.minimum(tl.maximum(u, 1.0e-20), 1.0 - 1.0e-7)
        gumbels = -tl.log(-tl.log(u))

    z = tl.where(mask, (logits + gumbels) / tau, -float("inf"))
    z_max = tl.max(z, axis=0)
    exp_z = tl.exp(z - z_max)
    denom = tl.sum(exp_z, axis=0)
    y = tl.where(mask, exp_z / denom, 0.0)

    tl.store(y_soft_ptr + row_off + cols, y, mask=mask)
    if HARD:
        # Tie-break to the first maximum so each row is exactly one-hot.
        is_max = z == z_max
        first_max = tl.min(tl.where(is_max, cols, BLOCK_V), axis=0)
        out = tl.where(cols == first_max, 1.0, 0.0)
        tl.store(out_ptr + row_off + cols, out, mask=mask)
    elif STORE_OUT:
        tl.store(out_ptr + row_off + cols, y, mask=mask)


@triton.jit
def _gumbel_softmax_hard_nograd_kernel(
    logits_ptr,
    gumbels_ptr,
    out_ptr,
    seed,
    n_rows,
    V: tl.constexpr,
    HAS_GUMBELS: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_V)
    mask = cols < V
    row_off = row.to(tl.int64) * V

    logits = tl.load(logits_ptr + row_off + cols, mask=mask, other=-float("inf")).to(tl.float32)
    if HAS_GUMBELS:
        gumbels = tl.load(gumbels_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    else:
        u = tl.rand(seed, row_off + cols)
        u = tl.minimum(tl.maximum(u, 1.0e-20), 1.0 - 1.0e-7)
        gumbels = -tl.log(-tl.log(u))

    z = tl.where(mask, logits + gumbels, -float("inf"))
    z_max = tl.max(z, axis=0)
    is_max = z == z_max
    first_max = tl.min(tl.where(is_max, cols, BLOCK_V), axis=0)
    out = tl.where(cols == first_max, 1.0, 0.0)
    tl.store(out_ptr + row_off + cols, out, mask=mask)


@triton.jit
def _chunked_stats_kernel(
    logits_ptr,
    gumbels_ptr,
    chunk_max_ptr,
    chunk_arg_ptr,
    V: tl.constexpr,
    tau: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    CHUNK_V: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = tl.arange(0, CHUNK_V)
    vocab_cols = chunk * CHUNK_V + cols
    mask = vocab_cols < V
    row_off = row.to(tl.int64) * V

    logits = tl.load(logits_ptr + row_off + vocab_cols, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    gumbels = tl.load(gumbels_ptr + row_off + vocab_cols, mask=mask, other=0.0).to(tl.float32)
    z = tl.where(mask, (logits + gumbels) / tau, -float("inf"))
    z_max = tl.max(z, axis=0)
    first_max = tl.min(tl.where(z == z_max, vocab_cols, V), axis=0)

    stats_off = row * N_CHUNKS + chunk
    tl.store(chunk_max_ptr + stats_off, z_max)
    tl.store(chunk_arg_ptr + stats_off, first_max)


@triton.jit
def _chunked_global_stats_kernel(
    chunk_max_ptr,
    chunk_arg_ptr,
    row_max_ptr,
    row_arg_ptr,
    N_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = chunks < N_CHUNKS
    stats_off = row * N_CHUNKS + chunks

    chunk_max = tl.load(chunk_max_ptr + stats_off, mask=mask, other=-float("inf"))
    chunk_arg = tl.load(chunk_arg_ptr + stats_off, mask=mask, other=2147483647)
    global_max = tl.max(chunk_max, axis=0)
    global_arg = tl.min(tl.where(chunk_max == global_max, chunk_arg, 2147483647), axis=0)

    tl.store(row_max_ptr + row, global_max)
    tl.store(row_arg_ptr + row, global_arg)


@triton.jit
def _chunked_sum_kernel(
    logits_ptr,
    gumbels_ptr,
    row_max_ptr,
    chunk_sum_ptr,
    V: tl.constexpr,
    tau: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    CHUNK_V: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = tl.arange(0, CHUNK_V)
    vocab_cols = chunk * CHUNK_V + cols
    mask = vocab_cols < V
    row_off = row.to(tl.int64) * V

    logits = tl.load(logits_ptr + row_off + vocab_cols, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    gumbels = tl.load(gumbels_ptr + row_off + vocab_cols, mask=mask, other=0.0).to(tl.float32)
    row_max = tl.load(row_max_ptr + row)
    z = tl.where(mask, (logits + gumbels) / tau, -float("inf"))
    exp_z = tl.exp(z - row_max)
    chunk_sum = tl.sum(tl.where(mask, exp_z, 0.0), axis=0)

    tl.store(chunk_sum_ptr + row * N_CHUNKS + chunk, chunk_sum)


@triton.jit
def _chunked_output_kernel(
    logits_ptr,
    gumbels_ptr,
    row_max_ptr,
    row_arg_ptr,
    chunk_sum_ptr,
    y_soft_ptr,
    out_ptr,
    V: tl.constexpr,
    tau: tl.constexpr,
    HARD: tl.constexpr,
    STORE_OUT: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    CHUNK_V: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = tl.arange(0, CHUNK_V)
    vocab_cols = chunk * CHUNK_V + cols
    mask = vocab_cols < V
    row_off = row.to(tl.int64) * V

    chunks = tl.arange(0, BLOCK_CHUNKS)
    chunk_mask = chunks < N_CHUNKS
    denom = tl.sum(
        tl.load(chunk_sum_ptr + row * N_CHUNKS + chunks, mask=chunk_mask, other=0.0),
        axis=0,
    )
    row_max = tl.load(row_max_ptr + row)
    logits = tl.load(logits_ptr + row_off + vocab_cols, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    gumbels = tl.load(gumbels_ptr + row_off + vocab_cols, mask=mask, other=0.0).to(tl.float32)
    z = tl.where(mask, (logits + gumbels) / tau, -float("inf"))
    y = tl.where(mask, tl.exp(z - row_max) / denom, 0.0)

    tl.store(y_soft_ptr + row_off + vocab_cols, y, mask=mask)
    if HARD:
        row_arg = tl.load(row_arg_ptr + row)
        out = tl.where(vocab_cols == row_arg, 1.0, 0.0)
        tl.store(out_ptr + row_off + vocab_cols, out, mask=mask)
    elif STORE_OUT:
        tl.store(out_ptr + row_off + vocab_cols, y, mask=mask)


@triton.jit
def _chunked_hard_output_kernel(
    row_arg_ptr,
    out_ptr,
    V: tl.constexpr,
    CHUNK_V: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = tl.arange(0, CHUNK_V)
    vocab_cols = chunk * CHUNK_V + cols
    mask = vocab_cols < V
    row_arg = tl.load(row_arg_ptr + row)
    out = tl.where(vocab_cols == row_arg, 1.0, 0.0)
    tl.store(out_ptr + row.to(tl.int64) * V + vocab_cols, out, mask=mask)


class _GumbelSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, gumbels, tau: float, hard: bool, seed: int):
        V = logits.shape[-1]
        lead_shape = logits.shape[:-1]
        logits_2d = logits.contiguous().view(-1, V)
        gumbels_2d = gumbels.contiguous().view(-1, V) if gumbels is not None else logits_2d
        n_rows = logits_2d.shape[0]
        block_v, num_warps = _launch_config(V)

        y_soft = torch.empty_like(logits_2d)
        out = torch.empty_like(logits_2d) if hard else y_soft
        _gumbel_softmax_fwd_kernel[(n_rows,)](
            logits_2d,
            gumbels_2d,
            y_soft,
            out,
            int(seed),
            n_rows,
            V,
            float(tau),
            HAS_GUMBELS=gumbels is not None,
            HARD=bool(hard),
            STORE_OUT=out is not y_soft,
            BLOCK_V=block_v,
            num_warps=num_warps,
        )

        ctx.save_for_backward(y_soft)
        ctx.tau = float(tau)
        ctx.logits_shape = tuple(logits.shape)
        ctx.logits_dtype = logits.dtype
        return out.view(*lead_shape, V)

    @staticmethod
    def backward(ctx, grad_output):
        (y_soft,) = ctx.saved_tensors
        grad_2d = grad_output.contiguous().view_as(y_soft)
        grad_logits = torch._softmax_backward_data(
            grad_2d,
            y_soft,
            -1,
            ctx.logits_dtype,
        )
        if ctx.tau != 1.0:
            grad_logits = grad_logits / ctx.tau
        return grad_logits.view(ctx.logits_shape).to(ctx.logits_dtype), None, None, None, None


class _ChunkedGumbelSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, gumbels, tau: float, hard: bool):
        V = logits.shape[-1]
        lead_shape = logits.shape[:-1]
        logits_2d = logits.contiguous().view(-1, V)
        gumbels_2d = gumbels.contiguous().view(-1, V)
        n_rows = logits_2d.shape[0]
        n_chunks = _num_chunks(V)
        block_chunks = triton.next_power_of_2(n_chunks)

        chunk_max = torch.empty((n_rows, n_chunks), device=logits.device, dtype=torch.float32)
        chunk_arg = torch.empty((n_rows, n_chunks), device=logits.device, dtype=torch.int64)
        row_max = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
        row_arg = torch.empty((n_rows,), device=logits.device, dtype=torch.int64)
        chunk_sum = torch.empty((n_rows, n_chunks), device=logits.device, dtype=torch.float32)
        y_soft = torch.empty_like(logits_2d)
        out = torch.empty_like(logits_2d) if hard else y_soft

        grid = (n_rows, n_chunks)
        _chunked_stats_kernel[grid](
            logits_2d,
            gumbels_2d,
            chunk_max,
            chunk_arg,
            V,
            float(tau),
            N_CHUNKS=n_chunks,
            CHUNK_V=_CHUNK_V,
            num_warps=16,
        )
        _chunked_global_stats_kernel[(n_rows,)](
            chunk_max,
            chunk_arg,
            row_max,
            row_arg,
            N_CHUNKS=n_chunks,
            BLOCK_CHUNKS=block_chunks,
            num_warps=1,
        )
        _chunked_sum_kernel[grid](
            logits_2d,
            gumbels_2d,
            row_max,
            chunk_sum,
            V,
            float(tau),
            N_CHUNKS=n_chunks,
            CHUNK_V=_CHUNK_V,
            num_warps=16,
        )
        _chunked_output_kernel[grid](
            logits_2d,
            gumbels_2d,
            row_max,
            row_arg,
            chunk_sum,
            y_soft,
            out,
            V,
            float(tau),
            HARD=bool(hard),
            STORE_OUT=out is not y_soft,
            N_CHUNKS=n_chunks,
            CHUNK_V=_CHUNK_V,
            BLOCK_CHUNKS=block_chunks,
            num_warps=16,
        )

        ctx.save_for_backward(y_soft)
        ctx.tau = float(tau)
        ctx.logits_shape = tuple(logits.shape)
        ctx.logits_dtype = logits.dtype
        return out.view(*lead_shape, V)

    @staticmethod
    def backward(ctx, grad_output):
        (y_soft,) = ctx.saved_tensors
        grad_2d = grad_output.contiguous().view_as(y_soft)
        grad_logits = torch._softmax_backward_data(
            grad_2d,
            y_soft,
            -1,
            ctx.logits_dtype,
        )
        if ctx.tau != 1.0:
            grad_logits = grad_logits / ctx.tau
        return grad_logits.view(ctx.logits_shape).to(ctx.logits_dtype), None, None, None


def _hard_gumbel_softmax_nograd(
    logits: torch.Tensor,
    gumbels: Optional[torch.Tensor],
    seed: int,
) -> torch.Tensor:
    V = logits.shape[-1]
    lead_shape = logits.shape[:-1]
    logits_2d = logits.contiguous().view(-1, V)
    gumbels_2d = gumbels.contiguous().view(-1, V) if gumbels is not None else logits_2d
    block_v, num_warps = _launch_config(V)

    out = torch.empty_like(logits_2d)
    _gumbel_softmax_hard_nograd_kernel[(logits_2d.shape[0],)](
        logits_2d,
        gumbels_2d,
        out,
        int(seed),
        logits_2d.shape[0],
        V,
        HAS_GUMBELS=gumbels is not None,
        BLOCK_V=block_v,
        num_warps=num_warps,
    )
    return out.view(*lead_shape, V)


def _chunked_hard_gumbel_softmax_nograd(
    logits: torch.Tensor,
    gumbels: torch.Tensor,
) -> torch.Tensor:
    V = logits.shape[-1]
    lead_shape = logits.shape[:-1]
    logits_2d = logits.contiguous().view(-1, V)
    gumbels_2d = gumbels.contiguous().view(-1, V)
    n_rows = logits_2d.shape[0]
    n_chunks = _num_chunks(V)
    block_chunks = triton.next_power_of_2(n_chunks)

    chunk_max = torch.empty((n_rows, n_chunks), device=logits.device, dtype=torch.float32)
    chunk_arg = torch.empty((n_rows, n_chunks), device=logits.device, dtype=torch.int64)
    row_max = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    row_arg = torch.empty((n_rows,), device=logits.device, dtype=torch.int64)
    out = torch.empty_like(logits_2d)

    grid = (n_rows, n_chunks)
    _chunked_stats_kernel[grid](
        logits_2d,
        gumbels_2d,
        chunk_max,
        chunk_arg,
        V,
        1.0,
        N_CHUNKS=n_chunks,
        CHUNK_V=_CHUNK_V,
        num_warps=16,
    )
    _chunked_global_stats_kernel[(n_rows,)](
        chunk_max,
        chunk_arg,
        row_max,
        row_arg,
        N_CHUNKS=n_chunks,
        BLOCK_CHUNKS=block_chunks,
        num_warps=1,
    )
    _chunked_hard_output_kernel[grid](
        row_arg,
        out,
        V,
        CHUNK_V=_CHUNK_V,
        num_warps=16,
    )
    return out.view(*lead_shape, V)


class TritonGumbelSoftmaxOp:
    """Triton Gumbel-Softmax sampler with straight-through hard samples."""

    def __call__(
        self,
        logits: torch.Tensor,
        *,
        tau: float = 1.0,
        hard: bool = False,
        gumbels: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        return self.apply(logits, tau=tau, hard=hard, gumbels=gumbels, seed=seed)

    def apply(
        self,
        logits: torch.Tensor,
        *,
        tau: float = 1.0,
        hard: bool = False,
        gumbels: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        _validate_gumbel_softmax_inputs(logits, float(tau), gumbels)
        if logits.device.type not in ("cuda", "xpu", "hip"):
            raise RuntimeError(
                "TritonGumbelSoftmaxOp requires a GPU tensor (CUDA / ROCm / XPU), got "
                f"device '{logits.device}'."
            )
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (), device="cpu").item())
        if gumbels is None and _needs_precomputed_gumbels(logits):
            gumbels = _sample_gumbels_like(logits, int(seed))
        if triton.next_power_of_2(logits.shape[-1]) > _MAX_BLOCK_V:
            if gumbels is None:
                gumbels = _sample_gumbels_like(logits, int(seed))
            if hard and (not torch.is_grad_enabled() or not logits.requires_grad):
                return _chunked_hard_gumbel_softmax_nograd(logits, gumbels)
            return _ChunkedGumbelSoftmaxFunction.apply(logits, gumbels, float(tau), bool(hard))
        if hard and (not torch.is_grad_enabled() or not logits.requires_grad):
            return _hard_gumbel_softmax_nograd(logits, gumbels, int(seed))
        return _GumbelSoftmaxFunction.apply(logits, gumbels, float(tau), bool(hard), int(seed))
