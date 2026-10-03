# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton forward/backward cores for Gemma's softcapped selected logprob.

Automatic forward selects a row loop or a two-kernel parallel implementation
using device, dtype, vocabulary width and row count. Both reduce vocabulary
tiles of 1024 elements in the same order, accumulating in FP32 with four warps
and no FP contraction. Parallel forward writes tile sums to FP32 scratch,
then merges them in the same ascending order in a second kernel.
Selection changes the launch schedule but preserves the arithmetic order. The
launchers prepare contiguous inputs; no full softcapped/probability buffer is
written to global memory. Forward saves one FP32 log_sum_exp per row so backward
can reuse it without repeating the reduction. Backward uses one program per
row/vocabulary tile, with disjoint gradient stores. The autograd wrapper saves the
inputs and these statistics. The registry exposes this as softcapped_selected_logprob.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Keep the accepted device types aligned with final_logit_softcap.
_SUPPORTED_DEVICES = ("cuda", "hip", "xpu", "musa")

_BLOCK_V = 1024


class SoftcappedLogprobStrategy(str, Enum):
    ROW = "row"
    PARALLEL = "parallel"


# Freeze fields so equal metadata produces equal, hashable dictionary keys.
@dataclass(frozen=True)
class ForwardConfigKey:
    """Exact input metadata used to select a forward implementation."""

    # "default" matches when no device-specific entry exists for this dtype/shape.
    # A specific device uses (backend, GPU model), e.g. ("cuda", "NVIDIA A40")
    # or ("rocm", "AMD Instinct MI300X"). This identifies the model, not its index
    # such as cuda:0; different backends have distinct configuration keys.
    device_key: str | tuple[str, str]

    # Storage dtype of the input logits: torch.float16, torch.bfloat16 or
    # torch.float32. This is separate from the FP32 intermediate computation
    # and output dtype; input dtype can change which implementation is faster.
    dtype: torch.dtype

    # M in logits.shape == [M, V]: the number of rows (token positions) processed
    # by this call. Matching uses this exact count, not a range. Row count affects
    # how much parallel work the row-loop implementation already exposes.
    n_rows: int

    # V in logits.shape == [M, V]: the number of vocabulary scores in each row.
    # This is the full vocabulary width, not the BLOCK_V tile size. Matching uses
    # this exact width, including non-multiples of BLOCK_V; no threshold is inferred.
    vocab_size: int


# A40, PyTorch 2.13.0+cu130, Triton 3.7.1, benchmark commit 18ec2ae.
# Two independent runs on 2026-10-03, four rounds per run, 200 samples/round.
# These 47 exact combinations passed the screen across all eight rounds:
# speedup >= 1.05x in every round, std/median and round-median spread <= 10%.
# Do not interpolate between vocabulary widths or row counts.
FORWARD_STRATEGY_CONFIGS: dict[ForwardConfigKey, SoftcappedLogprobStrategy] = {
    # Initially all devices use this A40-derived default policy.
    # Add device-specific entries only when measurements justify an override.
    # Performance on other devices remains unverified.
    ForwardConfigKey("default", torch.float16, 4, 36864): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 4, 65536): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 4, 262144): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 16, 36864): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 16, 49152): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 16, 65536): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 16, 262144): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 64, 36864): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 64, 49152): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 64, 65536): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 64, 262144): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 256, 28672): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 256, 32768): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 256, 32769): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 256, 36864): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 256, 49152): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 256, 65536): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 256, 262144): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 1, 36864): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 1, 49152): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 1, 65536): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 1, 262144): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 4, 36864): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 4, 262144): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 16, 36864): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 16, 49152): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 16, 65536): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 16, 262144): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 64, 49152): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 64, 65536): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 64, 262144): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 256, 28672): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 256, 32768): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 256, 32769): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 256, 36864): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 32769): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 4, 36864): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 4, 65536): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 4, 262144): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 16, 65536): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 16, 262144): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 64, 32768): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 64, 32769): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 64, 49152): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 64, 65536): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 64, 262144): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 256, 49152): SoftcappedLogprobStrategy.PARALLEL,
}


def select_softcapped_logprob_strategy(
    device_key: tuple[str, str], dtype: torch.dtype, n_rows: int, vocab_size: int
) -> SoftcappedLogprobStrategy:
    """Look up exact metadata: device entry, default entry, then ROW."""
    key = ForwardConfigKey(device_key, dtype, n_rows, vocab_size)
    strategy = FORWARD_STRATEGY_CONFIGS.get(key)
    if strategy is not None:
        return strategy
    return FORWARD_STRATEGY_CONFIGS.get(
        ForwardConfigKey("default", dtype, n_rows, vocab_size), SoftcappedLogprobStrategy.ROW
    )


@lru_cache(maxsize=None)
def _cuda_device_key(backend: str, index: int) -> tuple[str, str]:
    return backend, torch.cuda.get_device_name(index)


def softcapped_logprob_device_key(device: torch.device) -> tuple[str, str]:
    """Use the input tensor's GPU, including ROCm's torch.cuda namespace.

    Resolve an unspecified index before caching so changing the current GPU
    cannot accidentally reuse another GPU's configuration.
    """
    if device.type == "cuda":
        backend = "rocm" if torch.version.hip is not None else "cuda"
        index = device.index if device.index is not None else torch.cuda.current_device()
        return _cuda_device_key(backend, index)
    # No model-specific tuning is available for the other accepted backends.
    return device.type, ""


@triton.jit
def _softcapped_selected_logprob_fwd_kernel(
    logits_ptr,
    token_ids_ptr,
    selected_logprob_ptr,
    log_sum_exp_ptr,
    vocab_size: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # One program handles one row; widen before multiplying the row offset.
    row = tl.program_id(0).to(tl.int64)
    row_start = row * vocab_size
    cols = tl.arange(0, BLOCK_V)

    # 1. Accumulate sum(exp(softcap(logits))) in a fixed vocabulary order.
    sum_exp = tl.zeros((), dtype=tl.float32)
    for start in range(0, vocab_size, BLOCK_V):
        vocab_offsets = start + cols
        mask = vocab_offsets < vocab_size
        logits = tl.load(logits_ptr + row_start + vocab_offsets, mask=mask, other=0.0).to(tl.float32)  # noqa: E501 # fmt: skip

        scaled_logits = tl.div_rn(logits, 30.0)
        softcapped = 30.0 * libdevice.tanh(scaled_logits)
        exp_softcapped = tl.exp(softcapped)
        # Padding must contribute zero, not exp(softcap(0)) = 1.
        exp_softcapped = tl.where(mask, exp_softcapped, 0.0)
        tile_sum_exp = tl.sum(exp_softcapped, axis=0)
        sum_exp = sum_exp + tile_sum_exp

    # With softcap fixed at 30, no max-shift is needed to avoid overflow for
    # the Gemma vocabulary. This assumption must be revisited for other caps.
    log_sum_exp = tl.log(sum_exp)

    # 2. Load only the selected score and apply the same softcap arithmetic.
    token_id = tl.load(token_ids_ptr + row).to(tl.int64)
    selected_logit = tl.load(
        logits_ptr + row_start + token_id,
        mask=(token_id >= 0) & (token_id < vocab_size),
        other=float("nan"),
    ).to(tl.float32)
    selected_softcapped = 30.0 * libdevice.tanh(tl.div_rn(selected_logit, 30.0))

    # 3. Store one FP32 log-probability for this row.
    selected_logprob = selected_softcapped - log_sum_exp
    tl.store(selected_logprob_ptr + row, selected_logprob)
    tl.store(log_sum_exp_ptr + row, log_sum_exp)


@triton.jit
def _softcapped_selected_logprob_fwd_partial_kernel(
    logits_ptr,
    partial_sum_exp_ptr,
    vocab_size: tl.constexpr,
    n_tiles: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    row_start = row * vocab_size
    cols = tl.arange(0, BLOCK_V)
    vocab_offsets = tile * BLOCK_V + cols
    mask = vocab_offsets < vocab_size

    # Keep the original 1024-element tile arithmetic and reduction order.
    logits = tl.load(logits_ptr + row_start + vocab_offsets, mask=mask, other=0.0).to(tl.float32)  # noqa: E501 # fmt: skip
    scaled_logits = tl.div_rn(logits, 30.0)
    softcapped = 30.0 * libdevice.tanh(scaled_logits)
    exp_softcapped = tl.exp(softcapped)
    exp_softcapped = tl.where(mask, exp_softcapped, 0.0)
    tile_sum_exp = tl.sum(exp_softcapped, axis=0)

    # Each program owns exactly one entry in the [M, n_tiles] scratch array.
    tl.store(partial_sum_exp_ptr + row * n_tiles + tile, tile_sum_exp)


@triton.jit
def _softcapped_selected_logprob_fwd_merge_kernel(
    logits_ptr,
    token_ids_ptr,
    partial_sum_exp_ptr,
    selected_logprob_ptr,
    log_sum_exp_ptr,
    vocab_size: tl.constexpr,
    n_tiles: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_start = row * vocab_size
    partial_row_start = row * n_tiles

    # Preserve the original FP32 left-to-right accumulation, including the
    # initial zero. A tree reduction over partial sums would change rounding.
    sum_exp = tl.zeros((), dtype=tl.float32)
    for tile in range(0, n_tiles):
        tile_sum_exp = tl.load(partial_sum_exp_ptr + partial_row_start + tile)
        sum_exp = sum_exp + tile_sum_exp
    log_sum_exp = tl.log(sum_exp)

    token_id = tl.load(token_ids_ptr + row).to(tl.int64)
    selected_logit = tl.load(
        logits_ptr + row_start + token_id,
        mask=(token_id >= 0) & (token_id < vocab_size),
        other=float("nan"),
    ).to(tl.float32)
    selected_softcapped = 30.0 * libdevice.tanh(tl.div_rn(selected_logit, 30.0))
    selected_logprob = selected_softcapped - log_sum_exp
    tl.store(selected_logprob_ptr + row, selected_logprob)
    tl.store(log_sum_exp_ptr + row, log_sum_exp)


@triton.jit
def _softcapped_selected_logprob_bwd_kernel(
    logits_ptr,
    token_ids_ptr,
    grad_selected_logprob_ptr,
    log_sum_exp_ptr,
    grad_logits_ptr,
    vocab_size: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    row_start = row * vocab_size
    cols = tl.arange(0, BLOCK_V)
    vocab_offsets = tile * BLOCK_V + cols
    mask = vocab_offsets < vocab_size

    # 1. Reuse the FP32 log_sum_exp saved by forward for this row.
    log_sum_exp = tl.load(log_sum_exp_ptr + row)
    token_id = tl.load(token_ids_ptr + row).to(tl.int64)
    grad_selected_logprob = tl.load(grad_selected_logprob_ptr + row).to(tl.float32)

    # 2. Each program computes one tile using the shared, read-only row statistics.
    logits = tl.load(logits_ptr + row_start + vocab_offsets, mask=mask, other=0.0).to(tl.float32)  # noqa: E501 # fmt: skip

    scaled_logits = tl.div_rn(logits, 30.0)
    tanh_scaled_logits = libdevice.tanh(scaled_logits)
    softcapped = 30.0 * tanh_scaled_logits
    probability = tl.exp(softcapped - log_sum_exp)

    # 3. Chain through selected logprob: upstream * (selected - p).
    # 1.0 at the selected token position; 0.0 at all other positions.
    is_selected = tl.where(vocab_offsets == token_id, 1.0, 0.0)
    grad_softcapped = grad_selected_logprob * (is_selected - probability)

    # 4. Chain through softcap; each gradient element has exactly one writer.
    softcap_derivative = 1.0 - tanh_scaled_logits * tanh_scaled_logits
    grad_logits = grad_softcapped * softcap_derivative
    grad_logits = tl.where((token_id >= 0) & (token_id < vocab_size), grad_logits, float("nan"))
    tl.store(
        grad_logits_ptr + row_start + vocab_offsets,
        grad_logits.to(grad_logits_ptr.dtype.element_ty),
        mask=mask,
    )


def _validate_inputs(logits: torch.Tensor, token_ids: torch.Tensor) -> None:
    if logits.device.type not in _SUPPORTED_DEVICES:
        raise RuntimeError(
            "The Triton core requires a GPU tensor; "
            f"supported device types are {_SUPPORTED_DEVICES}, got '{logits.device.type}'."
        )
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("logits must have shape [M, V] with V > 0.")
    if logits.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"logits must have dtype {_SUPPORTED_DTYPES}, got {logits.dtype}.")
    if token_ids.shape != logits.shape[:1]:
        raise ValueError("token_ids must have shape [M], matching the logits rows.")
    if token_ids.dtype != torch.int64 or token_ids.device != logits.device:
        raise TypeError("token_ids must have dtype int64 and be on the logits device.")


def _validate_backward_inputs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    grad_selected_logprob: torch.Tensor,
    log_sum_exp: torch.Tensor,
) -> None:
    _validate_inputs(logits, token_ids)
    if grad_selected_logprob.shape != logits.shape[:1]:
        raise ValueError("grad_selected_logprob must have shape [M].")
    if grad_selected_logprob.device != logits.device:
        raise ValueError("grad_selected_logprob must be on the logits device.")
    if grad_selected_logprob.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"grad_selected_logprob must have dtype {_SUPPORTED_DTYPES}, "
            f"got {grad_selected_logprob.dtype}."
        )
    if log_sum_exp.shape != logits.shape[:1]:
        raise ValueError("log_sum_exp must have shape [M].")
    if log_sum_exp.device != logits.device:
        raise ValueError("log_sum_exp must be on the logits device.")
    if log_sum_exp.dtype != torch.float32:
        raise TypeError("log_sum_exp must have dtype float32.")


def _launch_softcapped_selected_logprob_fwd(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 selected_logprob and log_sum_exp, both [M], without autograd.

    Valid token IDs in [0, V) are a caller precondition. The kernel masks
    invalid IDs to NaN to avoid out-of-bounds reads, without a host/GPU sync.
    """
    _validate_inputs(logits, token_ids)

    n_rows, vocab_size = logits.shape
    logits_c = logits.contiguous()
    token_ids_c = token_ids.contiguous()
    output = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    log_sum_exp = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    if n_rows == 0:
        return output, log_sum_exp

    _softcapped_selected_logprob_fwd_kernel[(n_rows,)](
        logits_c,
        token_ids_c,
        output,
        log_sum_exp,
        vocab_size,
        BLOCK_V=_BLOCK_V,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, log_sum_exp


def _launch_softcapped_selected_logprob_fwd_parallel(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-kernel forward with FP32 [M, ceil(V / 1024)] scratch.

    Return the same FP32 [M] output/statistics pair as the row-loop version.
    Allocation, partial reduction and ordered merge all belong to this call.
    """
    _validate_inputs(logits, token_ids)

    n_rows, vocab_size = logits.shape
    logits_c = logits.contiguous()
    token_ids_c = token_ids.contiguous()
    output = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    log_sum_exp = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    if n_rows == 0:
        return output, log_sum_exp

    n_tiles = triton.cdiv(vocab_size, _BLOCK_V)
    partial_sum_exp = torch.empty((n_rows, n_tiles), device=logits.device, dtype=torch.float32)
    _softcapped_selected_logprob_fwd_partial_kernel[(n_rows, n_tiles)](
        logits_c,
        partial_sum_exp,
        vocab_size,
        n_tiles,
        BLOCK_V=_BLOCK_V,
        num_warps=4,
        enable_fp_fusion=False,
    )
    # Both kernels launch on the current stream, so the merge reads completed
    # partial sums without an explicit host synchronization.
    _softcapped_selected_logprob_fwd_merge_kernel[(n_rows,)](
        logits_c,
        token_ids_c,
        partial_sum_exp,
        output,
        log_sum_exp,
        vocab_size,
        n_tiles,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, log_sum_exp


def _launch_softcapped_selected_logprob_bwd(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    grad_selected_logprob: torch.Tensor,
    log_sum_exp: torch.Tensor,
) -> torch.Tensor:
    """Return input-dtype grad_logits; grad_selected_logprob has shape [M].

    log_sum_exp is the FP32 [M] statistic returned by forward for these inputs.
    The autograd wrapper supplies it together with the inputs and upstream gradient.
    As in forward, invalid token IDs produce NaN; valid IDs are a precondition.
    """
    _validate_backward_inputs(logits, token_ids, grad_selected_logprob, log_sum_exp)

    n_rows, vocab_size = logits.shape
    logits_c = logits.contiguous()
    token_ids_c = token_ids.contiguous()
    grad_selected_logprob_c = grad_selected_logprob.contiguous()
    log_sum_exp_c = log_sum_exp.contiguous()
    grad_logits = torch.empty_like(logits_c)
    if n_rows == 0:
        return grad_logits

    grid = (n_rows, triton.cdiv(vocab_size, _BLOCK_V))
    _softcapped_selected_logprob_bwd_kernel[grid](
        logits_c,
        token_ids_c,
        grad_selected_logprob_c,
        log_sum_exp_c,
        grad_logits,
        vocab_size,
        BLOCK_V=_BLOCK_V,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return grad_logits


# Each forward strategy accepts (logits, token_ids) and returns (output, log_sum_exp).
# Register new implementations here; the autograd wrapper only dispatches by key.
_FORWARD_LAUNCHERS: dict[
    SoftcappedLogprobStrategy,
    Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
] = {
    SoftcappedLogprobStrategy.ROW: _launch_softcapped_selected_logprob_fwd,
    SoftcappedLogprobStrategy.PARALLEL: _launch_softcapped_selected_logprob_fwd_parallel,
}


class _SoftcappedSelectedLogprobTritonFunction(torch.autograd.Function):
    """Connect the Triton forward/backward cores to PyTorch autograd."""

    @staticmethod
    def forward(
        ctx, logits: torch.Tensor, token_ids: torch.Tensor, strategy: SoftcappedLogprobStrategy
    ) -> torch.Tensor:
        logits_c = logits.contiguous()
        token_ids_c = token_ids.contiguous()
        launch_forward = _FORWARD_LAUNCHERS[strategy]
        selected_logprob, log_sum_exp = launch_forward(logits_c, token_ids_c)
        ctx.save_for_backward(logits_c, token_ids_c, log_sum_exp)
        return selected_logprob

    @staticmethod
    def backward(ctx, grad_selected_logprob: torch.Tensor):
        logits, token_ids, log_sum_exp = ctx.saved_tensors
        grad_logits = None
        if ctx.needs_input_grad[0]:
            grad_logits = _launch_softcapped_selected_logprob_bwd(
                logits, token_ids, grad_selected_logprob, log_sum_exp
            )
        # Integer token IDs and the forward implementation selector have no gradient.
        return grad_logits, None, None


class TritonSoftcappedSelectedLogprobOp:
    """Selected logprob after FP32 softcap, with first-order autograd.

    logits [M, V] must use a supported floating dtype on an accepted GPU; token_ids
    [M] must be int64 on the same device with values in [0, V). Output is FP32
    [M], and gradients use the logits dtype. Softcap is fixed at 30.0.
    forward_impl=None selects by device, dtype, vocabulary width and row count.
    Explicit SoftcappedLogprobStrategy values bypass
    selection. Both implementations reuse the same backward and arithmetic order.
    """

    op_class = "logprob"

    def __init__(self, *, forward_impl: SoftcappedLogprobStrategy | None = None):
        self.forward_impl = forward_impl

    def __call__(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(logits, token_ids)

    def forward(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        # Validate before reading [M, V] or querying hardware. No GPU data is
        # read to choose a strategy, and training/inference use the same rule.
        _validate_inputs(logits, token_ids)
        strategy = self.forward_impl
        if strategy is None:
            n_rows, vocab_size = logits.shape
            strategy = select_softcapped_logprob_strategy(
                softcapped_logprob_device_key(logits.device), logits.dtype, n_rows, vocab_size
            )
        return _SoftcappedSelectedLogprobTritonFunction.apply(logits, token_ids, strategy)
