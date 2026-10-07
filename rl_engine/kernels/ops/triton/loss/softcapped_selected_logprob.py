# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton forward/backward cores for Gemma's softcapped selected logprob.

Automatic forward selects a row loop or a two-kernel parallel implementation
using device, dtype, exact vocabulary width and bounded row-count ranges.
Both reduce vocabulary tiles of 1024 elements in the same order, in FP32 with four warps
and no FP contraction. Parallel forward writes tile sums to FP32 scratch,
then merges them in the same ascending order in a second kernel.
Explicit ROW_PIPELINED requests three loop stages with the ROW arithmetic.
An explicit ROW_ACCUMULATE experiment instead accumulates a 1024-element
FP32 vector across tiles and reduces it once. Its different rounding order
is not interchangeable with ROW/PARALLEL in the automatic selection table.
Automatic selection changes the launch schedule but preserves the arithmetic order. The
launchers prepare contiguous inputs; no full softcapped/probability buffer is
written to global memory. Forward saves one FP32 log_sum_exp per row so backward
can reuse it without repeating the reduction. Backward uses one program per
row/vocabulary tile, with disjoint gradient stores. The autograd wrapper saves the
inputs and these statistics. The registry exposes this as softcapped_selected_logprob.
"""

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, partial

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
    ROW_PIPELINED = "row_pipelined"
    PARALLEL = "parallel"
    ROW_ACCUMULATE = "row_accumulate"


# Final fallback for ANY supported shape not matched by a device/default map
# entry, including rows, vocabulary, or both above the measured ranges. ROW
# needs one forward launch and no partial-sum scratch. This is a conservative
# execution policy, not a claim that ROW is fastest outside the measured map.
# Changing this policy at runtime also requires clearing the selector's cache.
DEFAULT_FORWARD_STRATEGY = SoftcappedLogprobStrategy.ROW


# Freeze fields so equal metadata produces equal, hashable dictionary keys.
@dataclass(frozen=True)
class ForwardConfigKey:
    """A bounded row-count range for one device, dtype and exact vocabulary."""

    # "default" is the H100-derived policy shared by devices without an override.
    # A specific device uses (backend, GPU model), e.g. ("cuda", "NVIDIA H100 80GB HBM3")
    # or ("rocm", "AMD Instinct MI300X"). This identifies the model, not cuda:0;
    # equal model names on different backends do not share an override.
    device_key: str | tuple[str, str]

    # Storage dtype of logits: FP16, BF16 or FP32. Intermediate computation and
    # output remain FP32. Keep separate entries so future tuning can differ by dtype.
    dtype: torch.dtype

    # Inclusive lower and upper bounds on M in logits.shape == [M, V]. These
    # bounds select a launch schedule; they never pad, round or reshape the tensor.
    # Ranges for the same device/dtype/vocabulary must not overlap.
    min_rows: int
    max_rows: int

    # Exact V in logits.shape == [M, V], not the 1024-element kernel tile size.
    # Do not round vocabulary widths: each width needs its own measurements.
    vocab_size: int


# H100 80GB HBM3, PyTorch 2.13.0+cu130, Triton 3.7.1.
# Evidence: 837 eager/reuse cases and 27 CUDA Graph checks, each with 12 balanced rounds.
# A separate auto-dispatch check covered 39 eager and 12 CUDA Graph cases;
# accuracy and output/LSE/gradient bitwise checks passed for all measured cases.
# Bounds follow regions with stable >=1.05x PARALLEL wins. Noisy points
# retain the range when their medians agree. Small/tied
# cases and metadata outside these ranges keep the ROW fallback.
# 2049 includes the tested 2048+1 boundary; 1026 is the last adjacent-width
# probe. Such endpoints are evidence limits, NOT measured hardware crossovers.
# Intermediate rows are interpolated; vocabulary widths are never rounded.
# Selection matches every conclusive ROW/PARALLEL winner in these reports.
# Unmeasured rows and other execution environments still need separate validation.
# ROW_PIPELINED has no stable >=1.05x win and remains explicit-only.
# Other devices intentionally inherit this default; their speed is unverified.
# Device-specific ranges may override it after measurements on that hardware.
# fmt: off
FORWARD_STRATEGY_CONFIGS: dict[ForwardConfigKey, SoftcappedLogprobStrategy] = {
    # FLOAT16: exact measured widths; row ranges include both endpoints.
    ForwardConfigKey("default", torch.float16, 1, 16, 32767): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 2049, 32768): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 1024, 32769): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 2048, 49152): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 1024, 65535): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 2048, 65536): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 1024, 65537): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 4096, 128256): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 1024, 131071): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 4096, 131072): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 1024, 131073): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 4096, 151936): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 4096, 256000): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 1024, 262142): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 1026, 262143): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 4096, 262144): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 1026, 262145): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 1024, 262146): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 1024, 524287): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 2048, 524288): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 1024, 524289): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float16, 1, 1024, 1048575): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.float16, 1, 1024, 1048576): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.float16, 1, 16, 1048577): SoftcappedLogprobStrategy.PARALLEL,
    # BFLOAT16: exact measured widths; row ranges include both endpoints.
    ForwardConfigKey("default", torch.bfloat16, 1, 16, 32767): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 1, 2049, 32768): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 1, 1024, 32769): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 1, 2048, 49152): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 1, 1024, 65535): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 1, 2048, 65536): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 1, 1024, 65537): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.bfloat16, 1, 4096, 128256): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 1024, 131071): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 2048, 131072): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 1024, 131073): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 4096, 151936): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 4096, 256000): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 1024, 262142): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 1026, 262143): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 4096, 262144): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 1026, 262145): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 1024, 262146): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 1024, 524287): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 2048, 524288): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 1024, 524289): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 1024, 1048575): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 1024, 1048576): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.bfloat16, 1, 16, 1048577): SoftcappedLogprobStrategy.PARALLEL,
    # FLOAT32: exact measured widths; row ranges include both endpoints.
    ForwardConfigKey("default", torch.float32, 1, 16, 32767): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 2049, 32768): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 1024, 32769): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 2048, 49152): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 1024, 65535): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 2048, 65536): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 1024, 65537): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 2048, 128256): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 1024, 131071): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 2048, 131072): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 1024, 131073): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 2048, 151936): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 2048, 256000): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 1024, 262142): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 1026, 262143): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 2049, 262144): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 1026, 262145): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 1024, 262146): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 1024, 524287): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 2048, 524288): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 1024, 524289): SoftcappedLogprobStrategy.PARALLEL,
    ForwardConfigKey("default", torch.float32, 1, 1024, 1048575): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.float32, 1, 1024, 1048576): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey("default", torch.float32, 1, 16, 1048577): SoftcappedLogprobStrategy.PARALLEL,
}
# fmt: on


@lru_cache(maxsize=1024)
def select_softcapped_logprob_strategy(
    device_key: tuple[str, str], dtype: torch.dtype, n_rows: int, vocab_size: int
) -> SoftcappedLogprobStrategy:
    """Match device ranges, then default ranges, then DEFAULT_FORWARD_STRATEGY.

    Cache by exact device, dtype, row count and vocabulary size. Repeated
    metadata skips matching after warmup; unseen or evicted keys scan the map.
    Clear this cache after changing FORWARD_STRATEGY_CONFIGS at runtime.
    Reject ambiguous matches instead of allowing insertion order to decide.
    An explicit device ROW rule overrides a default PARALLEL rule. Only metadata
    is inspected, never tensor values or gradient mode.
    """
    for scope in (device_key, "default"):
        matched = None
        for key, strategy in FORWARD_STRATEGY_CONFIGS.items():
            if (
                key.device_key == scope
                and key.dtype == dtype
                and key.vocab_size == vocab_size
                and key.min_rows <= n_rows <= key.max_rows
            ):
                if matched is not None:
                    raise ValueError("overlapping forward strategy ranges")
                matched = strategy
        if matched is not None:
            return matched
    return DEFAULT_FORWARD_STRATEGY


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
    ACCUMULATE_BEFORE_REDUCE: tl.constexpr = False,
    PIPELINE_STAGES: tl.constexpr = None,
):
    # One program handles one row; widen before multiplying the row offset.
    row = tl.program_id(0).to(tl.int64)
    row_start = row * vocab_size
    cols = tl.arange(0, BLOCK_V)

    # 1. The experiment keeps one FP32 accumulator per tile position.
    # This constexpr chooses separate compiled kernels; there is no runtime
    # strategy branch inside the vocabulary loop.
    if ACCUMULATE_BEFORE_REDUCE:
        partial_exp = tl.zeros((BLOCK_V,), dtype=tl.float32)
    else:
        sum_exp = tl.zeros((), dtype=tl.float32)
    # None preserves the ROW baseline; ROW_PIPELINED requests three stages.
    # This compiler hint leaves tile arithmetic and reduction order unchanged.
    for start in tl.range(0, vocab_size, BLOCK_V, num_stages=PIPELINE_STAGES):
        vocab_offsets = start + cols
        mask = vocab_offsets < vocab_size
        logits = tl.load(logits_ptr + row_start + vocab_offsets, mask=mask, other=0.0).to(tl.float32)  # noqa: E501 # fmt: skip

        scaled_logits = tl.div_rn(logits, 30.0)
        softcapped = 30.0 * libdevice.tanh(scaled_logits)
        exp_softcapped = tl.exp(softcapped)
        # Padding must contribute zero, not exp(softcap(0)) = 1.
        exp_softcapped = tl.where(mask, exp_softcapped, 0.0)
        if ACCUMULATE_BEFORE_REDUCE:
            partial_exp = partial_exp + exp_softcapped
        else:
            tile_sum_exp = tl.sum(exp_softcapped, axis=0)
            sum_exp = sum_exp + tile_sum_exp

    if ACCUMULATE_BEFORE_REDUCE:
        sum_exp = tl.sum(partial_exp, axis=0)

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
    MERGE_UNROLL: tl.constexpr = 1,
):
    row = tl.program_id(0).to(tl.int64)
    row_start = row * vocab_size
    partial_row_start = row * n_tiles

    # Preserve the original FP32 left-to-right accumulation, including the
    # initial zero. A tree reduction over partial sums would change rounding.
    sum_exp = tl.zeros((), dtype=tl.float32)
    tl.static_assert(MERGE_UNROLL == 1 or MERGE_UNROLL == 4)
    if MERGE_UNROLL == 4:
        # Expose four independent loads before the dependent additions. This
        # changes loop grouping, not the FP32 addition order. Whether the loads
        # overlap, and whether this helps, depend on compiler output/hardware.
        for tile in range(0, (n_tiles // 4) * 4, 4):
            partial_ptr = partial_sum_exp_ptr + partial_row_start + tile
            s0 = tl.load(partial_ptr)
            s1 = tl.load(partial_ptr + 1)
            s2 = tl.load(partial_ptr + 2)
            s3 = tl.load(partial_ptr + 3)
            sum_exp = sum_exp + s0
            sum_exp = sum_exp + s1
            sum_exp = sum_exp + s2
            sum_exp = sum_exp + s3
        # Handle 0..3 remaining tiles without out-of-bounds reads or extra adds.
        for tile in tl.static_range((n_tiles // 4) * 4, n_tiles):
            tile_sum_exp = tl.load(partial_sum_exp_ptr + partial_row_start + tile)
            sum_exp = sum_exp + tile_sum_exp
    else:
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
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    accumulate_before_reduce: bool = False,
    pipeline_stages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 selected_logprob and log_sum_exp, both [M], without autograd.

    Valid token IDs in [0, V) are a caller precondition. The kernel masks
    invalid IDs to NaN to avoid out-of-bounds reads, without a host/GPU sync.
    pipeline_stages sets the loop hint, not the launch num_stages. ROW_PIPELINED
    requests three stages; None retains the original unhinted ROW baseline.
    """
    _validate_inputs(logits, token_ids)

    n_rows, vocab_size = logits.shape
    logits_c = logits.contiguous()
    token_ids_c = token_ids.contiguous()
    output = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    log_sum_exp = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    if n_rows == 0:
        return output, log_sum_exp

    # ROCm tensors also use torch.cuda; do not enter CUDA for other backends.
    with torch.cuda.device(logits.device) if logits.device.type == "cuda" else nullcontext():
        _softcapped_selected_logprob_fwd_kernel[(n_rows,)](
            logits_c,
            token_ids_c,
            output,
            log_sum_exp,
            vocab_size,
            BLOCK_V=_BLOCK_V,
            ACCUMULATE_BEFORE_REDUCE=accumulate_before_reduce,
            PIPELINE_STAGES=pipeline_stages,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output, log_sum_exp


def _launch_softcapped_selected_logprob_fwd_parallel(
    logits: torch.Tensor, token_ids: torch.Tensor, *, merge_unroll: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-kernel forward with FP32 [M, ceil(V / 1024)] scratch.

    Return the same FP32 [M] output/statistics pair as the row-loop version.
    Allocation, partial reduction and ordered merge all belong to this call.
    merge_unroll=4 is a benchmark-only experiment; normal dispatch uses 1.
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
    with torch.cuda.device(logits.device) if logits.device.type == "cuda" else nullcontext():
        _softcapped_selected_logprob_fwd_partial_kernel[(n_rows, n_tiles)](
            logits_c,
            partial_sum_exp,
            vocab_size,
            n_tiles,
            BLOCK_V=_BLOCK_V,
            num_warps=4,
            enable_fp_fusion=False,
        )
        # Both kernels use the input device's current stream, so the merge reads
        # completed partial sums without an explicit host synchronization.
        _softcapped_selected_logprob_fwd_merge_kernel[(n_rows,)](
            logits_c,
            token_ids_c,
            partial_sum_exp,
            output,
            log_sum_exp,
            vocab_size,
            n_tiles,
            MERGE_UNROLL=merge_unroll,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output, log_sum_exp


def _launch_softcapped_selected_logprob_bwd(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    grad_selected_logprob: torch.Tensor,
    log_sum_exp: torch.Tensor,
    *,
    block_v: int = _BLOCK_V,
    num_warps: int = 4,
) -> torch.Tensor:
    """Return input-dtype grad_logits; grad_selected_logprob has shape [M].

    log_sum_exp is the FP32 [M] statistic returned by forward for these inputs.
    The autograd wrapper supplies it together with the inputs and upstream gradient.
    As in forward, invalid token IDs produce NaN; valid IDs are a precondition.
    Private launch parameters let the benchmark sweep backward independently;
    autograd continues to use the default configuration.
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

    grid = (n_rows, triton.cdiv(vocab_size, block_v))
    with torch.cuda.device(logits.device) if logits.device.type == "cuda" else nullcontext():
        _softcapped_selected_logprob_bwd_kernel[grid](
            logits_c,
            token_ids_c,
            grad_selected_logprob_c,
            log_sum_exp_c,
            grad_logits,
            vocab_size,
            BLOCK_V=block_v,
            num_warps=num_warps,
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
    # H100 V=262144: stages=3 improved ROW at measured M<=512. Keep this
    # explicit: PARALLEL won the main comparison, and unmeasured shapes/devices
    # must not silently inherit a pipelined ROW fallback.
    SoftcappedLogprobStrategy.ROW_PIPELINED: partial(
        _launch_softcapped_selected_logprob_fwd, pipeline_stages=3
    ),
    SoftcappedLogprobStrategy.PARALLEL: _launch_softcapped_selected_logprob_fwd_parallel,
    SoftcappedLogprobStrategy.ROW_ACCUMULATE: partial(
        _launch_softcapped_selected_logprob_fwd, accumulate_before_reduce=True
    ),
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
    Explicit SoftcappedLogprobStrategy values bypass selection. All strategies
    reuse the same backward. ROW, ROW_PIPELINED and PARALLEL preserve arithmetic order;
    ROW_ACCUMULATE is an explicit experiment with a different reduction order
    and is never chosen by the automatic policy.
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
