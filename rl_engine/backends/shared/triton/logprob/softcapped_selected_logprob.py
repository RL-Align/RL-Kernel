# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Shared Triton forward/backward cores for softcapped selected logprob.

Automatic forward selects a row loop or a four-kernel parallel implementation
using device, dtype, row-count ranges and vocabulary tile-count ranges.
Both reduce vocabulary tiles of 1024 elements in the same order, in FP32 with
four warps and no FP contraction. Forward uses libdevice.exp for FP32 exponentials;
the fast tl.exp approximation can exceed the output tolerance at larger caps.
Every strategy first finds the same whole-row maximum, then subtracts it before
exponentiation. Parallel forward finds tile maxima and merges them, then reuses
the tile scratch for shifted sums and merges those in ascending tile order.
Explicit ROW_PIPELINED requests three loop stages with the ROW arithmetic.
The output is (selected_score - row_max) - log_shifted_sum, entirely in FP32.
Automatic selection changes the launch schedule but preserves the arithmetic
order. The launchers prepare contiguous inputs; no full softcapped/probability
buffer is written to global memory. Forward saves FP32 row_max and
log_sum_exp_shifted per row so backward can reuse them without repeating the
reduction. Backward uses one program per
row/vocabulary tile, with disjoint gradient stores. The autograd wrapper saves the
inputs and these statistics. The registry exposes this as softcapped_selected_logprob.
"""

import math
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, partial
from typing import TypeAlias

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
"""Logits storage dtypes; intermediate calculations and forward outputs use FP32."""

_SUPPORTED_DEVICES = ("cuda", "hip", "xpu", "musa")
"""Accepted device types; ROCm PyTorch also uses the CUDA namespace."""

_BLOCK_V = 1024
"""Vocabulary elements per kernel tile, also used to compute policy tile counts."""


class SoftcappedLogprobStrategy(str, Enum):
    """Choose the forward execution schedule; all strategies share one backward."""

    ROW = "row"
    """One program per row: find its maximum, then accumulate shifted tile sums in order."""

    ROW_PIPELINED = "row_pipelined"
    """Use ROW arithmetic with three loop pipeline stages; explicit selection only."""

    PARALLEL = "parallel"
    """Find a shared row maximum in two stages, then shifted sums in two more stages."""


DEFAULT_FORWARD_STRATEGY = SoftcappedLogprobStrategy.ROW
"""Fallback when no device-specific or default range matches.

ROW needs one forward launch and no partial-sum scratch. This is a conservative
execution policy, not a claim that ROW is fastest outside the configured ranges.
Clear the selector cache after changing this fallback at runtime.
"""


@dataclass(frozen=True)
class SoftcappedLogprobDefaultDevice:
    """Default variant: a shared policy with no device-specific attributes."""


@dataclass(frozen=True)
class SoftcappedLogprobDevice:
    """Concrete device variant, identified by backend and GPU model.

    This describes a hardware model, not a device ordinal such as cuda:0.
    Equal models on the same backend share rules; different backends do not.
    Frozen fields make the object usable as a dictionary and selector-cache key.
    """

    backend: str
    """Backend identity, e.g. 'cuda' or 'rocm'; ROCm is distinct despite torch.cuda."""

    gpu_name: str
    """GPU model returned by the backend; empty for backends without a model lookup."""


SoftcappedLogprobDeviceKey: TypeAlias = SoftcappedLogprobDefaultDevice | SoftcappedLogprobDevice
"""Device variants: Default carries no data; a concrete Device carries backend and model."""

_DEFAULT_DEVICE = SoftcappedLogprobDefaultDevice()
"""Shared policy scope, consulted after rules for the input tensor's concrete device."""


@dataclass(frozen=True)
class ForwardConfigKey:
    """A rectangular range of row counts and 1024-element vocabulary tile counts.

    Bounds are inclusive. max_tiles=None leaves the vocabulary range open-ended.
    Tile counts describe work per row; the tensor and kernel use the original V.
    Rules for the same device and dtype must not overlap in both dimensions.
    Frozen fields make this configuration a hashable, immutable map key.
    """

    device: SoftcappedLogprobDeviceKey
    """DefaultDevice shares a policy; Device requires an exact backend and model match."""

    dtype: torch.dtype
    """Logits storage dtype: FP16, BF16 or FP32; arithmetic and output remain FP32."""

    min_rows: int
    """Inclusive lower bound on M in logits.shape == [M, V]; no input padding is performed."""

    max_rows: int
    """Inclusive upper bound on M; larger batches continue to other rules or the fallback."""

    min_tiles: int
    """Inclusive lower bound on ceil(V / _BLOCK_V), computed from integer shape metadata."""

    max_tiles: int | None
    """Inclusive upper tile-count bound; None includes larger, potentially unmeasured V."""


# fmt: off
FORWARD_STRATEGY_CONFIGS: dict[ForwardConfigKey, SoftcappedLogprobStrategy] = {
    ForwardConfigKey(device=_DEFAULT_DEVICE, dtype=torch.float16, min_rows=1, max_rows=4096, min_tiles=32, max_tiles=None): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey(device=_DEFAULT_DEVICE, dtype=torch.bfloat16, min_rows=1, max_rows=4096, min_tiles=32, max_tiles=None): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey(device=_DEFAULT_DEVICE, dtype=torch.float32, min_rows=1, max_rows=4096, min_tiles=32, max_tiles=None): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
}
# fmt: on
"""Static forward policy, with concrete-device overrides before the shared default.

Generalized candidate derived from H100 80GB HBM3 measurements at softcap=30.
Replaying 837 eager and 27 graph cases against this rule agrees with every
conclusive ROW/PARALLEL winner; tied cases may select either implementation.
Unlike the old exact-width table, this policy interpolates vocabulary widths
and extrapolates beyond the measured maximum. New boundaries, unseen widths,
other caps and other hardware still require GPU performance validation.
Separate dtype entries allow future tuning without changing the selector.
ROW_PIPELINED remains an explicit-only experiment.
Clear select_softcapped_logprob_strategy's cache after modifying this map.
"""


@lru_cache(maxsize=1**10)
def select_softcapped_logprob_strategy(
    device_key: SoftcappedLogprobDevice,
    dtype: torch.dtype,
    n_rows: int,
    vocab_size: int,
) -> SoftcappedLogprobStrategy:
    """Match device ranges, then default ranges, then DEFAULT_FORWARD_STRATEGY.

    Cache by exact device, dtype, row count and vocabulary size. Repeated
    metadata skips matching after warmup; unseen or evicted keys scan the map.
    Clear this cache after changing FORWARD_STRATEGY_CONFIGS at runtime.
    Reject ambiguous matches instead of allowing insertion order to decide.
    An explicit device ROW rule overrides a default PARALLEL rule. Only metadata
    is inspected, never tensor values or gradient mode.

    Args:
        device_key: Concrete backend/GPU model returned by softcapped_logprob_device_key.
        dtype: Logits storage dtype; each rule matches this dtype exactly.
        n_rows: Actual number of rows M, including zero for empty inputs.
        vocab_size: Actual vocabulary width V; only policy matching uses ceil(V / _BLOCK_V).

    Returns:
        The matched strategy, or DEFAULT_FORWARD_STRATEGY when no rule matches.

    Raises:
        ValueError: More than one range matches in the same device scope.
    """
    n_tiles = (vocab_size + _BLOCK_V - 1) // _BLOCK_V
    for scope in (device_key, _DEFAULT_DEVICE):
        matched = None
        for key, strategy in FORWARD_STRATEGY_CONFIGS.items():
            if (
                key.device == scope
                and key.dtype == dtype
                and key.min_tiles <= n_tiles
                and (key.max_tiles is None or n_tiles <= key.max_tiles)
                and key.min_rows <= n_rows <= key.max_rows
            ):
                if matched is not None:
                    raise ValueError("overlapping forward strategy ranges")
                matched = strategy
        if matched is not None:
            return matched
    return DEFAULT_FORWARD_STRATEGY


@lru_cache(maxsize=None)
def _cuda_device_key(backend: str, index: int) -> SoftcappedLogprobDevice:
    """Cache the GPU model for a resolved CUDA/ROCm device index.

    The cache distinguishes indices; the returned key describes the model so
    identical GPUs can share tuning rules. Hardware-query errors propagate.
    """
    return SoftcappedLogprobDevice(backend=backend, gpu_name=torch.cuda.get_device_name(index))


def softcapped_logprob_device_key(device: torch.device) -> SoftcappedLogprobDevice:
    """Use the input tensor's GPU, including ROCm's torch.cuda namespace.

    Resolve an unspecified index before caching so changing the current GPU
    cannot accidentally reuse another GPU's configuration.
    Always return the concrete Device variant; DefaultDevice is reserved for
    shared configuration rules. Backends without a model lookup carry their
    actual backend name and an empty gpu_name, rather than becoming DefaultDevice.
    """
    if device.type == "cuda":
        backend = "rocm" if torch.version.hip is not None else "cuda"
        index = device.index if device.index is not None else torch.cuda.current_device()
        return _cuda_device_key(backend=backend, index=index)
    # No model-specific tuning is available for the other accepted backends.
    return SoftcappedLogprobDevice(backend=device.type, gpu_name="")


def _validate_softcap(softcap: float, vocab_size: int) -> None:
    """Check the scalar and the FP32 exp/sum range for a positive vocabulary width.

    Rely on type annotations. Require a finite positive FP32-normal cap,
    exp(-softcap) >= 2 * float32.tiny and V * exp(softcap) <= float32.max / 2.
    The factor-of-two margins keep exponentials and their sum away from limits.
    """
    limits = torch.finfo(torch.float32)
    if not math.isfinite(softcap) or softcap < limits.tiny:
        raise ValueError("softcap must be finite, positive and at least float32.tiny.")
    max_softcap = min(
        -math.log(2.0 * limits.tiny), math.log(limits.max / 2.0) - math.log(vocab_size)
    )
    if softcap > max_softcap:
        raise ValueError(
            f"softcap={softcap} with V={vocab_size} exceeds the FP32 exp/sum range "
            f"(maximum {max_softcap:.6g})."
        )


@triton.jit
def _ordered_tile_sum(values):
    """Sum one 1024-element FP32 tile with an explicit arithmetic tree.

    Fold eight adjacent values left to right in each of 128 groups. Within
    each 32-group section, pair the lower and upper halves at distances
    16, 8, 4, 2, 1. Finally combine the four sections as (s0+s2)+(s1+s3).
    Every addition rounds in FP32. These are logical groups, independent of
    physical warp size. Scheduling may change; the addition pairs may not.

    This spells out the CUDA tree observed in the captured row-939 diagnostic
    instead of relying on the compiler's choice for a 1024-wide tl.sum.
    A pure PyTorch test reference implements the same declared arithmetic.
    """
    tl.static_assert(values.shape[0] == 1024)
    groups = tl.arange(0, 128) * 8
    subtotal = tl.gather(values, groups, axis=0)
    for offset in tl.static_range(1, 8):
        subtotal = subtotal + tl.gather(values, groups + offset, axis=0)
    subtotal = tl.sum(tl.reshape(subtotal, (4, 2, 16)), axis=1)
    subtotal = tl.sum(tl.reshape(subtotal, (4, 2, 8)), axis=1)
    subtotal = tl.sum(tl.reshape(subtotal, (4, 2, 4)), axis=1)
    subtotal = tl.sum(tl.reshape(subtotal, (4, 2, 2)), axis=1)
    subtotal = tl.sum(tl.reshape(subtotal, (4, 2, 1)), axis=1)
    subtotal = tl.sum(tl.reshape(subtotal, (2, 2)), axis=0)
    return tl.sum(subtotal, axis=0)


@triton.jit
def _softcapped_selected_logprob_fwd_kernel(
    logits_ptr,
    token_ids_ptr,
    selected_logprob_ptr,
    row_max_ptr,
    log_sum_exp_shifted_ptr,
    vocab_size: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SOFTCAP: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr = None,
):
    # One program handles one row; widen before multiplying the row offset.
    row = tl.program_id(0).to(tl.int64)
    row_start = row * vocab_size
    cols = tl.arange(0, BLOCK_V)

    # 1. Find one maximum for the entire row before evaluating any exponential.
    # Padding must not replace a negative row maximum with zero.
    row_max = tl.full((), float("-inf"), tl.float32)
    for start in tl.range(0, vocab_size, BLOCK_V, num_stages=PIPELINE_STAGES):
        vocab_offsets = start + cols
        mask = vocab_offsets < vocab_size
        logits = tl.load(logits_ptr + row_start + vocab_offsets, mask=mask, other=0.0).to(tl.float32)  # noqa: E501 # fmt: skip
        scaled_logits = tl.div_rn(logits, SOFTCAP)
        softcapped = SOFTCAP * libdevice.tanh(scaled_logits)
        tile_max = tl.max(tl.where(mask, softcapped, float("-inf")), axis=0)
        row_max = tl.maximum(row_max, tile_max)
    # Give tied +0/-0 maxima the same representation in every execution schedule.
    row_max = tl.where(row_max == 0.0, 0.0, row_max)

    # 2. Shift every tile by that same maximum; add tile sums in vocabulary order.
    sum_exp = tl.zeros((), dtype=tl.float32)
    # None preserves the ROW baseline; ROW_PIPELINED requests three stages.
    # This compiler hint leaves tile arithmetic and reduction order unchanged.
    for start in tl.range(0, vocab_size, BLOCK_V, num_stages=PIPELINE_STAGES):
        vocab_offsets = start + cols
        mask = vocab_offsets < vocab_size
        logits = tl.load(logits_ptr + row_start + vocab_offsets, mask=mask, other=0.0).to(tl.float32)  # noqa: E501 # fmt: skip

        scaled_logits = tl.div_rn(logits, SOFTCAP)
        softcapped = SOFTCAP * libdevice.tanh(scaled_logits)
        exp_softcapped = libdevice.exp(softcapped - row_max)
        # Padded lanes never contribute to the sum, including all-negative rows.
        exp_softcapped = tl.where(mask, exp_softcapped, 0.0)
        tile_sum_exp = _ordered_tile_sum(exp_softcapped)
        sum_exp = sum_exp + tile_sum_exp

    log_sum_exp_shifted = tl.log(sum_exp)

    # 3. Load only the selected score and apply the same softcap arithmetic.
    token_id = tl.load(token_ids_ptr + row).to(tl.int64)
    selected_logit = tl.load(
        logits_ptr + row_start + token_id,
        mask=(token_id >= 0) & (token_id < vocab_size),
        other=float("nan"),
    ).to(tl.float32)
    selected_softcapped = SOFTCAP * libdevice.tanh(tl.div_rn(selected_logit, SOFTCAP))

    # Subtract the maximum first; do not round row_max + log_sum_exp_shifted.
    selected_logprob = (selected_softcapped - row_max) - log_sum_exp_shifted
    tl.store(selected_logprob_ptr + row, selected_logprob)
    tl.store(row_max_ptr + row, row_max)
    tl.store(log_sum_exp_shifted_ptr + row, log_sum_exp_shifted)


@triton.jit
def _softcapped_selected_logprob_fwd_max_partial_kernel(
    logits_ptr,
    tile_partials_ptr,
    vocab_size: tl.constexpr,
    n_tiles: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    """Write one softcapped maximum per tile; padded columns count as -infinity."""
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK_V)
    vocab_offsets = tile * BLOCK_V + cols
    mask = vocab_offsets < vocab_size
    logits = tl.load(logits_ptr + row * vocab_size + vocab_offsets, mask=mask, other=0.0).to(tl.float32)  # noqa: E501 # fmt: skip
    scaled_logits = tl.div_rn(logits, SOFTCAP)
    softcapped = SOFTCAP * libdevice.tanh(scaled_logits)
    tile_max = tl.max(tl.where(mask, softcapped, float("-inf")), axis=0)
    tl.store(tile_partials_ptr + row * n_tiles + tile, tile_max)


@triton.jit
def _softcapped_selected_logprob_fwd_max_merge_kernel(
    tile_partials_ptr,
    row_max_ptr,
    n_tiles: tl.constexpr,
):
    """Finish the whole-row maximum before any program computes shifted sums."""
    row = tl.program_id(0).to(tl.int64)
    row_max = tl.full((), float("-inf"), tl.float32)
    for tile in range(n_tiles):
        tile_max = tl.load(tile_partials_ptr + row * n_tiles + tile)
        row_max = tl.maximum(row_max, tile_max)
    row_max = tl.where(row_max == 0.0, 0.0, row_max)
    tl.store(row_max_ptr + row, row_max)


@triton.jit
def _softcapped_selected_logprob_fwd_partial_kernel(
    logits_ptr,
    row_max_ptr,
    partial_sum_exp_ptr,
    vocab_size: tl.constexpr,
    n_tiles: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    row_start = row * vocab_size
    cols = tl.arange(0, BLOCK_V)
    vocab_offsets = tile * BLOCK_V + cols
    mask = vocab_offsets < vocab_size
    row_max = tl.load(row_max_ptr + row)

    # Share the explicit 1024-element addition tree with ROW and ROW_PIPELINED.
    logits = tl.load(logits_ptr + row_start + vocab_offsets, mask=mask, other=0.0).to(tl.float32)  # noqa: E501 # fmt: skip
    scaled_logits = tl.div_rn(logits, SOFTCAP)
    softcapped = SOFTCAP * libdevice.tanh(scaled_logits)
    exp_softcapped = libdevice.exp(softcapped - row_max)
    exp_softcapped = tl.where(mask, exp_softcapped, 0.0)
    tile_sum_exp = _ordered_tile_sum(exp_softcapped)

    # Each program owns exactly one entry in the [M, n_tiles] scratch array.
    tl.store(partial_sum_exp_ptr + row * n_tiles + tile, tile_sum_exp)


@triton.jit
def _softcapped_selected_logprob_fwd_merge_kernel(
    logits_ptr,
    token_ids_ptr,
    row_max_ptr,
    partial_sum_exp_ptr,
    selected_logprob_ptr,
    log_sum_exp_shifted_ptr,
    vocab_size: tl.constexpr,
    n_tiles: tl.constexpr,
    SOFTCAP: tl.constexpr,
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
    row_max = tl.load(row_max_ptr + row)
    log_sum_exp_shifted = tl.log(sum_exp)

    token_id = tl.load(token_ids_ptr + row).to(tl.int64)
    selected_logit = tl.load(
        logits_ptr + row_start + token_id,
        mask=(token_id >= 0) & (token_id < vocab_size),
        other=float("nan"),
    ).to(tl.float32)
    selected_softcapped = SOFTCAP * libdevice.tanh(tl.div_rn(selected_logit, SOFTCAP))
    selected_logprob = (selected_softcapped - row_max) - log_sum_exp_shifted
    tl.store(selected_logprob_ptr + row, selected_logprob)
    tl.store(log_sum_exp_shifted_ptr + row, log_sum_exp_shifted)


@triton.jit
def _softcapped_selected_logprob_bwd_kernel(
    logits_ptr,
    token_ids_ptr,
    grad_selected_logprob_ptr,
    row_max_ptr,
    log_sum_exp_shifted_ptr,
    grad_logits_ptr,
    vocab_size: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    row_start = row * vocab_size
    cols = tl.arange(0, BLOCK_V)
    vocab_offsets = tile * BLOCK_V + cols
    mask = vocab_offsets < vocab_size

    # 1. Reuse both FP32 statistics without reconstructing an unshifted LSE.
    row_max = tl.load(row_max_ptr + row)
    log_sum_exp_shifted = tl.load(log_sum_exp_shifted_ptr + row)
    token_id = tl.load(token_ids_ptr + row).to(tl.int64)
    grad_selected_logprob = tl.load(grad_selected_logprob_ptr + row).to(tl.float32)

    # 2. Each program computes one tile using the shared, read-only row statistics.
    logits = tl.load(logits_ptr + row_start + vocab_offsets, mask=mask, other=0.0).to(tl.float32)  # noqa: E501 # fmt: skip

    scaled_logits = tl.div_rn(logits, SOFTCAP)
    tanh_scaled_logits = libdevice.tanh(scaled_logits)
    softcapped = SOFTCAP * tanh_scaled_logits
    probability = tl.exp((softcapped - row_max) - log_sum_exp_shifted)

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


def _validate_inputs(logits: torch.Tensor, token_ids: torch.Tensor, softcap: float) -> None:
    """Validate tensors and the FP32 row-sum bound for their vocabulary width."""
    if logits.device.type not in _SUPPORTED_DEVICES:
        raise RuntimeError(
            "The Triton core requires a GPU tensor; "
            f"supported device types are {_SUPPORTED_DEVICES}, got '{logits.device.type}'."
        )
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("logits must have shape [M, V] with V > 0.")
    _validate_softcap(softcap, logits.shape[1])
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
    row_max: torch.Tensor,
    log_sum_exp_shifted: torch.Tensor,
    softcap: float,
) -> None:
    _validate_inputs(logits, token_ids, softcap)
    if grad_selected_logprob.shape != logits.shape[:1]:
        raise ValueError("grad_selected_logprob must have shape [M].")
    if grad_selected_logprob.device != logits.device:
        raise ValueError("grad_selected_logprob must be on the logits device.")
    if grad_selected_logprob.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"grad_selected_logprob must have dtype {_SUPPORTED_DTYPES}, "
            f"got {grad_selected_logprob.dtype}."
        )
    for name, statistic in (
        ("row_max", row_max),
        ("log_sum_exp_shifted", log_sum_exp_shifted),
    ):
        if statistic.shape != logits.shape[:1]:
            raise ValueError(f"{name} must have shape [M].")
        if statistic.device != logits.device:
            raise ValueError(f"{name} must be on the logits device.")
        if statistic.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype float32.")


def _launch_softcapped_selected_logprob_fwd(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    softcap: float,
    *,
    pipeline_stages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return output, row_max and log_sum_exp_shifted as FP32 [M] tensors.

    The caller must validate tensor metadata and softcap before launching.
    Valid token IDs in [0, V) are a caller precondition. The kernel masks
    invalid IDs to NaN to avoid out-of-bounds reads, without a host/GPU sync.
    pipeline_stages sets the loop hint, not the launch num_stages. ROW_PIPELINED
    requests three stages; None retains the original unhinted ROW baseline.
    """
    n_rows, vocab_size = logits.shape
    logits_c = logits.contiguous()
    token_ids_c = token_ids.contiguous()
    output = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    row_max = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    log_sum_exp_shifted = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    if n_rows == 0:
        return output, row_max, log_sum_exp_shifted

    # ROCm tensors also use torch.cuda; do not enter CUDA for other backends.
    with torch.cuda.device(logits.device) if logits.device.type == "cuda" else nullcontext():
        _softcapped_selected_logprob_fwd_kernel[(n_rows,)](
            logits_c,
            token_ids_c,
            output,
            row_max,
            log_sum_exp_shifted,
            vocab_size,
            BLOCK_V=_BLOCK_V,
            SOFTCAP=softcap,
            PIPELINE_STAGES=pipeline_stages,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output, row_max, log_sum_exp_shifted


def _launch_softcapped_selected_logprob_fwd_parallel(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    softcap: float,
    *,
    merge_unroll: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Four stages with one reusable FP32 [M, ceil(V / 1024)] scratch array.

    The caller must validate tensor metadata and softcap before launching.
    Return output, row_max and log_sum_exp_shifted, just like the row loop.
    First reduce tile maxima to row_max. After that merge finishes, overwrite
    the same tile scratch with shifted sums, then add them in ROW's order.
    All four kernels use the same stream; no host synchronization is needed.
    merge_unroll=4 is a benchmark-only experiment; normal dispatch uses 1.
    """
    n_rows, vocab_size = logits.shape
    logits_c = logits.contiguous()
    token_ids_c = token_ids.contiguous()
    output = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    row_max = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    log_sum_exp_shifted = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    if n_rows == 0:
        return output, row_max, log_sum_exp_shifted

    n_tiles = triton.cdiv(vocab_size, _BLOCK_V)
    tile_partials = torch.empty((n_rows, n_tiles), device=logits.device, dtype=torch.float32)
    with torch.cuda.device(logits.device) if logits.device.type == "cuda" else nullcontext():
        _softcapped_selected_logprob_fwd_max_partial_kernel[(n_rows, n_tiles)](
            logits_c,
            tile_partials,
            vocab_size,
            n_tiles,
            BLOCK_V=_BLOCK_V,
            SOFTCAP=softcap,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _softcapped_selected_logprob_fwd_max_merge_kernel[(n_rows,)](
            tile_partials,
            row_max,
            n_tiles,
            num_warps=4,
            enable_fp_fusion=False,
        )
        # The maximum merge has consumed the scratch; reuse it for shifted sums.
        _softcapped_selected_logprob_fwd_partial_kernel[(n_rows, n_tiles)](
            logits_c,
            row_max,
            tile_partials,
            vocab_size,
            n_tiles,
            BLOCK_V=_BLOCK_V,
            SOFTCAP=softcap,
            num_warps=4,
            enable_fp_fusion=False,
        )
        # Merge in ascending tile order, not in program completion order.
        _softcapped_selected_logprob_fwd_merge_kernel[(n_rows,)](
            logits_c,
            token_ids_c,
            row_max,
            tile_partials,
            output,
            log_sum_exp_shifted,
            vocab_size,
            n_tiles,
            MERGE_UNROLL=merge_unroll,
            SOFTCAP=softcap,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output, row_max, log_sum_exp_shifted


def _launch_softcapped_selected_logprob_bwd(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    grad_selected_logprob: torch.Tensor,
    row_max: torch.Tensor,
    log_sum_exp_shifted: torch.Tensor,
    softcap: float,
    *,
    block_v: int = _BLOCK_V,
    num_warps: int = 4,
) -> torch.Tensor:
    """Return input-dtype grad_logits; grad_selected_logprob has shape [M].

    row_max and log_sum_exp_shifted are FP32 [M] statistics from the same forward.
    Keep them separate: reconstructing an unshifted log-sum would add rounding.
    The autograd wrapper supplies them together with the inputs and upstream gradient.
    As in forward, invalid token IDs produce NaN; valid IDs are a precondition.
    Private launch parameters let the benchmark sweep backward independently;
    autograd continues to use the default configuration.
    """
    _validate_backward_inputs(
        logits, token_ids, grad_selected_logprob, row_max, log_sum_exp_shifted, softcap
    )

    n_rows, vocab_size = logits.shape
    logits_c = logits.contiguous()
    token_ids_c = token_ids.contiguous()
    grad_selected_logprob_c = grad_selected_logprob.contiguous()
    row_max_c = row_max.contiguous()
    log_sum_exp_shifted_c = log_sum_exp_shifted.contiguous()
    grad_logits = torch.empty_like(logits_c)
    if n_rows == 0:
        return grad_logits

    grid = (n_rows, triton.cdiv(vocab_size, block_v))
    with torch.cuda.device(logits.device) if logits.device.type == "cuda" else nullcontext():
        _softcapped_selected_logprob_bwd_kernel[grid](
            logits_c,
            token_ids_c,
            grad_selected_logprob_c,
            row_max_c,
            log_sum_exp_shifted_c,
            grad_logits,
            vocab_size,
            BLOCK_V=block_v,
            SOFTCAP=softcap,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    return grad_logits


# Forward accepts (logits, token_ids, softcap); returns (output, row_max, log_sum_exp_shifted).
# Register new implementations here; the autograd wrapper only dispatches by key.
_FORWARD_LAUNCHERS: dict[
    SoftcappedLogprobStrategy,
    Callable[
        [torch.Tensor, torch.Tensor, float],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
] = {
    SoftcappedLogprobStrategy.ROW: _launch_softcapped_selected_logprob_fwd,
    # Historical unshifted H100 V=262144: stages=3 improved ROW at M<=512. Keep this
    # explicit: PARALLEL won the main comparison, and unmeasured shapes/devices
    # must not silently inherit a pipelined ROW fallback.
    SoftcappedLogprobStrategy.ROW_PIPELINED: partial(
        _launch_softcapped_selected_logprob_fwd, pipeline_stages=3
    ),
    SoftcappedLogprobStrategy.PARALLEL: _launch_softcapped_selected_logprob_fwd_parallel,
}


class _SoftcappedSelectedLogprobTritonFunction(torch.autograd.Function):
    """Connect the Triton forward/backward cores to PyTorch autograd."""

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        strategy: SoftcappedLogprobStrategy,
        softcap: float,
    ) -> torch.Tensor:
        logits_c = logits.contiguous()
        token_ids_c = token_ids.contiguous()
        launch_forward = _FORWARD_LAUNCHERS[strategy]
        selected_logprob, row_max, log_sum_exp_shifted = launch_forward(
            logits_c, token_ids_c, softcap
        )
        ctx.save_for_backward(logits_c, token_ids_c, row_max, log_sum_exp_shifted)
        ctx.softcap = softcap
        return selected_logprob

    @staticmethod
    def backward(ctx, grad_selected_logprob: torch.Tensor):
        logits, token_ids, row_max, log_sum_exp_shifted = ctx.saved_tensors
        grad_logits = None
        if ctx.needs_input_grad[0]:
            grad_logits = _launch_softcapped_selected_logprob_bwd(
                logits,
                token_ids,
                grad_selected_logprob,
                row_max,
                log_sum_exp_shifted,
                ctx.softcap,
            )
        # Token IDs, the strategy and the scalar softcap have no gradient.
        return grad_logits, None, None, None


class TritonSoftcappedSelectedLogprobOp:
    """Selected logprob after FP32 softcap, with first-order autograd.

    logits [M, V] must use a supported floating dtype on an accepted GPU; token_ids
    [M] must be int64 on the same device with values in [0, V). Output is FP32
    [M], and gradients use the logits dtype. Each forward call accepts softcap,
    defaulting to 30.0, as a positive scalar within the FP32 exp/sum range for
    the vocabulary. Autograd saves that call's value for its own backward.
    forward_impl=None selects by device, dtype, vocabulary width and row count.
    Explicit SoftcappedLogprobStrategy values bypass selection. All strategies
    reuse the same backward. ROW, ROW_PIPELINED and PARALLEL preserve arithmetic
    order, outputs, saved statistics and gradients. The row maximum, shifted
    exponentials, ordered sum, log and final subtraction all use FP32.
    """

    op_class = "logprob"

    def __init__(self, *, forward_impl: SoftcappedLogprobStrategy | None = None):
        self.forward_impl = forward_impl

    def __call__(
        self, logits: torch.Tensor, token_ids: torch.Tensor, *, softcap: float = 30.0
    ) -> torch.Tensor:
        return self.forward(logits, token_ids, softcap=softcap)

    def forward(
        self, logits: torch.Tensor, token_ids: torch.Tensor, *, softcap: float = 30.0
    ) -> torch.Tensor:
        """Compute with this call's softcap and save it for the matching backward."""
        # Validate before reading [M, V] or querying hardware. No GPU data is
        # read to choose a strategy, and training/inference use the same rule.
        _validate_inputs(logits, token_ids, softcap)
        strategy = self.forward_impl
        if strategy is None:
            n_rows, vocab_size = logits.shape
            strategy = select_softcapped_logprob_strategy(
                softcapped_logprob_device_key(logits.device),
                logits.dtype,
                n_rows,
                vocab_size,
            )
        return _SoftcappedSelectedLogprobTritonFunction.apply(
            logits, token_ids, strategy, float(softcap)
        )
