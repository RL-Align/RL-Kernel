# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Shared softcap + selected logprob with a fixed FP32 reduction order.

ROW, ROW_PIPELINED and PARALLEL share a whole-row maximum, the explicit
configured tile tree (1024 elements by default) and an ascending fold of tile sums. They change the
execution schedule while preserving the arithmetic and saved statistics.
Backward reuses the maximum and shifted log-sum, with one program per
row/vocabulary tile and no reduction. No full score/probability buffer is saved.

File order follows the dependencies used by the public Op:
1. Strategy/device types, range keys, tuning defaults and maps.
2. Cached strategy/pipeline selection and device lookup.
3. Input validation and numerical compiler options.
4. Shared kernel helpers, ROW/parallel forward kernels, then backward.
5. Launcher interfaces, forward/backward launchers and strategy dispatch.
6. Autograd state management and the public Op.
Independent PyTorch accuracy and fixed-order test references remain separate.
"""

import math
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Protocol, TypeAlias

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from rl_engine.backends.shared.triton.reduction import sum_partials, sum_tile
from rl_engine.contracts.operators.softcapped_selected_logprob import (
    REDUCTION_CONFIG,
    SUPPORTED_DTYPES,
    TRITON_SUPPORTED_DEVICES,
)

# Strategy, device and tuning configuration types.


class SoftcappedLogprobStrategy(str, Enum):
    """Choose the forward execution schedule; all strategies share one backward."""

    ROW = "row"
    """One program per row: find its maximum, then accumulate shifted tile sums in order."""

    ROW_PIPELINED = "row_pipelined"
    """Use ROW arithmetic with configured loop pipeline stages; explicit selection only."""

    PARALLEL = "parallel"
    """Find a shared row maximum in two stages, then shifted sums in two more stages."""


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


@dataclass(frozen=True)
class ForwardConfigKey:
    """A rectangular range of row counts and configured vocabulary tile counts.

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
    """Inclusive lower bound on ceil(V / REDUCTION_CONFIG.tile_size)."""

    max_tiles: int | None
    """Inclusive upper tile-count bound; None includes larger, potentially unmeasured V."""

    def matches(
        self,
        device: SoftcappedLogprobDeviceKey,
        dtype: torch.dtype,
        n_rows: int,
        n_tiles: int,
    ) -> bool:
        """Match one device scope and the actual row/tile counts, including range endpoints.

        Selectors check the concrete device and default scope separately, in
        that order. Strategy and pipeline tuning use the same range semantics.
        """
        return (
            self.device == device
            and self.dtype == dtype
            and self.min_rows <= n_rows <= self.max_rows
            and self.min_tiles <= n_tiles
            and (self.max_tiles is None or n_tiles <= self.max_tiles)
        )


@dataclass(frozen=True)
class RowPipelineConfig:
    """Scheduling parameters used only by ROW_PIPELINED, with ROW's arithmetic."""

    pipeline_stages: int
    """Positive tl.range pipeline depth, selected for a device/dtype/shape range."""

    def __post_init__(self) -> None:
        """Reject invalid depths before they reach the Triton compiler."""
        if self.pipeline_stages <= 0:
            raise ValueError("pipeline_stages must be positive")


# Tuning defaults and range maps; arithmetic configuration comes from contracts.


DEFAULT_FORWARD_STRATEGY = SoftcappedLogprobStrategy.ROW
"""Fallback when no device-specific or default range matches.

ROW needs one forward launch and no partial-sum scratch. This is a conservative
execution policy, not a claim that ROW is fastest outside the configured ranges.
Clear the selector cache after changing this fallback at runtime.
"""


DEFAULT_ROW_PIPELINE_CONFIG = RowPipelineConfig(pipeline_stages=3)
"""Provisional candidate for shapes without a measured pipeline range.

This retains the previous experimental depth, not a claim of optimal performance.
Clear select_softcapped_logprob_pipeline_config's cache after changing it.
"""


_DEFAULT_DEVICE = SoftcappedLogprobDefaultDevice()
"""Shared policy scope, consulted after rules for the input tensor's concrete device."""


# fmt: off
FORWARD_STRATEGY_CONFIGS: dict[ForwardConfigKey, SoftcappedLogprobStrategy] = {
    ForwardConfigKey(device=_DEFAULT_DEVICE, dtype=torch.float16, min_rows=1, max_rows=4096, min_tiles=32, max_tiles=None): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey(device=_DEFAULT_DEVICE, dtype=torch.bfloat16, min_rows=1, max_rows=4096, min_tiles=32, max_tiles=None): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
    ForwardConfigKey(device=_DEFAULT_DEVICE, dtype=torch.float32, min_rows=1, max_rows=4096, min_tiles=32, max_tiles=None): SoftcappedLogprobStrategy.PARALLEL,  # noqa: E501
}
# fmt: on
"""Static forward policy, with concrete-device overrides before the shared default.

Initial policy derived from earlier H100 measurements at softcap=30.
The current maximum-shifted kernels and explicit reduction tree still need
performance qualification. These ranges are scheduling choices, not input
limits or a guarantee of the fastest strategy on every shape and device.
Separate dtype entries allow future tuning without changing the selector.
ROW_PIPELINED remains an explicit-only experiment.
Clear select_softcapped_logprob_strategy's cache after modifying this map.
"""


ROW_PIPELINE_CONFIGS: dict[ForwardConfigKey, RowPipelineConfig] = {}
"""Measured ROW_PIPELINED depths by device, dtype, row range and vocabulary tile range.

Intentionally empty until server benchmarks establish useful ranges. This table
selects parameters only when ROW_PIPELINED runs; it does not change which strategy
automatic dispatch selects. Concrete-device rules precede shared default rules.
Unmatched inputs use DEFAULT_ROW_PIPELINE_CONFIG, including larger unseen shapes.
Clear select_softcapped_logprob_pipeline_config's cache after changing this table.
"""


# Resolve the execution schedule and pipeline depth from input metadata.


@lru_cache(maxsize=2**10)
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
        vocab_size: Actual vocabulary width V; matching uses the shared arithmetic tile size.

    Returns:
        The matched strategy, or DEFAULT_FORWARD_STRATEGY when no rule matches.

    Raises:
        ValueError: More than one range matches in the same device scope.
    """
    tile_size = REDUCTION_CONFIG.tile_size
    n_tiles = (vocab_size + tile_size - 1) // tile_size
    for scope in (device_key, _DEFAULT_DEVICE):
        matched = None
        for key, strategy in FORWARD_STRATEGY_CONFIGS.items():
            if key.matches(scope, dtype, n_rows, n_tiles):
                if matched is not None:
                    raise ValueError("overlapping forward strategy ranges")
                matched = strategy
        if matched is not None:
            return matched
    return DEFAULT_FORWARD_STRATEGY


@lru_cache(maxsize=2**10)
def select_softcapped_logprob_pipeline_config(
    device_key: SoftcappedLogprobDevice,
    dtype: torch.dtype,
    n_rows: int,
    vocab_size: int,
) -> RowPipelineConfig:
    """Choose ROW_PIPELINED parameters for this input, independently of strategy selection.

    Match concrete-device ranges first, shared ranges second, then use the
    provisional default. Reject overlapping matches at the same scope. Cache
    exact (device, dtype, M, V) metadata, without inspecting values or grad mode.
    Clear this cache after changing ROW_PIPELINE_CONFIGS, its fallback, or the
    reduction tile width. Explicit and automatic ROW_PIPELINED use this selector.
    """
    n_tiles = (vocab_size + REDUCTION_CONFIG.tile_size - 1) // REDUCTION_CONFIG.tile_size
    for scope in (device_key, _DEFAULT_DEVICE):
        matched = None
        for key, config in ROW_PIPELINE_CONFIGS.items():
            if key.matches(scope, dtype, n_rows, n_tiles):
                if matched is not None:
                    raise ValueError("overlapping row pipeline ranges")
                matched = config
        if matched is not None:
            return matched
    return DEFAULT_ROW_PIPELINE_CONFIG


@lru_cache(maxsize=2**10)
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


# Input validation and compiler options for the shared numerical contract.


def _validate_softcap(softcap: float, vocab_size: int) -> None:
    """Check the scalar and the FP32 exp/sum range for a positive vocabulary width.

    Rely on type annotations. Require a finite positive FP32-normal cap,
    exp(-softcap) >= 2 * float32.tiny and V * exp(softcap) <= float32.max / 2.
    These bounds retain the accepted domain of the original unshifted exp/sum
    implementation. The current kernels shift by the row maximum; relaxing the
    input domain requires separate numerical qualification.
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


def _validate_inputs(logits: torch.Tensor, token_ids: torch.Tensor, softcap: float) -> None:
    """Validate tensors and the FP32 row-sum bound for their vocabulary width."""
    if logits.device.type not in TRITON_SUPPORTED_DEVICES:
        raise RuntimeError(
            "The Triton core requires a GPU tensor; "
            f"supported device types are {TRITON_SUPPORTED_DEVICES}, got '{logits.device.type}'."
        )
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("logits must have shape [M, V] with V > 0.")
    _validate_softcap(softcap, logits.shape[1])
    if logits.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"logits must have dtype {SUPPORTED_DTYPES}, got {logits.dtype}.")
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
    """Validate the upstream and FP32 row statistics from the matching forward."""
    _validate_inputs(logits, token_ids, softcap)
    if grad_selected_logprob.shape != logits.shape[:1]:
        raise ValueError("grad_selected_logprob must have shape [M].")
    if grad_selected_logprob.device != logits.device:
        raise ValueError("grad_selected_logprob must be on the logits device.")
    if grad_selected_logprob.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            f"grad_selected_logprob must have dtype {SUPPORTED_DTYPES}, "
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


def _math_launch_options(device: torch.device) -> dict[str, bool]:
    """Return Triton compiler options that preserve the FP32 arithmetic contract.

    Launchers pass this dictionary as ``**options`` to ``kernel[grid](...)``.
    These are compiler settings, not tensor arguments passed into the kernel.

    ``enable_fp_fusion=False`` is shared by the backends: keep multiplication
    and addition separately rounded instead of automatically fusing them into
    an FMA with one rounding step. This fixes rounding points, not the sum tree.

    FTZ (flush-to-zero) is independent of FMA and has a different option name
    in each backend. NVIDIA CUDA uses ``enable_reflect_ftz=False`` to preserve
    subnormal values in libdevice; AMD ROCm uses ``allow_flush_denorm=False``
    to disallow flushing subnormals. The backend-specific keys are not
    interchangeable. Tiny nonzero exp results matter because a large finite
    upstream gradient can make their contribution significant.

    PyTorch uses ``device.type == "cuda"`` for both NVIDIA and AMD GPU tensors,
    including ROCm tensors. Within that branch, ``torch.version.hip`` identifies
    a ROCm build of PyTorch; no separate ``"hip"`` device branch is needed.
    Other backends receive only the common fusion flag; this helper sets no FTZ
    policy for them.
    """
    options = {"enable_fp_fusion": False}
    if device.type == "cuda":
        if torch.version.hip is None:
            from triton.backends.nvidia.compiler import CUDAOptions

            # Fail explicitly if this Triton version cannot request the FTZ policy.
            if not hasattr(CUDAOptions, "enable_reflect_ftz"):
                raise RuntimeError(
                    "Preserving tiny softcapped probabilities requires a Triton CUDA backend "
                    "with enable_reflect_ftz support (verified with Triton 3.7.1). "
                    "Upgrade Triton before using this operator."
                )
            options["enable_reflect_ftz"] = False
        else:
            # AMD GPU through PyTorch's CUDA-compatible device namespace.
            options["allow_flush_denorm"] = False
    return options


# Shared forward arithmetic, followed by ROW, parallel and backward kernels.


@triton.jit
def _softcapped_exp_tile_sum(
    logits_row_ptr,
    vocab_size: tl.constexpr,
    tile,
    row_max,
    BLOCK_V: tl.constexpr,
    SOFTCAP: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    SECTION_COUNT: tl.constexpr,
):
    """Compute one shifted-exponential tile and hand its values to the reducer.

    Both forward schedules call this operator-local helper. It owns loading,
    FP32 softcap/exp and tail masking; the shared utility only reduces the
    resulting values. The tile stays inside the enclosing kernel.
    """
    offsets = tl.cast(tile, tl.int64) * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offsets < vocab_size
    logits = tl.load(logits_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scaled = tl.div_rn(logits, SOFTCAP)
    softcapped = SOFTCAP * libdevice.tanh(scaled)
    exponentials = libdevice.exp(softcapped - row_max)
    exponentials = tl.where(mask, exponentials, 0.0)
    return sum_tile(
        exponentials,
        COMPUTE_DTYPE=tl.float32,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        SECTION_COUNT=SECTION_COUNT,
    )


@triton.jit
def _store_softcapped_selected_logprob(
    logits_row_ptr,
    token_id_ptr,
    selected_logprob_ptr,
    log_sum_exp_shifted_ptr,
    row_max,
    sum_exp,
    vocab_size: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    """Finish one row and save its selected logprob and shifted log-sum.

    Pointers already address this row's logits, token ID and scalar outputs.
    Both forward schedules use the same FP32 arithmetic and cast points here.
    row_max is saved separately by the caller or its preceding maximum kernel.
    Invalid token IDs produce NaN without reading outside the logits row.
    """
    log_sum_exp_shifted = libdevice.log(sum_exp)
    token_id = tl.load(token_id_ptr).to(tl.int64)
    selected_logit = tl.load(
        logits_row_ptr + token_id,
        mask=(token_id >= 0) & (token_id < vocab_size),
        other=float("nan"),
    ).to(tl.float32)
    selected_softcapped = SOFTCAP * libdevice.tanh(tl.div_rn(selected_logit, SOFTCAP))
    # Subtract the maximum first; do not round row_max + log_sum_exp_shifted.
    selected_logprob = (selected_softcapped - row_max) - log_sum_exp_shifted
    tl.store(selected_logprob_ptr, selected_logprob)
    tl.store(log_sum_exp_shifted_ptr, log_sum_exp_shifted)


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
    MAX_GROUP_SIZE: tl.constexpr,
    SECTION_COUNT: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr = None,
):
    """Scan raw logits for the maximum, then compute the ordered shifted-exp sum.

    Positive softcap is monotone: apply it once to the maximum logit instead
    of applying division/tanh to every element during the first scan. The saved
    row_max still contains a softcapped score, as required by backward.
    ROW and ROW_PIPELINED use this kernel with the same arithmetic configuration.
    PIPELINE_STAGES changes loop scheduling; tile totals still accumulate in
    ascending vocabulary order from positive zero.
    """
    # One program handles one row; widen before multiplying the row offset.
    row = tl.program_id(0).to(tl.int64)
    row_start = row * vocab_size
    cols = tl.arange(0, BLOCK_V)

    # 1. Find the maximum raw logit; padding must not replace a negative maximum.
    max_logit = tl.full((), float("-inf"), tl.float32)
    for start in tl.range(0, vocab_size, BLOCK_V, num_stages=PIPELINE_STAGES):
        vocab_offsets = start + cols
        mask = vocab_offsets < vocab_size
        logits = tl.load(logits_ptr + row_start + vocab_offsets, mask=mask, other=0.0).to(tl.float32)  # noqa: E501 # fmt: skip
        tile_max = tl.max(tl.where(mask, logits, float("-inf")), axis=0)
        max_logit = tl.maximum(max_logit, tile_max)
    # Use the same FP32 softcap arithmetic as the scores in the second scan.
    row_max = SOFTCAP * libdevice.tanh(tl.div_rn(max_logit, SOFTCAP))
    # Give tied +0/-0 maxima the same representation in every execution schedule.
    row_max = tl.where(row_max == 0.0, 0.0, row_max)

    # 2. Compute each tile locally and fold its sum in ascending tile order.
    n_tiles: tl.constexpr = (vocab_size + BLOCK_V - 1) // BLOCK_V
    sum_exp = tl.full((), 0.0, tl.float32)
    for tile in tl.range(0, n_tiles, num_stages=PIPELINE_STAGES):
        tile_sum_exp = _softcapped_exp_tile_sum(
            logits_ptr + row_start,
            vocab_size,
            tile,
            row_max,
            BLOCK_V=BLOCK_V,
            SOFTCAP=SOFTCAP,
            MAX_GROUP_SIZE=MAX_GROUP_SIZE,
            SECTION_COUNT=SECTION_COUNT,
        )
        sum_exp = sum_exp + tile_sum_exp

    # 3. Finish the selected logprob and save the statistics needed by backward.
    _store_softcapped_selected_logprob(
        logits_row_ptr=logits_ptr + row_start,
        token_id_ptr=token_ids_ptr + row,
        selected_logprob_ptr=selected_logprob_ptr + row,
        log_sum_exp_shifted_ptr=log_sum_exp_shifted_ptr + row,
        row_max=row_max,
        sum_exp=sum_exp,
        vocab_size=vocab_size,
        SOFTCAP=SOFTCAP,
    )
    tl.store(row_max_ptr + row, row_max)


@triton.jit
def _softcapped_selected_logprob_fwd_max_partial_kernel(
    logits_ptr,
    tile_partials_ptr,
    vocab_size: tl.constexpr,
    n_tiles: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Write one FP32 raw-logit maximum per tile; padded columns count as -infinity.

    Softcap is applied only after the merge finds the whole-row maximum.
    This stage therefore needs neither the softcap parameter nor division/tanh.
    """
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK_V)
    vocab_offsets = tile * BLOCK_V + cols
    mask = vocab_offsets < vocab_size
    logits = tl.load(logits_ptr + row * vocab_size + vocab_offsets, mask=mask, other=0.0).to(tl.float32)  # noqa: E501 # fmt: skip
    tile_max = tl.max(tl.where(mask, logits, float("-inf")), axis=0)
    tl.store(tile_partials_ptr + row * n_tiles + tile, tile_max)


@triton.jit
def _softcapped_selected_logprob_fwd_max_merge_kernel(
    tile_partials_ptr,
    row_max_ptr,
    n_tiles: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    """Merge raw-logit maxima, then softcap once to save the row's score maximum.

    The output keeps the same meaning as ROW's row_max and backward's saved
    statistic; only the temporary tile maxima contain uncapped logits.
    """
    row = tl.program_id(0).to(tl.int64)
    max_logit = tl.full((), float("-inf"), tl.float32)
    for tile in range(n_tiles):
        tile_max = tl.load(tile_partials_ptr + row * n_tiles + tile)
        max_logit = tl.maximum(max_logit, tile_max)
    row_max = SOFTCAP * libdevice.tanh(tl.div_rn(max_logit, SOFTCAP))
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
    MAX_GROUP_SIZE: tl.constexpr,
    SECTION_COUNT: tl.constexpr,
):
    """Store one FP32 shifted tile sum at [row, tile], using the completed row maximum."""
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    row_start = row * vocab_size
    row_max = tl.load(row_max_ptr + row)
    tile_sum_exp = _softcapped_exp_tile_sum(
        logits_ptr + row_start,
        vocab_size,
        tile,
        row_max,
        BLOCK_V=BLOCK_V,
        SOFTCAP=SOFTCAP,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        SECTION_COUNT=SECTION_COUNT,
    )

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
    """Merge stored sums in ROW's order, then write the output and backward statistic.

    Producers may finish in any order; their tile slots determine the addition
    order. MERGE_UNROLL changes loop expansion without changing combine pairs.
    """
    row = tl.program_id(0).to(tl.int64)
    row_start = row * vocab_size
    partial_row_start = row * n_tiles

    sum_exp = sum_partials(
        partial_sum_exp_ptr + partial_row_start,
        n_tiles,
        MERGE_UNROLL=MERGE_UNROLL,
    )
    row_max = tl.load(row_max_ptr + row)
    _store_softcapped_selected_logprob(
        logits_row_ptr=logits_ptr + row_start,
        token_id_ptr=token_ids_ptr + row,
        selected_logprob_ptr=selected_logprob_ptr + row,
        log_sum_exp_shifted_ptr=log_sum_exp_shifted_ptr + row,
        row_max=row_max,
        sum_exp=sum_exp,
        vocab_size=vocab_size,
        SOFTCAP=SOFTCAP,
    )


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
    """Write one gradient tile from read-only forward statistics, without reduction.

    Arithmetic is FP32; each output element is cast once to the logits dtype.
    Every element has one writer, so no cross-program accumulation is needed.
    """
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
    probability = libdevice.exp((softcapped - row_max) - log_sum_exp_shifted)

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


# Host launchers: return contract, forward schedules, then the shared backward.


@dataclass(frozen=True)
class _ForwardResult:
    """Forward output and the matching row statistics needed by backward.

    All fields are FP32 [M] tensors on the logits device. This object only
    groups tensor references; it does not copy or recompute their contents.
    """

    selected_logprob: torch.Tensor
    """One selected-token log-probability per row; returned by the public Op."""

    row_max: torch.Tensor
    """Maximum softcapped score per row, subtracted before exponentiation."""

    log_sum_exp_shifted: torch.Tensor
    """log(sum(exp(softcapped - row_max))) per row, saved separately from row_max."""


class _ForwardLauncher(Protocol):
    """Common callable interface for all forward strategies."""

    def __call__(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        softcap: float,
    ) -> _ForwardResult:
        """Launch forward using validated inputs.

        Args:
            logits: Floating scores with shape [M, V] on a supported GPU.
            token_ids: Int64 selected column indices [M] on the logits device.
            softcap: Positive softcap value for this call, with no internal default.

        Returns:
            The selected logprob and matching row maximum/shifted log-sum.
        """
        ...


def _launch_softcapped_selected_logprob_fwd(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    softcap: float,
    *,
    pipeline_stages: int | None = None,
) -> _ForwardResult:
    """Return the selected logprob and named FP32 [M] backward statistics.

    The caller must validate tensor metadata and softcap before launching.
    Valid token IDs in [0, V) are a caller precondition. The kernel masks
    invalid IDs to NaN to avoid out-of-bounds reads, without a host/GPU sync.
    pipeline_stages sets the loop hint, not the launch num_stages. ROW_PIPELINED
    selects a depth by input metadata; None retains the unhinted ROW baseline.
    Benchmarks can supply candidate depths directly without editing range rules.
    """
    n_rows, vocab_size = logits.shape
    logits_c = logits.contiguous()
    token_ids_c = token_ids.contiguous()
    output = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    row_max = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    log_sum_exp_shifted = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    result = _ForwardResult(
        selected_logprob=output,
        row_max=row_max,
        log_sum_exp_shifted=log_sum_exp_shifted,
    )
    if n_rows == 0:
        return result

    math_options = _math_launch_options(logits.device)
    # ROCm tensors also use torch.cuda; do not enter CUDA for other backends.
    with torch.cuda.device(logits.device) if logits.device.type == "cuda" else nullcontext():
        _softcapped_selected_logprob_fwd_kernel[(n_rows,)](
            logits_c,
            token_ids_c,
            output,
            row_max,
            log_sum_exp_shifted,
            vocab_size,
            BLOCK_V=REDUCTION_CONFIG.tile_size,
            SOFTCAP=softcap,
            MAX_GROUP_SIZE=REDUCTION_CONFIG.group_size,
            SECTION_COUNT=REDUCTION_CONFIG.section_count,
            PIPELINE_STAGES=pipeline_stages,
            num_warps=4,
            **math_options,
        )
    return result


def _launch_softcapped_selected_logprob_fwd_pipelined(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    softcap: float,
) -> _ForwardResult:
    """Resolve this call's pipeline depth by shape, then launch the shared ROW kernel.

    The caller validates inputs. Only tensor metadata and the input GPU identity
    select the config; batch splits may choose different depths without changing
    the prescribed reduction order. Backward reuses the saved forward statistics
    and has no pipeline configuration to resolve.
    """
    n_rows, vocab_size = logits.shape
    config = select_softcapped_logprob_pipeline_config(
        softcapped_logprob_device_key(logits.device), logits.dtype, n_rows, vocab_size
    )
    return _launch_softcapped_selected_logprob_fwd(
        logits, token_ids, softcap, pipeline_stages=config.pipeline_stages
    )


def _launch_softcapped_selected_logprob_fwd_parallel(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    softcap: float,
    *,
    merge_unroll: int = 1,
) -> _ForwardResult:
    """Four stages with one reusable FP32 [M, ceil(V / tile_size)] scratch array.

    The caller must validate tensor metadata and softcap before launching.
    Return the same named output and backward statistics as the row loop.
    First merge raw-logit tile maxima and softcap the row maximum once. After
    that merge finishes, overwrite the same tile scratch with shifted sums,
    then add them in ROW's order.
    All four kernels use the same stream; no host synchronization is needed.
    tile_size comes from REDUCTION_CONFIG. merge_unroll accepts positive integers
    for conformance checks; normal dispatch uses 1.
    """
    n_rows, vocab_size = logits.shape
    logits_c = logits.contiguous()
    token_ids_c = token_ids.contiguous()
    output = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    row_max = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    log_sum_exp_shifted = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    result = _ForwardResult(
        selected_logprob=output,
        row_max=row_max,
        log_sum_exp_shifted=log_sum_exp_shifted,
    )
    if n_rows == 0:
        return result

    n_tiles = triton.cdiv(vocab_size, REDUCTION_CONFIG.tile_size)
    tile_partials = torch.empty((n_rows, n_tiles), device=logits.device, dtype=torch.float32)
    math_options = _math_launch_options(logits.device)
    with torch.cuda.device(logits.device) if logits.device.type == "cuda" else nullcontext():
        _softcapped_selected_logprob_fwd_max_partial_kernel[(n_rows, n_tiles)](
            logits_c,
            tile_partials,
            vocab_size,
            n_tiles,
            BLOCK_V=REDUCTION_CONFIG.tile_size,
            num_warps=4,
            **math_options,
        )
        _softcapped_selected_logprob_fwd_max_merge_kernel[(n_rows,)](
            tile_partials,
            row_max,
            n_tiles,
            SOFTCAP=softcap,
            num_warps=4,
            **math_options,
        )
        # The maximum merge has consumed the scratch; reuse it for shifted sums.
        _softcapped_selected_logprob_fwd_partial_kernel[(n_rows, n_tiles)](
            logits_c,
            row_max,
            tile_partials,
            vocab_size,
            n_tiles,
            BLOCK_V=REDUCTION_CONFIG.tile_size,
            SOFTCAP=softcap,
            MAX_GROUP_SIZE=REDUCTION_CONFIG.group_size,
            SECTION_COUNT=REDUCTION_CONFIG.section_count,
            num_warps=4,
            **math_options,
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
            **math_options,
        )
    return result


def _launch_softcapped_selected_logprob_bwd(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    grad_selected_logprob: torch.Tensor,
    row_max: torch.Tensor,
    log_sum_exp_shifted: torch.Tensor,
    softcap: float,
) -> torch.Tensor:
    """Return input-dtype grad_logits; grad_selected_logprob has shape [M].

    row_max and log_sum_exp_shifted are FP32 [M] statistics from the same forward.
    Keep them separate: reconstructing an unshifted log-sum would add rounding.
    The autograd wrapper supplies them together with the inputs and upstream gradient.
    As in forward, invalid token IDs produce NaN; valid IDs are a precondition.
    The caller validates inputs. All forward strategies use this same backward
    configuration, with the shared tile width and four warps.
    """
    n_rows, vocab_size = logits.shape
    logits_c = logits.contiguous()
    token_ids_c = token_ids.contiguous()
    grad_selected_logprob_c = grad_selected_logprob.contiguous()
    row_max_c = row_max.contiguous()
    log_sum_exp_shifted_c = log_sum_exp_shifted.contiguous()
    grad_logits = torch.empty_like(logits_c)
    if n_rows == 0:
        return grad_logits

    grid = (n_rows, triton.cdiv(vocab_size, REDUCTION_CONFIG.tile_size))
    math_options = _math_launch_options(logits.device)
    with torch.cuda.device(logits.device) if logits.device.type == "cuda" else nullcontext():
        _softcapped_selected_logprob_bwd_kernel[grid](
            logits_c,
            token_ids_c,
            grad_selected_logprob_c,
            row_max_c,
            log_sum_exp_shifted_c,
            grad_logits,
            vocab_size,
            BLOCK_V=REDUCTION_CONFIG.tile_size,
            SOFTCAP=softcap,
            num_warps=4,
            **math_options,
        )
    return grad_logits


# Register new implementations here; the autograd wrapper only dispatches by key.
_FORWARD_LAUNCHERS: dict[SoftcappedLogprobStrategy, _ForwardLauncher] = {
    SoftcappedLogprobStrategy.ROW: _launch_softcapped_selected_logprob_fwd,
    # Pipeline scheduling is opt-in; automatic selection uses ROW or PARALLEL.
    SoftcappedLogprobStrategy.ROW_PIPELINED: _launch_softcapped_selected_logprob_fwd_pipelined,
    SoftcappedLogprobStrategy.PARALLEL: _launch_softcapped_selected_logprob_fwd_parallel,
}


# Autograd state and the public operator interface.


class _SoftcappedSelectedLogprobTritonFunction(torch.autograd.Function):
    """Connect validated public inputs and their saved statistics to PyTorch autograd."""

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
        result = launch_forward(logits_c, token_ids_c, softcap)
        ctx.save_for_backward(logits_c, token_ids_c, result.row_max, result.log_sum_exp_shifted)
        ctx.softcap = softcap
        return result.selected_logprob

    @staticmethod
    def backward(ctx, grad_selected_logprob: torch.Tensor):
        logits, token_ids, row_max, log_sum_exp_shifted = ctx.saved_tensors
        grad_logits = None
        if ctx.needs_input_grad[0]:
            _validate_backward_inputs(
                logits, token_ids, grad_selected_logprob, row_max, log_sum_exp_shifted, ctx.softcap
            )
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
    [M], and gradients use the logits dtype. Each forward call requires softcap
    as a positive scalar within the FP32 exp/sum range for
    the vocabulary. Autograd saves that call's value for its own backward.
    forward_impl=None selects by device, dtype, vocabulary width and row count.
    Explicit SoftcappedLogprobStrategy values bypass strategy selection;
    ROW_PIPELINED still selects its pipeline depth from input metadata. All strategies
    reuse the same backward. ROW, ROW_PIPELINED and PARALLEL preserve arithmetic
    order, outputs, saved statistics and gradients. The row maximum, shifted
    exponentials, ordered sum, log and final subtraction all use FP32.
    """

    op_class = "logprob"

    def __init__(self, *, forward_impl: SoftcappedLogprobStrategy | None = None):
        self.forward_impl = forward_impl

    def __call__(
        self, logits: torch.Tensor, token_ids: torch.Tensor, *, softcap: float
    ) -> torch.Tensor:
        return self.forward(logits, token_ids, softcap=softcap)

    def forward(
        self, logits: torch.Tensor, token_ids: torch.Tensor, *, softcap: float
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
