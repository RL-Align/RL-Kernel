# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from rl_engine.executors.stateless_executor import (
    RewardAdapter,
    StatelessForwardInputs,
    StatelessForwardMode,
    StatelessForwardOutputs,
    StatelessForwardResult,
    TensorTreeSummary,
    default_reward_adapter,
    extract_kv_cache_outputs,
    score_reference_logprobs,
    score_rewards,
    summarize_tensor_tree,
)


@dataclass(frozen=True)
class PagedKVScoringConfig:
    """Configuration for a generation-style paged-KV scoring baseline."""

    mode: StatelessForwardMode = "reference"
    num_layers: int = 4
    num_kv_heads: int = 8
    head_dim: int = 32
    block_size: int = 16
    kv_cache_dtype: torch.dtype = torch.float16
    kv_cache_blocks: Optional[int] = None
    use_cache: bool = True
    detach_outputs: bool = True
    return_token_scores: bool = False
    temperature: float = 1.0
    output_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.mode not in {"reference", "reward", "both"}:
            raise ValueError("mode must be 'reference', 'reward', or 'both'")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be greater than zero")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be greater than zero")
        if self.block_size <= 0:
            raise ValueError("block_size must be greater than zero")
        if self.kv_cache_blocks is not None and self.kv_cache_blocks <= 0:
            raise ValueError("kv_cache_blocks must be greater than zero")
        if not self.use_cache:
            raise ValueError("PagedKVScoringConfig.use_cache must be True")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be greater than zero")


@dataclass(frozen=True)
class PagedKVCacheReservation:
    """Allocated paged-KV tensors plus the block table needed to address them."""

    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_tables: torch.Tensor
    sequence_lengths: torch.Tensor
    blocks_per_sequence: torch.Tensor
    block_size: int
    required_blocks: int
    reserved_blocks: int
    cache_bytes: int
    metadata_bytes: int
    reserved_bytes: int

    @property
    def reserved_mb(self) -> float:
        return self.reserved_bytes / 1_048_576.0


class PagedKVScoringBaseline:
    """
    Correctness-first baseline for generation-engine-style scoring.

    This wrapper reserves a paged KV-cache and block table before running the
    wrapped model with ``use_cache=True`` when the model accepts that keyword.
    It does not implement a vLLM engine; it provides a local, reproducible
    memory reservation baseline for comparing stateless scoring against a
    generation path that keeps paged KV state alive while scoring a full
    sequence.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[PagedKVScoringConfig] = None,
        *,
        reward_adapter: Optional[RewardAdapter] = None,
    ):
        self.model = model
        self.config = config or PagedKVScoringConfig()
        self.reward_adapter = reward_adapter or default_reward_adapter

    def score(self, inputs: StatelessForwardInputs) -> StatelessForwardResult:
        _validate_inputs(inputs, self.config)

        device = inputs.input_ids.device
        cuda_tracking = device.type == "cuda" and torch.cuda.is_available()
        if cuda_tracking:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        started_at = time.perf_counter()
        reservation = reserve_paged_kv_cache(inputs, self.config)
        with torch.no_grad():
            raw_outputs, use_cache_passed = _run_cache_forward(
                self.model,
                inputs,
                use_cache=self.config.use_cache,
            )
            kv_cache = extract_kv_cache_outputs(raw_outputs)
            kv_cache_summary = summarize_tensor_tree(kv_cache)
            outputs = StatelessForwardOutputs(
                raw=raw_outputs,
                logits=_extract_logits(raw_outputs),
                kv_cache=kv_cache,
            )

            reference_logps: Optional[torch.Tensor] = None
            rewards: Optional[torch.Tensor] = None
            token_scores: Optional[torch.Tensor] = None

            if self.config.mode in {"reference", "both"}:
                if outputs.logits is None:
                    raise ValueError("reference mode requires model outputs to expose logits")
                reference_logps = score_reference_logprobs(
                    outputs.logits,
                    inputs,
                    temperature=self.config.temperature,
                    output_dtype=self.config.output_dtype,
                )
                if self.config.return_token_scores:
                    token_scores = reference_logps

            if self.config.mode in {"reward", "both"}:
                rewards = score_rewards(
                    outputs,
                    inputs,
                    reward_adapter=self.reward_adapter,
                    output_dtype=self.config.output_dtype,
                )

            if self.config.detach_outputs:
                reference_logps = _detach_optional(reference_logps)
                rewards = _detach_optional(rewards)
                token_scores = _detach_optional(token_scores)

        if cuda_tracking:
            torch.cuda.synchronize(device)
        finished_at = time.perf_counter()

        metrics = collect_paged_kv_metrics(
            inputs,
            reservation,
            config=self.config,
            elapsed_seconds=finished_at - started_at,
            use_cache_passed=use_cache_passed,
            cuda_tracking=cuda_tracking,
            model_kv_cache_summary=kv_cache_summary,
        )
        return StatelessForwardResult(
            reference_logps=reference_logps,
            rewards=rewards,
            token_scores=token_scores,
            metrics=metrics,
        )


def reserve_paged_kv_cache(
    inputs: StatelessForwardInputs,
    config: PagedKVScoringConfig,
) -> PagedKVCacheReservation:
    """Allocate paged KV tensors and a dense block table for the scoring batch."""

    _validate_inputs(inputs, config)

    device = inputs.input_ids.device
    sequence_lengths = _bool_mask(inputs.attention_mask, device=device).sum(dim=1).to(torch.int32)
    if not bool((sequence_lengths > 0).all().item()):
        raise ValueError("attention_mask must contain at least one token per sequence")

    blocks_per_sequence = torch.div(
        sequence_lengths + config.block_size - 1,
        config.block_size,
        rounding_mode="floor",
    ).to(torch.int32)
    required_blocks = int(blocks_per_sequence.sum().item())
    reserved_blocks = int(config.kv_cache_blocks or required_blocks)
    if reserved_blocks < required_blocks:
        raise ValueError(
            f"kv_cache_blocks={reserved_blocks} cannot fit required blocks={required_blocks}"
        )

    key_cache = torch.empty(
        (
            config.num_layers,
            reserved_blocks,
            config.block_size,
            config.num_kv_heads,
            config.head_dim,
        ),
        device=device,
        dtype=config.kv_cache_dtype,
    )
    value_cache = torch.empty_like(key_cache)

    max_blocks_per_sequence = int(blocks_per_sequence.max().item())
    block_tables = torch.full(
        (inputs.input_ids.shape[0], max_blocks_per_sequence),
        -1,
        device=device,
        dtype=torch.int32,
    )
    cursor = 0
    for row, block_count_tensor in enumerate(blocks_per_sequence):
        block_count = int(block_count_tensor.item())
        block_tables[row, :block_count] = torch.arange(
            cursor,
            cursor + block_count,
            device=device,
            dtype=torch.int32,
        )
        cursor += block_count

    cache_bytes = _tensor_nbytes(key_cache) + _tensor_nbytes(value_cache)
    metadata_bytes = (
        _tensor_nbytes(block_tables)
        + _tensor_nbytes(sequence_lengths)
        + _tensor_nbytes(blocks_per_sequence)
    )
    return PagedKVCacheReservation(
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        sequence_lengths=sequence_lengths,
        blocks_per_sequence=blocks_per_sequence,
        block_size=config.block_size,
        required_blocks=required_blocks,
        reserved_blocks=reserved_blocks,
        cache_bytes=cache_bytes,
        metadata_bytes=metadata_bytes,
        reserved_bytes=cache_bytes + metadata_bytes,
    )


def collect_paged_kv_metrics(
    inputs: StatelessForwardInputs,
    reservation: PagedKVCacheReservation,
    *,
    config: PagedKVScoringConfig,
    elapsed_seconds: float,
    use_cache_passed: bool,
    cuda_tracking: bool,
    model_kv_cache_summary: TensorTreeSummary,
) -> dict[str, float | int | str | bool]:
    input_ids = inputs.input_ids
    active_tokens = int(_bool_mask(inputs.completion_mask, device=input_ids.device).sum().item())
    total_kv_cache_bytes = reservation.reserved_bytes + model_kv_cache_summary.total_bytes
    metrics: dict[str, float | int | str | bool] = {
        "baseline_kind": "generation_engine_paged_kv_reservation",
        "baseline_includes_model_kv_cache": model_kv_cache_summary.tensor_count > 0,
        "mode": config.mode,
        "batch_size": int(input_ids.shape[0]),
        "sequence_len": int(input_ids.shape[1]),
        "active_completion_tokens": active_tokens,
        "attention_tokens": int(reservation.sequence_lengths.sum().item()),
        "device": str(input_ids.device),
        "dtype": str(input_ids.dtype).replace("torch.", ""),
        "kv_cache_dtype": str(config.kv_cache_dtype).replace("torch.", ""),
        "elapsed_ms": elapsed_seconds * 1000.0,
        "use_cache": True,
        "use_cache_passed": bool(use_cache_passed),
        "detached_outputs": bool(config.detach_outputs),
        "paged_kv_layers": int(config.num_layers),
        "paged_kv_heads": int(config.num_kv_heads),
        "paged_kv_head_dim": int(config.head_dim),
        "paged_kv_block_size": int(config.block_size),
        "paged_kv_required_blocks": int(reservation.required_blocks),
        "paged_kv_blocks": int(reservation.reserved_blocks),
        "paged_kv_max_blocks_per_sequence": int(reservation.block_tables.shape[1]),
        "paged_kv_cache_reserved_mb": reservation.reserved_bytes / 1_048_576.0,
        "paged_kv_cache_payload_mb": reservation.cache_bytes / 1_048_576.0,
        "paged_kv_metadata_mb": reservation.metadata_bytes / 1_048_576.0,
        "model_kv_cache_output_present": model_kv_cache_summary.tensor_count > 0,
        "model_kv_cache_output_tensors": model_kv_cache_summary.tensor_count,
        "model_kv_cache_output_bytes": model_kv_cache_summary.total_bytes,
        "model_kv_cache_output_mb": model_kv_cache_summary.total_mb,
        "total_kv_cache_bytes": total_kv_cache_bytes,
        "total_kv_cache_mb": total_kv_cache_bytes / 1_048_576.0,
    }
    if cuda_tracking:
        device = input_ids.device
        metrics["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1_048_576.0
        metrics["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 1_048_576.0
    return metrics


def _validate_inputs(inputs: StatelessForwardInputs, config: PagedKVScoringConfig) -> None:
    input_ids = inputs.input_ids
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B, S], got {tuple(input_ids.shape)}")
    if inputs.attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask shape must match input_ids shape")
    if inputs.completion_mask.shape != input_ids.shape:
        raise ValueError("completion_mask shape must match input_ids shape")
    if inputs.labels is not None and inputs.labels.shape != input_ids.shape:
        raise ValueError("labels shape must match input_ids shape")
    if inputs.attention_mask.device != input_ids.device:
        raise ValueError("attention_mask device must match input_ids device")
    if inputs.completion_mask.device != input_ids.device:
        raise ValueError("completion_mask device must match input_ids device")
    if inputs.labels is not None and inputs.labels.device != input_ids.device:
        raise ValueError("labels device must match input_ids device")
    if config.mode in {"reference", "both"}:
        if input_ids.shape[1] < 2:
            raise ValueError("reference scoring requires sequence_len >= 2")
        if _completion_mask_starts_at_position_zero(
            inputs.completion_mask,
            device=input_ids.device,
        ):
            raise ValueError(
                "completion_mask[:, 0] must be False; completions cannot start at position 0"
            )
    if not bool(_bool_mask(inputs.completion_mask, device=input_ids.device).any().item()):
        raise ValueError("completion_mask must contain at least one active token")


def _run_cache_forward(
    model: torch.nn.Module,
    inputs: StatelessForwardInputs,
    *,
    use_cache: bool,
) -> tuple[Any, bool]:
    kwargs: dict[str, Any] = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
    }
    if _call_accepts_keyword(model, "use_cache"):
        kwargs["use_cache"] = use_cache
        return model(**kwargs), True

    try:
        kwargs["use_cache"] = use_cache
        return model(**kwargs), True
    except TypeError as exc:
        if "use_cache" not in str(exc):
            raise
        kwargs.pop("use_cache", None)
        return model(**kwargs), False


def _call_accepts_keyword(model: torch.nn.Module, keyword: str) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword:
            return True
    return False


def _extract_logits(raw_outputs: Any) -> Optional[torch.Tensor]:
    if isinstance(raw_outputs, torch.Tensor):
        return raw_outputs
    if isinstance(raw_outputs, Mapping):
        value = raw_outputs.get("logits")
        return value if isinstance(value, torch.Tensor) else None
    logits = getattr(raw_outputs, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(raw_outputs, (tuple, list)) and raw_outputs:
        first = raw_outputs[0]
        return first if isinstance(first, torch.Tensor) else None
    return None


def _bool_mask(mask: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    return mask.to(device=device, dtype=torch.bool)


def _completion_mask_starts_at_position_zero(
    completion_mask: torch.Tensor,
    *,
    device: torch.device,
) -> bool:
    if completion_mask.shape[1] == 0:
        return False
    return bool(_bool_mask(completion_mask, device=device)[:, 0].any().item())


def _detach_optional(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return tensor.detach() if tensor is not None else None


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())
