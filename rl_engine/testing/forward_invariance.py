# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch

BatchInvariantOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class BatchInvariantConfig:
    """One target-row placement in a batch-invariance sweep."""

    batch_size: int
    target_index: int
    seed: int
    label: str

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not 0 <= self.target_index < self.batch_size:
            raise ValueError("target_index must be within the batch")


DEFAULT_BATCH_INVARIANT_SWEEP: tuple[BatchInvariantConfig, ...] = (
    BatchInvariantConfig(batch_size=1, target_index=0, seed=14901, label="batch1"),
    BatchInvariantConfig(batch_size=2, target_index=0, seed=14902, label="batch2-first"),
    BatchInvariantConfig(batch_size=4, target_index=2, seed=14903, label="batch4-middle"),
    BatchInvariantConfig(batch_size=9, target_index=8, seed=14904, label="batch9-last"),
)

BatchInvariantOp = Callable[[Mapping[str, torch.Tensor]], BatchInvariantOutput]
BatchInputFactory = Callable[[BatchInvariantConfig, torch.Generator], Mapping[str, torch.Tensor]]


def assert_batch_invariant_across_configs(
    reference_inputs: Mapping[str, torch.Tensor],
    op: BatchInvariantOp,
    make_batched_inputs: BatchInputFactory,
    *,
    configs: Sequence[BatchInvariantConfig] = DEFAULT_BATCH_INVARIANT_SWEEP,
    reference_index: int = 0,
    case_name: str = "batch-invariant op",
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    """Assert that a target row is stable across batch placements."""

    reference_output = _select_batch_output(op(reference_inputs), reference_index)
    device = _first_tensor(reference_inputs).device

    for config in configs:
        generator = torch.Generator(device=device)
        generator.manual_seed(config.seed)
        batched_inputs = make_batched_inputs(config, generator)
        batched_output = op(batched_inputs)
        actual = _select_batch_output(batched_output, config.target_index)
        _assert_outputs_match(
            actual,
            reference_output,
            case_name=f"{case_name}/{config.label}",
            atol=atol,
            rtol=rtol,
        )


def _select_batch_output(
    output: BatchInvariantOutput,
    index: int,
) -> BatchInvariantOutput:
    if isinstance(output, torch.Tensor):
        return output[index].clone()
    return tuple(item[index].clone() for item in output)


def _assert_outputs_match(
    actual: BatchInvariantOutput,
    expected: BatchInvariantOutput,
    *,
    case_name: str,
    atol: float,
    rtol: float,
) -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if not _tensor_matches(actual, expected, atol=atol, rtol=rtol):
            raise AssertionError(f"{case_name} drifted across batch configs")
        return

    if not isinstance(actual, tuple) or not isinstance(expected, tuple):
        raise TypeError("actual and expected outputs must have matching structures")
    if len(actual) != len(expected):
        raise AssertionError(f"{case_name} output arity changed across batch configs")

    for output_index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
        if not _tensor_matches(actual_item, expected_item, atol=atol, rtol=rtol):
            raise AssertionError(f"{case_name} output {output_index} drifted across batch configs")


def _tensor_matches(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> bool:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    if atol == 0.0 and rtol == 0.0:
        return torch.equal(actual, expected)
    return torch.allclose(actual, expected, atol=atol, rtol=rtol)


def _first_tensor(inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    for value in inputs.values():
        if isinstance(value, torch.Tensor):
            return value
    raise ValueError("reference_inputs must contain at least one tensor")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension for LLaMA/Qwen-style RoPE."""

    if x.size(-1) % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    x_first, x_second = x.chunk(2, dim=-1)
    return torch.cat((-x_second, x_first), dim=-1)


def build_rope_cache(
    max_position: int,
    head_dim: int,
    *,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic RoPE cos/sin lookup table."""

    if max_position <= 0:
        raise ValueError("max_position must be greater than zero")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer")
    if base <= 0.0:
        raise ValueError("base must be greater than zero")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("RoPE cache dtype must be a floating-point dtype")

    target_device = torch.device("cpu") if device is None else torch.device(device)
    positions = torch.arange(max_position, device=target_device, dtype=torch.float32)
    dims = torch.arange(0, head_dim, 2, device=target_device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (dims / float(head_dim)))
    freqs = torch.outer(positions, inv_freq)
    angles = torch.cat((freqs, freqs), dim=-1)

    return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)


def apply_rope_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    position_ids: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE with explicit position-id lookup and no batch-dependent reduction."""

    _validate_rope_inputs(query, key, position_ids, cos, sin)

    batch_size, seq_len, _, head_dim = query.shape
    flat_position_ids = position_ids.reshape(-1).long()
    cos_pos = cos.index_select(0, flat_position_ids).reshape(batch_size, seq_len, 1, head_dim)
    sin_pos = sin.index_select(0, flat_position_ids).reshape(batch_size, seq_len, 1, head_dim)
    cos_pos = cos_pos.to(dtype=query.dtype)
    sin_pos = sin_pos.to(dtype=query.dtype)

    query_rotated = (query * cos_pos) + (rotate_half(query) * sin_pos)
    key_rotated = (key * cos_pos) + (rotate_half(key) * sin_pos)
    return query_rotated, key_rotated


def _validate_rope_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    position_ids: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> None:
    if query.shape != key.shape:
        raise ValueError("query and key must have the same shape")
    if query.dtype != key.dtype:
        raise ValueError("query and key must have the same dtype")
    if query.dim() != 4:
        raise ValueError("query and key must have shape [batch, seq, heads, head_dim]")

    batch_size, seq_len, _, head_dim = query.shape
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    if tuple(position_ids.shape) != (batch_size, seq_len):
        raise ValueError("position_ids must have shape [batch, seq]")
    if position_ids.dtype == torch.bool or position_ids.is_floating_point():
        raise ValueError("position_ids must use an integer dtype")

    if cos.shape != sin.shape:
        raise ValueError("cos and sin must have the same shape")
    if cos.dtype != sin.dtype:
        raise ValueError("cos and sin must have the same dtype")
    if cos.dim() != 2 or cos.size(-1) != head_dim:
        raise ValueError("cos and sin must have shape [max_position, head_dim]")
    if cos.device != query.device or sin.device != query.device:
        raise ValueError("cos and sin must be on the same device as query and key")
    if position_ids.device != query.device:
        raise ValueError("position_ids must be on the same device as query and key")

    if position_ids.numel() == 0:
        return

    min_position = int(position_ids.min().item())
    max_position = int(position_ids.max().item())
    if min_position < 0:
        raise ValueError("position_ids must be non-negative")
    if max_position >= cos.size(0):
        raise ValueError("position_ids exceed the RoPE cache length")
