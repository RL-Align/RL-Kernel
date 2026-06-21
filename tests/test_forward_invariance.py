# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from rl_engine.testing import (
    BatchInvariantConfig,
    apply_rope_reference,
    assert_batch_invariant_across_configs,
    build_rope_cache,
    rotate_half,
)


@dataclass(frozen=True)
class _ElementwiseCase:
    name: str
    apply: Callable[[dict[str, torch.Tensor]], torch.Tensor]
    atol: float = 0.0
    rtol: float = 0.0


_ELEMENTWISE_CASES = (
    _ElementwiseCase("silu activation", lambda tensors: F.silu(tensors["x"])),
    _ElementwiseCase(
        "gelu activation",
        lambda tensors: F.gelu(tensors["x"], approximate="tanh"),
        atol=1e-6,
        rtol=1e-6,
    ),
    _ElementwiseCase("residual add", lambda tensors: tensors["x"] + tensors["residual"]),
    _ElementwiseCase("scalar scaling", lambda tensors: tensors["x"] * tensors["scale"]),
    _ElementwiseCase("bias add", lambda tensors: tensors["x"] + tensors["bias"]),
    _ElementwiseCase(
        "mask fill",
        lambda tensors: tensors["x"].masked_fill(tensors["mask"], -0.75),
    ),
)


def _available_devices() -> list[object]:
    devices = [pytest.param(torch.device("cpu"), id="cpu")]
    if torch.cuda.is_available():
        devices.append(pytest.param(torch.device("cuda"), id="cuda"))
    return devices


def _skip_unsupported_dtype(device: torch.device, dtype: torch.dtype) -> None:
    if device.type == "cpu" and dtype != torch.float32:
        pytest.skip("low-precision elementwise CPU coverage is intentionally omitted")
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 is not supported on this device")


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _embed_item(
    item: torch.Tensor,
    *,
    batch_size: int,
    target_index: int,
    generator: torch.Generator,
) -> torch.Tensor:
    batch = torch.randn(
        (batch_size, *item.shape[1:]),
        device=item.device,
        dtype=item.dtype,
        generator=generator,
    )
    batch[target_index].copy_(item[0])
    return batch


def _embed_bool_item(
    item: torch.Tensor,
    *,
    batch_size: int,
    target_index: int,
    generator: torch.Generator,
) -> torch.Tensor:
    batch = (
        torch.rand(
            (batch_size, *item.shape[1:]),
            device=item.device,
            generator=generator,
        )
        > 0.5
    )
    batch[target_index].copy_(item[0])
    return batch


def _embed_position_ids(
    item: torch.Tensor,
    *,
    batch_size: int,
    target_index: int,
    max_position: int,
    generator: torch.Generator,
) -> torch.Tensor:
    batch = torch.randint(
        low=0,
        high=max_position,
        size=(batch_size, *item.shape[1:]),
        device=item.device,
        dtype=item.dtype,
        generator=generator,
    )
    batch[target_index].copy_(item[0])
    return batch


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("case", _ELEMENTWISE_CASES, ids=lambda case: case.name)
def test_forward_path_elementwise_ops_are_batch_invariant(
    device: torch.device,
    dtype: torch.dtype,
    case: _ElementwiseCase,
):
    _skip_unsupported_dtype(device, dtype)
    generator = _generator(device, seed=149)
    hidden_size = 16

    target_x = torch.randn((1, 3, hidden_size), device=device, dtype=dtype, generator=generator)
    target_residual = torch.randn(
        target_x.shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    target_mask = torch.tensor(
        [[[False, True] * (hidden_size // 2)] * target_x.size(1)],
        device=device,
    )
    payload = {
        "x": target_x,
        "residual": target_residual,
        "scale": torch.tensor(0.125, device=device, dtype=dtype),
        "bias": torch.linspace(-0.5, 0.5, steps=hidden_size, device=device, dtype=dtype),
        "mask": target_mask,
    }

    def make_batched_inputs(
        config: BatchInvariantConfig,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        return {
            "x": _embed_item(
                target_x,
                batch_size=config.batch_size,
                target_index=config.target_index,
                generator=generator,
            ),
            "residual": _embed_item(
                target_residual,
                batch_size=config.batch_size,
                target_index=config.target_index,
                generator=generator,
            ),
            "scale": payload["scale"],
            "bias": payload["bias"],
            "mask": _embed_bool_item(
                target_mask,
                batch_size=config.batch_size,
                target_index=config.target_index,
                generator=generator,
            ),
        }

    assert_batch_invariant_across_configs(
        payload,
        case.apply,
        make_batched_inputs,
        case_name=case.name,
        atol=case.atol,
        rtol=case.rtol,
    )


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("target_dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_dtype_casts_are_batch_invariant(
    device: torch.device,
    target_dtype: torch.dtype,
):
    if (
        target_dtype == torch.bfloat16
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("CUDA bfloat16 is not supported on this device")

    generator = _generator(device, seed=150)
    target_x = torch.randn((1, 4, 9), device=device, dtype=torch.float32, generator=generator)
    payload = {"x": target_x}

    def apply_dtype_cast(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        return tensors["x"].to(dtype=target_dtype)

    def make_batched_inputs(
        config: BatchInvariantConfig,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        return {
            "x": _embed_item(
                target_x,
                batch_size=config.batch_size,
                target_index=config.target_index,
                generator=generator,
            )
        }

    assert_batch_invariant_across_configs(
        payload,
        apply_dtype_cast,
        make_batched_inputs,
        case_name=f"dtype cast to {target_dtype}",
    )
    assert apply_dtype_cast(payload).dtype == target_dtype


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_rope_fixed_positions_are_batch_invariant(
    device: torch.device,
    dtype: torch.dtype,
):
    _skip_unsupported_dtype(device, dtype)
    generator = _generator(device, seed=151)
    seq_len = 4
    num_heads = 2
    head_dim = 8
    max_position = 16

    target_q = torch.randn(
        (1, seq_len, num_heads, head_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    target_k = torch.randn(
        target_q.shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    target_position_ids = torch.tensor([[0, 1, 4, 7]], device=device, dtype=torch.long)
    cos, sin = build_rope_cache(max_position, head_dim, device=device, dtype=torch.float32)
    payload = {
        "query": target_q,
        "key": target_k,
        "position_ids": target_position_ids,
        "cos": cos,
        "sin": sin,
    }

    def apply_rope(tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rope_reference(
            tensors["query"],
            tensors["key"],
            tensors["position_ids"],
            tensors["cos"],
            tensors["sin"],
        )

    def make_batched_inputs(
        config: BatchInvariantConfig,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        return {
            "query": _embed_item(
                target_q,
                batch_size=config.batch_size,
                target_index=config.target_index,
                generator=generator,
            ),
            "key": _embed_item(
                target_k,
                batch_size=config.batch_size,
                target_index=config.target_index,
                generator=generator,
            ),
            "position_ids": _embed_position_ids(
                target_position_ids,
                batch_size=config.batch_size,
                target_index=config.target_index,
                max_position=max_position,
                generator=generator,
            ),
            "cos": cos,
            "sin": sin,
        }

    assert_batch_invariant_across_configs(
        payload,
        apply_rope,
        make_batched_inputs,
        case_name="RoPE fixed position",
    )


@pytest.mark.parametrize("device", _available_devices())
def test_rope_padding_position_ids_do_not_shift_valid_tokens(device: torch.device):
    generator = _generator(device, seed=152)
    dtype = torch.float32
    num_heads = 2
    head_dim = 8
    valid_positions = torch.tensor([1, 3, 5], device=device)
    compact_position_ids = torch.tensor([[0, 1, 2]], device=device, dtype=torch.long)
    cos, sin = build_rope_cache(8, head_dim, device=device)

    compact_q = torch.randn(
        (1, 3, num_heads, head_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    compact_k = torch.randn(
        compact_q.shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    expected_q, expected_k = apply_rope_reference(
        compact_q,
        compact_k,
        compact_position_ids,
        cos,
        sin,
    )

    padded_q = torch.randn((1, 6, num_heads, head_dim), device=device, generator=generator)
    padded_k = torch.randn(padded_q.shape, device=device, generator=generator)
    padded_q[0, valid_positions] = compact_q[0]
    padded_k[0, valid_positions] = compact_k[0]
    padded_position_ids = torch.zeros((1, 6), device=device, dtype=torch.long)
    padded_position_ids[0, valid_positions] = compact_position_ids[0]

    actual_q, actual_k = apply_rope_reference(padded_q, padded_k, padded_position_ids, cos, sin)

    assert torch.equal(actual_q[0, valid_positions], expected_q[0])
    assert torch.equal(actual_k[0, valid_positions], expected_k[0])


@pytest.mark.parametrize("device", _available_devices())
def test_rope_packed_sequence_position_reset_matches_standalone_segment(device: torch.device):
    generator = _generator(device, seed=153)
    dtype = torch.float32
    num_heads = 2
    head_dim = 8
    segment_len = 3
    cos, sin = build_rope_cache(8, head_dim, device=device)
    local_position_ids = torch.tensor([[0, 1, 2]], device=device, dtype=torch.long)

    target_q = torch.randn(
        (1, segment_len, num_heads, head_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    target_k = torch.randn(
        target_q.shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    expected_q, expected_k = apply_rope_reference(target_q, target_k, local_position_ids, cos, sin)

    packed_q = torch.randn(
        (1, segment_len * 2, num_heads, head_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    packed_k = torch.randn(
        packed_q.shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    packed_q[:, segment_len:].copy_(target_q)
    packed_k[:, segment_len:].copy_(target_k)
    packed_position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]], device=device, dtype=torch.long)

    actual_q, actual_k = apply_rope_reference(packed_q, packed_k, packed_position_ids, cos, sin)

    assert torch.equal(actual_q[:, segment_len:], expected_q)
    assert torch.equal(actual_k[:, segment_len:], expected_k)


def test_rope_cache_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="max_position"):
        build_rope_cache(0, 8)
    with pytest.raises(ValueError, match="head_dim"):
        build_rope_cache(4, 7)
    with pytest.raises(ValueError, match="base"):
        build_rope_cache(4, 8, base=0.0)


@pytest.mark.parametrize("dtype", (torch.bool, torch.int64, torch.complex64))
def test_rope_cache_rejects_non_floating_dtypes(dtype: torch.dtype):
    with pytest.raises(ValueError, match="floating-point dtype"):
        build_rope_cache(4, 8, dtype=dtype)


def test_rope_application_rejects_invalid_inputs():
    query = torch.randn(1, 2, 1, 8)
    key = torch.randn_like(query)
    cos, sin = build_rope_cache(4, 8)
    position_ids = torch.tensor([[0, 1]])

    with pytest.raises(ValueError, match="same shape"):
        apply_rope_reference(query, key[:, :, :, :4], position_ids, cos, sin)
    with pytest.raises(ValueError, match="same dtype"):
        apply_rope_reference(query, key.to(dtype=torch.float64), position_ids, cos, sin)
    with pytest.raises(ValueError, match="position_ids"):
        apply_rope_reference(query, key, torch.tensor([[0.0, 1.0]]), cos, sin)
    with pytest.raises(ValueError, match="cos and sin must have the same dtype"):
        apply_rope_reference(query, key, position_ids, cos, sin.to(dtype=torch.float64))
    with pytest.raises(ValueError, match="non-negative"):
        apply_rope_reference(query, key, torch.tensor([[0, -1]]), cos, sin)
    with pytest.raises(ValueError, match="cache length"):
        apply_rope_reference(query, key, torch.tensor([[0, 4]]), cos, sin)
    with pytest.raises(ValueError, match="even"):
        rotate_half(torch.randn(1, 7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rope_application_rejects_device_mismatches():
    query = torch.randn(1, 2, 1, 8, device="cuda")
    key = torch.randn_like(query)
    cos, sin = build_rope_cache(4, 8, device="cuda")
    position_ids = torch.tensor([[0, 1]], device="cuda")

    with pytest.raises(ValueError, match="position_ids must be on the same device"):
        apply_rope_reference(query, key, position_ids.cpu(), cos, sin)
    with pytest.raises(ValueError, match="cos and sin must be on the same device"):
        apply_rope_reference(query, key, position_ids, cos.cpu(), sin.cpu())
