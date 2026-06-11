# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from pathlib import Path

import pytest
import torch

from rl_engine.kernels.registry import kernel_registry


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert _tensor_bytes(actual) == _tensor_bytes(expected)


def _deterministic_cuda_op():
    try:
        op = kernel_registry.get_op("logp_deterministic")
    except RuntimeError as exc:
        pytest.skip(f"deterministic logp backend is unavailable: {exc}")
    if op.__class__.__name__ != "DeterministicLogpCUDAOp":
        pytest.skip("deterministic CUDA logp extension is not compiled")
    return op


def _make_target(device: torch.device, dtype: torch.dtype, seq_len: int, vocab_size: int):
    generator = torch.Generator(device=device).manual_seed(1234)
    logits = torch.randn(
        seq_len,
        vocab_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    token_ids = torch.randint(
        0,
        vocab_size,
        (seq_len,),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    return logits, token_ids


def _pack_target(
    target_logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    batch_size: int,
    position: int,
    seed: int,
):
    generator = torch.Generator(device=target_logits.device).manual_seed(seed)
    seq_len, vocab_size = target_logits.shape
    logits = torch.randn(
        batch_size,
        seq_len,
        vocab_size,
        device=target_logits.device,
        dtype=target_logits.dtype,
        generator=generator,
    )
    token_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        device=target_logits.device,
        dtype=torch.long,
        generator=generator,
    )
    logits[position].copy_(target_logits)
    token_ids[position].copy_(target_ids)
    return logits, token_ids


def test_deterministic_logp_source_locks_reduction_contract():
    source = Path(__file__).resolve().parents[1] / "csrc" / "deterministic_logp_kernel.cu"
    text = source.read_text(encoding="utf-8")

    assert "kDeterministicLogpBlockSize = 256" in text
    assert "atomicAdd" not in text
    assert "cub::BlockReduce" not in text
    assert "select_deterministic" not in text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_deterministic_logp_repeatability_cuda():
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(2026)
    logits = torch.randn(6, 1021, device=device, dtype=torch.float16, generator=generator)
    token_ids = torch.randint(0, logits.size(-1), (6,), device=device, dtype=torch.long)

    baseline = op.apply_fp32(logits, token_ids)
    for _ in range(20):
        actual = op.apply_fp32(logits, token_ids)
        torch.cuda.synchronize()
        _assert_bitwise_equal(actual, baseline)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_deterministic_logp_batch_size_invariance_cuda():
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    target_logits, target_ids = _make_target(
        device,
        torch.float16,
        seq_len=7,
        vocab_size=4099,
    )
    baseline = op.apply_fp32(target_logits.unsqueeze(0), target_ids.unsqueeze(0))[0]

    for seed, batch_size, position in (
        (11, 1, 0),
        (12, 2, 1),
        (13, 4, 2),
        (14, 8, 5),
        (15, 16, 11),
    ):
        logits, token_ids = _pack_target(
            target_logits,
            target_ids,
            batch_size=batch_size,
            position=position,
            seed=seed,
        )
        actual = op.apply_fp32(logits, token_ids)[position]
        _assert_bitwise_equal(actual, baseline)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_deterministic_logp_batch_position_invariance_cuda():
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    target_logits, target_ids = _make_target(
        device,
        torch.float16,
        seq_len=5,
        vocab_size=2053,
    )
    baseline = op.apply_fp32(target_logits.unsqueeze(0), target_ids.unsqueeze(0))[0]

    for position in range(8):
        logits, token_ids = _pack_target(
            target_logits,
            target_ids,
            batch_size=8,
            position=position,
            seed=100 + position,
        )
        actual = op.apply_fp32(logits, token_ids)[position]
        _assert_bitwise_equal(actual, baseline)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_deterministic_logp_indexed_matches_dense_bits_cuda():
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(707)
    logits = torch.randn(4, 5, 1031, device=device, dtype=torch.float16, generator=generator)
    token_ids = torch.randint(0, logits.size(-1), (4, 5), device=device, dtype=torch.long)
    dense = op.apply_fp32(logits, token_ids)
    dense_flat = dense.reshape(-1)
    target_row = 7
    target_baseline = None

    index_sets = (
        torch.tensor([target_row], device=device, dtype=torch.long),
        torch.tensor([0, 3, target_row, 11, 19], device=device, dtype=torch.long),
        torch.arange(dense_flat.numel(), device=device, dtype=torch.long),
    )

    for row_indices in index_sets:
        indexed = op.indexed_fp32(logits, token_ids, row_indices)
        indexed_flat = indexed.reshape(-1)

        _assert_bitwise_equal(indexed_flat[row_indices], dense_flat[row_indices])

        active_mask = torch.zeros(dense_flat.numel(), device=device, dtype=torch.bool)
        active_mask[row_indices] = True
        assert torch.equal(indexed_flat[~active_mask], torch.zeros_like(indexed_flat[~active_mask]))

        current_target = indexed_flat[target_row : target_row + 1]
        if target_baseline is None:
            target_baseline = current_target.clone()
        else:
            _assert_bitwise_equal(current_target, target_baseline)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
def test_deterministic_logp_matches_reference_tolerance_cuda(dtype: torch.dtype):
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(808)
    logits = torch.randn(3, 4, 257, device=device, dtype=dtype, generator=generator)
    token_ids = torch.randint(0, logits.size(-1), (3, 4), device=device, dtype=torch.long)

    actual = op.apply_fp32(logits, token_ids)
    ref = torch.log_softmax(logits.float(), dim=-1)
    ref = torch.gather(ref, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)

    tolerance = 2e-3 if dtype is torch.float16 else 1e-4
    assert torch.allclose(actual, ref, atol=tolerance, rtol=tolerance)
