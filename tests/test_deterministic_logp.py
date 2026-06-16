# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from pathlib import Path

import pytest
import torch

from rl_engine.kernels.registry import kernel_registry

CUDA_SHAPE_CASES = (
    pytest.param(1, 1, 1, id="single-token-vocab"),
    pytest.param(1, 3, 2, id="tiny-vocab"),
    pytest.param(2, 5, 31, id="below-warp"),
    pytest.param(2, 7, 32, id="one-warp"),
    pytest.param(3, 4, 33, id="above-warp"),
    pytest.param(3, 3, 127, id="below-small-bucket"),
    pytest.param(3, 3, 128, id="small-bucket-boundary"),
    pytest.param(3, 3, 129, id="medium-bucket-start"),
    pytest.param(4, 3, 255, id="below-block"),
    pytest.param(4, 5, 256, id="one-block"),
    pytest.param(4, 5, 257, id="above-block"),
    pytest.param(2, 6, 1024, id="multi-block-stride"),
    pytest.param(2, 3, 4095, id="below-medium-boundary"),
    pytest.param(2, 3, 4096, id="medium-bucket-boundary"),
    pytest.param(2, 3, 4097, id="large-bucket-start"),
    pytest.param(2, 3, 4099, id="large-prime-vocab"),
    pytest.param(1, 2, 8192, id="large-power-two-vocab"),
)

CUDA_BUCKET_BOUNDARY_VOCABS = (128, 129, 4096, 4097)


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


def _skip_if_cuda_dtype_unavailable(dtype: torch.dtype) -> None:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")


def _dtype_tolerance(dtype: torch.dtype) -> float:
    if dtype is torch.float16:
        return 2e-3
    if dtype is torch.bfloat16:
        return 2e-2
    if dtype is torch.float64:
        return 1e-5
    return 1e-4


def _reference_selected_logp(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    ref = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(ref, dim=-1, index=token_ids.long().unsqueeze(-1)).squeeze(-1)


def _assert_close_to_reference(
    actual: torch.Tensor,
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.float32,
) -> None:
    expected = _reference_selected_logp(logits, token_ids).to(output_dtype)
    tolerance = _dtype_tolerance(output_dtype)
    assert torch.allclose(
        actual.float(),
        expected.float(),
        atol=tolerance,
        rtol=tolerance,
    )


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

    assert "kDeterministicLogpSmallBlockSize = 128" in text
    assert "kDeterministicLogpMediumBlockSize = 256" in text
    assert "kDeterministicLogpLargeBlockSize = 512" in text
    assert "kDeterministicLogpSmallVocabLimit = 128" in text
    assert "kDeterministicLogpMediumVocabLimit = 4096" in text
    assert "vocab_size <= kDeterministicLogpSmallVocabLimit" in text
    assert "vocab_size <= kDeterministicLogpMediumVocabLimit" in text
    assert "shared[lane]" not in text
    assert "shared[shared_idx]" in text
    assert "duplicate row ids" in text
    assert "writes are idempotent" in text
    assert "atomicAdd" not in text
    assert "cub::BlockReduce" not in text
    assert "select_deterministic" not in text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("batch_size,seq_len,vocab_size", CUDA_SHAPE_CASES)
@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32),
    ids=("fp16", "bf16", "fp32"),
)
def test_deterministic_logp_shape_dtype_matrix_cuda(
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
):
    _skip_if_cuda_dtype_unavailable(dtype)
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(
        1000 + batch_size * 101 + seq_len * 17 + vocab_size
    )
    logits = torch.randn(
        batch_size,
        seq_len,
        vocab_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    token_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        device=device,
        dtype=torch.long,
        generator=generator,
    )

    actual = op.apply_fp32(logits, token_ids)

    assert actual.shape == token_ids.shape
    assert actual.dtype == torch.float32
    _assert_close_to_reference(actual, logits, token_ids)


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
@pytest.mark.parametrize(
    "output_dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
    ids=("fp16", "bf16", "fp32", "fp64"),
)
def test_deterministic_logp_out_dtype_matrix_reuses_storage_cuda(output_dtype: torch.dtype):
    _skip_if_cuda_dtype_unavailable(output_dtype)
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(303)
    logits = torch.randn(3, 4, 257, device=device, dtype=torch.float16, generator=generator)
    token_ids = torch.randint(0, logits.size(-1), (3, 4), device=device, dtype=torch.long)
    output = torch.full(token_ids.shape, 123.0, device=device, dtype=output_dtype)

    actual = op.out(logits, token_ids, output)

    assert actual.data_ptr() == output.data_ptr()
    assert actual.dtype == output_dtype
    _assert_close_to_reference(actual, logits, token_ids, output_dtype=output_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_deterministic_logp_non_contiguous_inputs_cuda():
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(404)
    batch_size, seq_len, vocab_size = 3, 5, 129
    base_logits = torch.randn(
        batch_size,
        seq_len,
        vocab_size * 2,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    logits = base_logits[..., ::2]
    base_token_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len * 2),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    token_ids = base_token_ids[:, ::2]

    assert not logits.is_contiguous()
    assert not token_ids.is_contiguous()

    actual = op.apply_fp32(logits, token_ids)

    _assert_close_to_reference(actual, logits, token_ids)


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
@pytest.mark.parametrize("vocab_size", CUDA_BUCKET_BOUNDARY_VOCABS)
def test_deterministic_logp_bucket_boundaries_are_batch_and_indexed_invariant_cuda(
    vocab_size: int,
):
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    target_logits, target_ids = _make_target(
        device,
        torch.float16,
        seq_len=6,
        vocab_size=vocab_size,
    )
    baseline = op.apply_fp32(target_logits.unsqueeze(0), target_ids.unsqueeze(0))[0]

    for seed, batch_size, position in (
        (210 + vocab_size, 1, 0),
        (220 + vocab_size, 2, 1),
        (230 + vocab_size, 8, 3),
        (240 + vocab_size, 16, 9),
    ):
        logits, token_ids = _pack_target(
            target_logits,
            target_ids,
            batch_size=batch_size,
            position=position,
            seed=seed,
        )
        dense = op.apply_fp32(logits, token_ids)
        _assert_bitwise_equal(dense[position], baseline)

        row_start = position * target_ids.numel()
        row_indices = torch.arange(
            row_start,
            row_start + target_ids.numel(),
            device=device,
            dtype=torch.long,
        )
        indexed = op.indexed_fp32(logits, token_ids, row_indices)
        indexed_flat = indexed.reshape(-1)
        _assert_bitwise_equal(indexed_flat[row_indices], baseline)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_deterministic_logp_ignores_batch_noise_bitwise_cuda():
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    target_logits, target_ids = _make_target(
        device,
        torch.float16,
        seq_len=11,
        vocab_size=769,
    )
    baseline = op.apply_fp32(target_logits.unsqueeze(0), target_ids.unsqueeze(0))[0]

    for seed in range(20, 30):
        logits, token_ids = _pack_target(
            target_logits,
            target_ids,
            batch_size=32,
            position=seed % 32,
            seed=seed,
        )
        logits.add_(torch.randn_like(logits) * 0.01)
        token_ids.random_(0, logits.size(-1))
        logits[seed % 32].copy_(target_logits)
        token_ids[seed % 32].copy_(target_ids)

        actual = op.apply_fp32(logits, token_ids)[seed % 32]
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
def test_deterministic_logp_indexed_out_preserves_inactive_rows_cuda():
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(909)
    logits = torch.randn(3, 4, 263, device=device, dtype=torch.float16, generator=generator)
    token_ids = torch.randint(0, logits.size(-1), (3, 4), device=device, dtype=torch.long)
    dense = op.apply_fp32(logits, token_ids).reshape(-1)
    sentinel = torch.tensor(123.0, device=device, dtype=torch.float32)

    output = torch.full(token_ids.shape, sentinel.item(), device=device, dtype=torch.float32)
    empty_indices = torch.empty(0, device=device, dtype=torch.long)
    empty_result = op.indexed_out(logits, token_ids, empty_indices, output)

    assert empty_result.data_ptr() == output.data_ptr()
    assert torch.equal(empty_result, torch.full_like(empty_result, sentinel.item()))

    row_indices = torch.tensor([7, 0, 11, 7, 4], device=device, dtype=torch.long)
    output.fill_(sentinel.item())
    indexed = op.indexed_out(logits, token_ids, row_indices, output).reshape(-1)
    active = torch.unique(row_indices)
    inactive_mask = torch.ones_like(indexed, dtype=torch.bool)
    inactive_mask[active] = False

    _assert_bitwise_equal(indexed[active], dense[active])
    assert torch.equal(
        indexed[inactive_mask],
        torch.full_like(indexed[inactive_mask], sentinel.item()),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_deterministic_logp_indexed_fp32_empty_indices_zero_fills_cuda():
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    logits = torch.randn(2, 3, 17, device=device, dtype=torch.float16)
    token_ids = torch.randint(0, logits.size(-1), (2, 3), device=device, dtype=torch.long)
    row_indices = torch.empty(0, device=device, dtype=torch.long)

    actual = op.indexed_fp32(logits, token_ids, row_indices)

    assert actual.dtype == torch.float32
    assert torch.equal(actual, torch.zeros_like(actual))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_deterministic_logp_invalid_token_ids_zero_fill_cuda():
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    vocab_size = 17
    logits = torch.randn(1, 5, vocab_size, device=device, dtype=torch.float16)
    token_ids = torch.tensor([[-100, vocab_size, 0, vocab_size - 1, -1]], device=device)

    actual = op.apply_fp32(logits, token_ids)
    valid = (token_ids >= 0) & (token_ids < vocab_size)
    safe_token_ids = token_ids.clamp(0, vocab_size - 1)
    expected = _reference_selected_logp(logits, safe_token_ids)

    assert torch.equal(actual[~valid], torch.zeros_like(actual[~valid]))
    assert torch.allclose(actual[valid], expected[valid], atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_deterministic_logp_extreme_logits_are_stable_cuda():
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    vocab_size = 4099
    rows = 5
    logits = torch.empty(rows, vocab_size, device=device, dtype=torch.float32)

    logits[0].fill_(0.0)
    logits[1].fill_(80.0)
    logits[2].fill_(-80.0)
    logits[3] = torch.linspace(-80.0, 80.0, vocab_size, device=device)
    logits[4] = torch.linspace(80.0, -80.0, vocab_size, device=device)
    token_ids = torch.tensor([0, vocab_size - 1, vocab_size // 2, vocab_size - 1, 0], device=device)

    actual = op.apply_fp32(logits, token_ids)
    expected = _reference_selected_logp(logits, token_ids)

    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_deterministic_logp_rejects_bad_shapes_and_output_dtype_cuda():
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    logits = torch.randn(2, 3, 17, device=device, dtype=torch.float16)
    token_ids = torch.randint(0, logits.size(-1), (2, 3), device=device, dtype=torch.long)

    with pytest.raises(RuntimeError, match="token_ids length must match logits rows"):
        op.apply_fp32(logits, token_ids[:, :2])

    with pytest.raises(ValueError, match="output shape"):
        op.out(logits, token_ids, torch.empty(2, 2, device=device, dtype=torch.float32))

    with pytest.raises(RuntimeError, match="output dtype"):
        op.out(logits, token_ids, torch.empty(token_ids.shape, device=device, dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_deterministic_logp_out_of_range_indices_do_not_overwrite_output_cuda():
    op = _deterministic_cuda_op()
    device = torch.device("cuda")
    logits = torch.randn(2, 3, 29, device=device, dtype=torch.float16)
    token_ids = torch.randint(0, logits.size(-1), (2, 3), device=device, dtype=torch.long)
    dense = op.apply_fp32(logits, token_ids).reshape(-1)
    output = torch.full(token_ids.shape, -77.0, device=device, dtype=torch.float32)
    row_indices = torch.tensor([-1, 2, 999], device=device, dtype=torch.long)

    actual = op.indexed_out(logits, token_ids, row_indices, output).reshape(-1)

    assert actual[2].item() == pytest.approx(dense[2].item())
    inactive = torch.ones_like(actual, dtype=torch.bool)
    inactive[2] = False
    assert torch.equal(actual[inactive], torch.full_like(actual[inactive], -77.0))


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
