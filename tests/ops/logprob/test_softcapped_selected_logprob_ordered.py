# SPDX-License-Identifier: Apache-2.0
"""Fixed-arithmetic conformance, separate from independent numerical accuracy.

The pure PyTorch reference defines the FP32 addition tree and cast points.
GPU primitive-boundary probes account explicitly for tl.log/tl.exp, which
need not have native torch log/exp bits. Passing these tests does not waive
any independent FP32/autograd or FP64 accuracy failure.
"""

import pytest
import torch

from tests.helpers.softcapped_ordered_reference import (
    TILE_SIZE,
    fixed_tile_sum,
    gradient_from_probability,
    ordered_backward,
    ordered_exp_sum,
    ordered_forward,
    selected_from_statistics,
)

pytestmark = pytest.mark.softcap_consistency
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_CAPS = (1e-6, 1e-3, 0.125, 0.5, 1.0, 3.7, 10.0, 17.3, 20.0, 30.0, 39.5, 50.0, 64.0, 70.0)


def _assert_bits(actual, expected, stage):
    """Require matching dtype, shape and raw bytes, including signed zero."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, stage
    actual = actual.detach().clone(memory_format=torch.contiguous_format).view(torch.uint8)
    expected = expected.detach().clone(memory_format=torch.contiguous_format).view(torch.uint8)
    assert torch.equal(actual, expected), f"Fixed arithmetic mismatch at {stage}"


@pytest.mark.parametrize(
    "positions, expected",
    [
        ((0, 1, 2, 3), 1.0),  # Eight-element sequential fold, not adjacent pairs.
        ((0, 8, 128, 136), 2.0),  # Pair groups 0+16 and 1+17 first.
        ((0, 256, 512, 768), 2.0),  # Pair sections 0+2 and 1+3 first.
    ],
)
def test_fixed_tile_tree_has_explicit_nonassociative_pairs(positions, expected):
    tile = torch.zeros((1, TILE_SIZE), dtype=torch.float32)
    tile[0, list(positions)] = torch.tensor([1e20, 1.0, -1e20, 1.0])
    _assert_bits(fixed_tile_sum(tile), torch.tensor([expected]), "tile tree")


def test_fixed_tile_tree_rounds_each_add_in_fp32():
    tile = torch.zeros((1, TILE_SIZE), dtype=torch.float32)
    tile[0, :3] = torch.tensor([2**24, 1.0, 1.0])
    actual = fixed_tile_sum(tile)
    _assert_bits(actual, torch.tensor([float(2**24)]), "per-add FP32 rounding")
    assert actual.item() != tile.double().sum().float().item()


def test_tile_partials_merge_left_to_right_with_a_zero_initial_value():
    values = torch.zeros((1, 3 * TILE_SIZE + 1), dtype=torch.float32)
    values[0, ::TILE_SIZE] = torch.tensor([1e20, 1.0, -1e20, 1.0])
    partials, total = ordered_exp_sum(values)
    _assert_bits(partials, values[:, ::TILE_SIZE], "partial values")
    _assert_bits(total, torch.tensor([1.0]), "ascending tile merge")


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (1, 1023, 1024, 1025, 3073))
def test_ordered_reference_preserves_rows_tails_and_gradient_dtype(dtype, vocab):
    generator = torch.Generator().manual_seed(465)
    logits = (30 * torch.randn((3, vocab * 2), generator=generator)).to(dtype)[:, ::2]
    ids = torch.tensor([0, vocab // 2, vocab - 1])
    upstream = torch.tensor([0.5, -2.0, 0.0])
    full = ordered_forward(logits, ids, softcap=30.0)
    gradient = ordered_backward(full, ids, upstream, output_dtype=dtype)
    assert full.output.dtype == torch.float32 and not full.output.requires_grad
    assert gradient.dtype == dtype and not gradient.requires_grad
    for row in range(3):
        single = ordered_forward(logits[row : row + 1], ids[row : row + 1], softcap=30.0)
        for name in ("output", "row_max", "tile_sums", "sum_exp", "log_sum_exp_shifted"):
            _assert_bits(getattr(single, name), getattr(full, name)[row : row + 1], name)
        _assert_bits(
            ordered_backward(
                single, ids[row : row + 1], upstream[row : row + 1], output_dtype=dtype
            ),
            gradient[row : row + 1],
            "manual backward row independence",
        )


def test_ordered_reference_empty_rows_and_positive_zero_maximum():
    empty = ordered_forward(torch.empty((0, 1025)), torch.empty(0, dtype=torch.int64), softcap=30.0)
    assert empty.output.shape == (0,) and empty.tile_sums.shape == (0, 2)
    assert ordered_backward(
        empty, torch.empty(0, dtype=torch.int64), torch.empty(0), output_dtype=torch.float32
    ).shape == (0, 1025)
    zero = ordered_forward(torch.tensor([[-0.0, 0.0]]), torch.tensor([0]), softcap=30.0)
    _assert_bits(zero.row_max, torch.tensor([0.0]), "zero maximum")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("num_warps", (4, 8))
@pytest.mark.parametrize("seed", (415, 438, 939))
def test_production_tile_tree_matches_pytorch_on_identical_fp32_values(num_warps, seed):
    """Isolate reduction from exp/tanh; changing scheduling cannot change pairs."""
    from tests.helpers.softcapped_path_kernels import tile_sum_probe

    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((16, TILE_SIZE), generator=generator)
    values *= torch.pow(2.0, torch.randint(-30, 30, values.shape, generator=generator))
    values = values.cuda()
    expected = fixed_tile_sum(values)
    actual = torch.empty(16, device="cuda", dtype=torch.float32)
    tile_sum_probe[(16,)](values, actual, num_warps=num_warps, enable_fp_fusion=False)
    _assert_bits(actual, expected, "isolated production tile tree")


_PATH_CASES = [(1025, cap) for cap in _CAPS] + [
    (1, 30.0),
    (1023, 30.0),
    (1024, 30.0),
    (3073, 50.0),
    (32767, 64.0),
    (262145, 70.0),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab, softcap", _PATH_CASES)
def test_all_strategies_follow_pytorch_fixed_arithmetic(dtype, vocab, softcap):
    """Check actual partials, saved statistics, output and analytic backward.

    Native PyTorch supplies division/tanh/forward-exp. Tile partial equality
    checks those stages jointly with the fixed tree; the isolated test above
    distinguishes a tree error from an elementary-function difference.
    Only log and backward-exp pass through explicitly identified intrinsic
    probes. The independent accuracy suites never use these probes or this oracle.
    """
    import triton

    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
        _softcapped_selected_logprob_fwd_partial_kernel,
    )
    from tests.helpers.softcapped_path_kernels import intrinsic_boundary_probe

    generator = torch.Generator().manual_seed(466)
    logits = (softcap * torch.randn((3, vocab), generator=generator)).to("cuda", dtype)
    logits[1, -1] = 3000.0
    logits[2].fill_(-3000.0)
    ids = torch.tensor([0, vocab - 1, vocab // 2], device="cuda")
    upstream = torch.tensor([0.5, -2.0, 0.0], device="cuda")
    expected = ordered_forward(logits, ids, softcap=softcap)
    n_tiles = triton.cdiv(vocab, TILE_SIZE)

    actual_partials = torch.empty_like(expected.tile_sums)
    _softcapped_selected_logprob_fwd_partial_kernel[(3, n_tiles)](
        logits,
        expected.row_max,
        actual_partials,
        vocab,
        n_tiles,
        BLOCK_V=TILE_SIZE,
        SOFTCAP=softcap,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _assert_bits(actual_partials, expected.tile_sums, "forward-exp and tile partials")

    primitive_logs = torch.empty_like(expected.sum_exp)
    primitive_probabilities = torch.empty_like(expected.scores)
    intrinsic_boundary_probe[(3, n_tiles)](
        expected.scores,
        expected.sum_exp,
        expected.row_max,
        primitive_logs,
        primitive_probabilities,
        V=vocab,
        num_warps=4,
        enable_fp_fusion=False,
    )
    expected_output = selected_from_statistics(
        expected.scores, ids, expected.row_max, primitive_logs
    )
    expected_gradient = gradient_from_probability(
        expected.tanh_scaled, primitive_probabilities, ids, upstream, output_dtype=dtype
    )
    for strategy in (*SoftcappedLogprobStrategy, None):
        op = TritonSoftcappedSelectedLogprobOp(forward_impl=strategy)
        leaf = logits.detach().requires_grad_(True)
        output = op(leaf, ids, softcap=softcap)
        _, _, maximum, logarithm = output.grad_fn.saved_tensors
        gradient = torch.autograd.grad(output, leaf, grad_outputs=upstream)[0]
        label = "auto" if strategy is None else strategy.value
        _assert_bits(maximum, expected.row_max, f"{label}: row maximum")
        _assert_bits(logarithm, primitive_logs, f"{label}: ordered total and log")
        _assert_bits(output, expected_output, f"{label}: final subtraction")
        _assert_bits(gradient, expected_gradient, f"{label}: backward arithmetic and cast")
