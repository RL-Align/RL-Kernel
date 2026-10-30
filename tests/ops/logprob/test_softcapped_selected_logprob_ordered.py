# SPDX-License-Identifier: Apache-2.0
"""Fixed-arithmetic conformance, separate from independent numerical accuracy.

The pure PyTorch reference defines the FP32 addition tree and cast points.
GPU primitive-boundary probes account explicitly for libdevice.log/exp, which
need not have native torch log/exp bits. Passing these tests does not waive
any independent FP32/autograd or FP64 accuracy failure.
"""

import pytest
import torch

from rl_engine.contracts.operators.softcapped_selected_logprob import (
    REDUCTION_CONFIG,
    SoftcappedLogprobReductionConfig,
)
from rl_engine.reference.logprob.softcapped_selected_logprob import (
    gradient_from_probability,
    ordered_backward,
    ordered_forward,
    selected_from_statistics,
)
from tests.helpers.softcapped_checks import assert_bitwise

pytestmark = pytest.mark.softcap_consistency
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_CAPS = (
    1e-6,
    1e-3,
    0.125,
    0.5,
    1.0,
    3.7,
    10.0,
    17.3,
    20.0,
    30.0,
    39.5,
    50.0,
    64.0,
    70.0,
)


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
        for name in (
            "output",
            "row_max",
            "tile_sums",
            "sum_exp",
            "log_sum_exp_shifted",
        ):
            assert_bitwise(getattr(single, name), getattr(full, name)[row : row + 1], name)
        assert_bitwise(
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
        empty,
        torch.empty(0, dtype=torch.int64),
        torch.empty(0),
        output_dtype=torch.float32,
    ).shape == (0, 1025)
    zero = ordered_forward(torch.tensor([[-0.0, 0.0]]), torch.tensor([0]), softcap=30.0)
    assert_bitwise(zero.row_max, torch.tensor([0.0]), "zero maximum")


def _max_scan_inputs(dtype, vocab, softcap):
    """Finite rows that expose mask, saturation, rounding and signed-zero mistakes."""
    generator = torch.Generator().manual_seed(467)
    logits = (softcap * torch.randn((10, vocab), generator=generator)).to(dtype)
    # Negative maximum in a masked tail: padding zero must not win.
    logits[1].fill_(-3.0 * softcap)
    logits[1, -1] = -0.25 * softcap
    logits[2].zero_()
    logits[2, ::2] = -0.0
    logits[3].fill_(-0.0)
    # Distinct logits can become tied scores after tanh saturates.
    logits[4].fill_(1000.0)
    logits[4, -1] = 3000.0
    logits[5].fill_(-3000.0)
    logits[5, -1] = -1000.0
    logits[6].fill_(-softcap)
    logits[6, 0] = softcap
    logits[6, -1] = softcap
    # Division may overflow to infinity for an otherwise finite valid input.
    logits[7].fill_(-torch.finfo(dtype).max)
    logits[7, -1] = torch.finfo(dtype).max
    value = torch.tensor(0.75 * softcap, dtype=dtype)
    logits[8].fill_(value)
    logits[8, -1] = torch.nextafter(value, torch.full_like(value, float("inf")))
    logits[9].fill_(torch.nextafter(torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)))
    ids = torch.full((logits.shape[0],), vocab - 1, dtype=torch.int64)
    return logits, ids


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (1, 1025, 3073))
@pytest.mark.parametrize("softcap", (torch.finfo(torch.float32).tiny, *_CAPS))
def test_ordered_max_logit_matches_softcap_before_max(dtype, vocab, softcap):
    """The fixed-path reference keeps the previous score maximum, bit for bit."""
    logits, ids = _max_scan_inputs(dtype, vocab, softcap)
    result = ordered_forward(logits, ids, softcap=softcap)
    previous_max = result.scores.amax(dim=-1)
    previous_max = torch.where(previous_max == 0.0, torch.zeros_like(previous_max), previous_max)
    assert_bitwise(result.row_max, previous_max, "max logit vs per-element softcap")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (1, 1025, 3073))
@pytest.mark.parametrize("softcap", (torch.finfo(torch.float32).tiny, *_CAPS))
def test_all_strategy_maxima_match_previous_per_element_softcap(dtype, vocab, softcap):
    """Compare saved maxima to the old GPU arithmetic, including scalar/vector tanh.

    The independent accuracy and full output/gradient consistency suites remain
    separate; this test isolates the maximum-pass optimization itself.
    """
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        _FORWARD_LAUNCHERS,
        _math_launch_options,
    )
    from tests.helpers.softcapped_path_kernels import softcap_before_max_probe

    logits, ids = _max_scan_inputs(dtype, vocab, softcap)
    logits, ids = logits.cuda(), ids.cuda()
    previous_max = torch.empty(logits.shape[0], device=logits.device, dtype=torch.float32)
    softcap_before_max_probe[(logits.shape[0],)](
        logits,
        previous_max,
        V=vocab,
        BLOCK_V=REDUCTION_CONFIG.tile_size,
        SOFTCAP=softcap,
        num_warps=4,
        **_math_launch_options(logits.device),
    )
    for strategy, launch in _FORWARD_LAUNCHERS.items():
        actual = launch(logits, ids, softcap)
        assert_bitwise(actual.row_max, previous_max, f"{strategy.value}: previous maximum")


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
    _check_all_strategies_fixed_arithmetic(dtype, vocab, softcap, REDUCTION_CONFIG.tile_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("tile_size", (1, 2, 4, 8, 16, 32, 64, 256, 512, 2048, 4096, 8192))
def test_all_strategies_with_configured_tile_width(dtype, tile_size, monkeypatch):
    """Same configured width: strategies agree, including tails and saved backward state."""
    from rl_engine.backends.shared.triton.logprob import softcapped_selected_logprob as core

    reduction_config = SoftcappedLogprobReductionConfig(tile_size=tile_size)
    monkeypatch.setattr(core, "REDUCTION_CONFIG", reduction_config)
    core.select_softcapped_logprob_strategy.cache_clear()
    core.select_softcapped_logprob_pipeline_config.cache_clear()
    try:
        _check_all_strategies_fixed_arithmetic(dtype, 2 * tile_size + 1, 30.0, tile_size)
    finally:
        core.select_softcapped_logprob_strategy.cache_clear()
        core.select_softcapped_logprob_pipeline_config.cache_clear()


def _check_all_strategies_fixed_arithmetic(dtype, vocab, softcap, tile_size):
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
        _math_launch_options,
        _softcapped_selected_logprob_fwd_partial_kernel,
    )
    from tests.helpers.softcapped_path_kernels import intrinsic_boundary_probe

    generator = torch.Generator().manual_seed(466)
    logits = (softcap * torch.randn((3, vocab), generator=generator)).to("cuda", dtype)
    logits[1, -1] = 3000.0
    logits[2].fill_(-3000.0)
    ids = torch.tensor([0, vocab - 1, vocab // 2], device="cuda")
    upstream = torch.tensor([0.5, -2.0, 0.0], device="cuda")
    expected = ordered_forward(logits, ids, softcap=softcap, tile_size=tile_size)
    n_tiles = triton.cdiv(vocab, tile_size)

    actual_partials = torch.empty_like(expected.tile_sums)
    math_options = _math_launch_options(logits.device)
    _softcapped_selected_logprob_fwd_partial_kernel[(3, n_tiles)](
        logits,
        expected.row_max,
        actual_partials,
        vocab,
        n_tiles,
        BLOCK_V=tile_size,
        SOFTCAP=softcap,
        MAX_GROUP_SIZE=REDUCTION_CONFIG.group_size,
        SECTION_COUNT=REDUCTION_CONFIG.section_count,
        num_warps=4,
        **math_options,
    )
    assert_bitwise(actual_partials, expected.tile_sums, "forward-exp and tile partials")

    primitive_logs = torch.empty_like(expected.sum_exp)
    primitive_probabilities = torch.empty_like(expected.scores)
    intrinsic_boundary_probe[(3, n_tiles)](
        expected.scores,
        expected.sum_exp,
        expected.row_max,
        primitive_logs,
        primitive_probabilities,
        V=vocab,
        BLOCK_V=tile_size,
        num_warps=4,
        **math_options,
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
        assert_bitwise(maximum, expected.row_max, f"{label}: row maximum")
        assert_bitwise(logarithm, primitive_logs, f"{label}: ordered total and log")
        assert_bitwise(output, expected_output, f"{label}: final subtraction")
        assert_bitwise(gradient, expected_gradient, f"{label}: backward arithmetic and cast")
