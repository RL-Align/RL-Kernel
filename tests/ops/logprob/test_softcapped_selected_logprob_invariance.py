# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Bitwise invariance across batching, layouts, strategies and gradient modes."""

from functools import partial

import pytest
import torch

from rl_engine.reference.logprob.softcapped_selected_logprob import (
    NativeSoftcappedSelectedLogprobOp,
    softcapped_selected_logprob,
)
from tests.helpers.softcapped_checks import (
    DTYPES,
    SOFTCAPS,
    assert_bitwise,
    evaluate_triton_with_statistics,
    triton_backward,
    triton_forward,
)


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", ((0, 7), (1, 1), (3, 1), (1, 7), (3, 7)))
def test_bitwise_comparison_handles_strided_and_degenerate_shapes(dtype, shape):
    """Accept equal bytes even when an empty or single-column view has stride two."""
    storage = torch.arange(shape[0] * shape[1] * 2, dtype=dtype).reshape(shape[0], shape[1] * 2)
    actual = storage[:, ::2]
    expected = actual.clone(memory_format=torch.contiguous_format)
    assert actual.stride(-1) == 2
    assert_bitwise(actual, expected)
    if actual.numel():
        expected[0, 0] += 1
        with pytest.raises(AssertionError):
            assert_bitwise(actual, expected)


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", ((), (1,)))
def test_bitwise_comparison_distinguishes_signed_zero(dtype, shape):
    positive = torch.full(shape, 0.0, dtype=dtype)
    negative = torch.full(shape, -0.0, dtype=dtype)
    assert torch.equal(positive, negative)
    assert_bitwise(positive, positive.clone())
    with pytest.raises(AssertionError):
        assert_bitwise(positive, negative)


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab", (1025, 262144))
@pytest.mark.parametrize("forward_impl", ("row", "parallel", "row_pipelined"))
def test_triton_row_invariance(dtype, vocab, forward_impl):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    forward = partial(triton_forward, forward_impl=forward_impl)

    generator = torch.Generator().manual_seed(417)
    logits = (torch.randn((5, vocab), generator=generator) * 30).to(device="cuda", dtype=dtype)
    token_ids = torch.tensor([0, 1, vocab // 2, vocab - 2, vocab - 1], device="cuda")
    together = forward(logits, token_ids)
    separate = torch.cat([forward(logits[i : i + 1], token_ids[i : i + 1]) for i in range(5)])
    assert_bitwise(together, separate)

    permutation = torch.tensor([4, 2, 0, 3, 1], device="cuda")
    reordered = forward(logits[permutation], token_ids[permutation])
    assert_bitwise(reordered, together[permutation])

    logits[0].fill_(3000.0)
    changed = forward(logits, token_ids)
    assert_bitwise(changed[1:], together[1:])


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", DTYPES)
def test_explicit_integer_and_float_softcaps_match(dtype):
    """Explicit 30 and 30.0 describe the same computation and gradient."""
    generator = torch.Generator().manual_seed(437)
    logits = (torch.randn((3, 1025), generator=generator) * 30).to(dtype).requires_grad_(True)
    ids = torch.tensor([0, 512, 1024])
    upstream = torch.tensor([0.25, -2.0, 1.5])
    expected = softcapped_selected_logprob(logits, ids, softcap=30.0)
    expected_grad = torch.autograd.grad(expected, logits, grad_outputs=upstream)[0]
    op = NativeSoftcappedSelectedLogprobOp()
    for kwargs in ({"softcap": 30}, {"softcap": 30.0}):
        output = op(logits, ids, **kwargs)
        gradient = torch.autograd.grad(output, logits, grad_outputs=upstream)[0]
        assert_bitwise(output, expected)
        assert_bitwise(gradient, expected_grad)


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab", (1025, 262144))
@pytest.mark.parametrize("forward_impl", ("row", "parallel", "row_pipelined"))
def test_triton_backward_row_invariance(dtype, vocab, forward_impl):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    backward = partial(triton_backward, forward_impl=forward_impl)

    generator = torch.Generator().manual_seed(420)
    logits = (torch.randn((3, vocab), generator=generator) * 15).to(device="cuda", dtype=dtype)
    token_ids = torch.tensor([0, vocab // 2, vocab - 1], device="cuda")
    upstream = torch.tensor([0.25, -2.0, 1.5], device="cuda")
    together = backward(logits, token_ids, upstream)
    separate = torch.cat(
        [backward(logits[i : i + 1], token_ids[i : i + 1], upstream[i : i + 1]) for i in range(3)]
    )
    assert_bitwise(together, separate)

    permutation = torch.tensor([2, 0, 1], device="cuda")
    reordered = backward(logits[permutation], token_ids[permutation], upstream[permutation])
    assert_bitwise(reordered, together[permutation])

    logits[0].fill_(3000.0)
    changed = backward(logits, token_ids, upstream)
    assert_bitwise(changed[1:], together[1:])


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("forward_impl", ("row", "parallel", "row_pipelined"))
def test_triton_padding_and_strided_layout_invariance(dtype, forward_impl):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
    )

    op = TritonSoftcappedSelectedLogprobOp(forward_impl=SoftcappedLogprobStrategy(forward_impl))
    generator = torch.Generator().manual_seed(422)
    row = (torch.randn((1, 1025), generator=generator) * 15).to(device="cuda", dtype=dtype)
    target = torch.tensor([1024], device="cuda")
    upstream = torch.tensor([-1.75], device="cuda")

    def evaluate(logits, ids, grad_output):
        leaf = logits.detach().requires_grad_(True)
        output = op(leaf, ids, softcap=30.0)
        gradient = torch.autograd.grad(output, leaf, grad_outputs=grad_output)[0]
        return output.detach(), gradient

    expected = evaluate(row, target, upstream)
    storage = torch.zeros((7, 2050), device="cuda", dtype=dtype)
    view = storage[:, ::2]
    view[3:4] = row
    ids = torch.zeros(7, device="cuda", dtype=torch.int64)
    ids[3] = target[0]
    grads = torch.zeros(7, device="cuda")
    grads[3] = upstream[0]
    for _ in range(2):
        actual = evaluate(view, ids, grads)
        assert_bitwise(actual[0][3:4], expected[0])
        assert_bitwise(actual[1][3:4], expected[1])
        storage[:, 1::2].fill_(3000.0)
        view[:3].fill_(-3000.0)


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "vocab",
    (1, 1023, 1024, 1025, 2048, 2049, 3073, 4096, 4097, 5121, 6145, 7169, 262144),
)
@pytest.mark.parametrize("merge_unroll", (1, 3, 4))
def test_triton_forward_variants_match_bitwise(dtype, vocab, merge_unroll):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        _launch_softcapped_selected_logprob_bwd,
        _launch_softcapped_selected_logprob_fwd,
        _launch_softcapped_selected_logprob_fwd_parallel,
    )

    generator = torch.Generator().manual_seed(426)
    logits = (torch.randn((5, vocab), generator=generator) * 30).to(device="cuda", dtype=dtype)
    logits[0].zero_()
    logits[1].fill_(3000.0)
    logits[2].fill_(-3000.0)
    logits[3, vocab - 1] = 3000.0
    ids = torch.tensor([0, vocab - 1, vocab // 2, vocab - 1, 0], device="cuda")
    upstream = torch.tensor([1.0, -2.0, 0.0, -0.75, 0.25], device="cuda")
    expected = _launch_softcapped_selected_logprob_fwd(logits, ids, softcap=30.0)
    actual = _launch_softcapped_selected_logprob_fwd_parallel(
        logits, ids, softcap=30.0, merge_unroll=merge_unroll
    )

    assert_bitwise(actual.selected_logprob, expected.selected_logprob)
    assert_bitwise(actual.row_max, expected.row_max)
    assert_bitwise(actual.log_sum_exp_shifted, expected.log_sum_exp_shifted)
    grad_expected = _launch_softcapped_selected_logprob_bwd(
        logits, ids, upstream, expected.row_max, expected.log_sum_exp_shifted, softcap=30.0
    )
    grad_actual = _launch_softcapped_selected_logprob_bwd(
        logits, ids, upstream, actual.row_max, actual.log_sum_exp_shifted, softcap=30.0
    )
    assert_bitwise(grad_actual, grad_expected)


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape,softcap",
    [
        ((0, 7), 30.0),
        ((4, 262144), 30.0),
        ((16, 49152), 30.0),
        ((64, 32768), 30.0),
        ((256, 65536), 30.0),
        ((8, 202048), 20.0),
        ((2, 2097153), 30.0),
    ],
)
def test_triton_auto_matches_explicit_row_and_inference(dtype, shape, softcap):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
    )

    generator = torch.Generator().manual_seed(428)
    logits = (torch.randn(shape, generator=generator) * 30).to(device="cuda", dtype=dtype)
    ids = torch.arange(shape[0], device="cuda", dtype=torch.int64) * 1025 % shape[1]
    upstream = torch.randn(shape[0], generator=generator).to(device="cuda")
    op = partial(TritonSoftcappedSelectedLogprobOp(forward_impl=None), softcap=softcap)
    reference = evaluate_triton_with_statistics(
        partial(
            TritonSoftcappedSelectedLogprobOp(forward_impl=SoftcappedLogprobStrategy.ROW),
            softcap=softcap,
        ),
        logits,
        ids,
        upstream,
    )
    actual = evaluate_triton_with_statistics(op, logits, ids, upstream)
    for result, expected in zip(actual, reference, strict=True):
        assert_bitwise(result, expected)
    with torch.no_grad():
        assert_bitwise(op(logits, ids), reference[0])
    with torch.inference_mode():
        assert_bitwise(op(logits, ids), reference[0])


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("softcap", SOFTCAPS)
@pytest.mark.parametrize("vocab", (1025, 262145))
def test_all_strategies_preserve_output_statistics_and_all_gradients(dtype, softcap, vocab):
    """Every selectable strategy agrees, including wide tails and nondefault caps."""
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
    )

    generator = torch.Generator().manual_seed(464)
    logits = (30 * torch.randn((3, vocab), generator=generator)).to(device="cuda", dtype=dtype)
    ids = torch.tensor([0, vocab // 2, vocab - 1], device="cuda")
    upstream = torch.tensor([0.5, -2.0, 0.0], device="cuda")
    expected = None
    for strategy in (*SoftcappedLogprobStrategy, None):
        op = partial(TritonSoftcappedSelectedLogprobOp(forward_impl=strategy), softcap=softcap)
        actual = evaluate_triton_with_statistics(op, logits, ids, upstream)
        if expected is None:
            expected = actual
        else:
            for result, reference in zip(actual, expected, strict=True):
                assert_bitwise(result, reference)
        for context in (torch.no_grad(), torch.inference_mode()):
            with context:
                assert_bitwise(op(logits, ids), expected[0])


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("automatic", (False, True))
def test_pipeline_depth_changes_across_batch_split_preserve_bits(dtype, automatic, monkeypatch):
    """Synthetic range rules switch depth; real kernels must retain ROW's results."""
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob import softcapped_selected_logprob as core

    whole_key = core.ForwardConfigKey(
        device=core._DEFAULT_DEVICE,
        dtype=dtype,
        min_rows=3,
        max_rows=4,
        min_tiles=2,
        max_tiles=2,
    )
    split_key = core.ForwardConfigKey(
        device=core._DEFAULT_DEVICE,
        dtype=dtype,
        min_rows=1,
        max_rows=2,
        min_tiles=2,
        max_tiles=2,
    )
    monkeypatch.setattr(
        core,
        "ROW_PIPELINE_CONFIGS",
        {
            whole_key: core.RowPipelineConfig(pipeline_stages=4),
            split_key: core.RowPipelineConfig(pipeline_stages=2),
        },
    )
    monkeypatch.setattr(
        core,
        "FORWARD_STRATEGY_CONFIGS",
        dict.fromkeys((whole_key, split_key), core.SoftcappedLogprobStrategy.ROW_PIPELINED),
    )
    core.select_softcapped_logprob_strategy.cache_clear()
    core.select_softcapped_logprob_pipeline_config.cache_clear()
    try:
        generator = torch.Generator().manual_seed(431)
        logits = (torch.randn((4, 1025), generator=generator) * 30).to(device="cuda", dtype=dtype)
        ids = torch.tensor([0, 1023, 1024, 512], device="cuda")
        upstream = torch.tensor([0.25, -2.0, 0.0, 1.5], device="cuda")
        strategy = None if automatic else core.SoftcappedLogprobStrategy.ROW_PIPELINED
        op = partial(core.TritonSoftcappedSelectedLogprobOp(forward_impl=strategy), softcap=30.0)
        row_op = partial(
            core.TritonSoftcappedSelectedLogprobOp(forward_impl=core.SoftcappedLogprobStrategy.ROW),
            softcap=30.0,
        )
        device = core.softcapped_logprob_device_key(logits.device)
        assert (
            core.select_softcapped_logprob_pipeline_config(device, dtype, 4, 1025).pipeline_stages
            == 4
        )
        assert (
            core.select_softcapped_logprob_pipeline_config(device, dtype, 2, 1025).pipeline_stages
            == 2
        )
        together = evaluate_triton_with_statistics(op, logits, ids, upstream)
        reference = evaluate_triton_with_statistics(row_op, logits, ids, upstream)
        for actual, expected in zip(together, reference, strict=True):
            assert_bitwise(actual, expected)
        halves = [
            evaluate_triton_with_statistics(
                op, logits[i : i + 2], ids[i : i + 2], upstream[i : i + 2]
            )
            for i in (0, 2)
        ]
        for expected, *parts in zip(together, *halves, strict=True):
            assert_bitwise(torch.cat(parts), expected)
        for context in (torch.no_grad(), torch.inference_mode()):
            with context:
                assert_bitwise(op(logits, ids), together[0])
                assert_bitwise(
                    torch.cat([op(logits[i : i + 2], ids[i : i + 2]) for i in (0, 2)]), together[0]
                )
    finally:
        core.select_softcapped_logprob_strategy.cache_clear()
        core.select_softcapped_logprob_pipeline_config.cache_clear()


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_auto_batch_split_switches_strategy_without_changing_bits(
    dtype, monkeypatch, request
):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob import softcapped_selected_logprob as config
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
        select_softcapped_logprob_strategy,
        softcapped_logprob_device_key,
    )

    # Exercise an automatic switch with a small batch, independently of the
    # production boundaries covered by the dedicated tests below.
    select_softcapped_logprob_strategy.cache_clear()
    request.addfinalizer(select_softcapped_logprob_strategy.cache_clear)
    monkeypatch.setattr(
        config,
        "FORWARD_STRATEGY_CONFIGS",
        {
            config.ForwardConfigKey(
                device=config._DEFAULT_DEVICE,
                dtype=dtype,
                min_rows=3,
                max_rows=1024,
                min_tiles=256,
                max_tiles=256,
            ): SoftcappedLogprobStrategy.PARALLEL,
        },
    )
    generator = torch.Generator().manual_seed(429)
    logits = (torch.randn((4, 262144), generator=generator) * 30).to(device="cuda", dtype=dtype)
    ids = torch.tensor([0, 1023, 262143, 1024], device="cuda")
    upstream = torch.tensor([0.25, -2.0, 0.0, 1.5], device="cuda")
    key = softcapped_logprob_device_key(logits.device)
    assert (
        select_softcapped_logprob_strategy(key, dtype, 4, 262144)
        is SoftcappedLogprobStrategy.PARALLEL
    )
    assert (
        select_softcapped_logprob_strategy(key, dtype, 2, 262144) is SoftcappedLogprobStrategy.ROW
    )
    op = partial(TritonSoftcappedSelectedLogprobOp(), softcap=30.0)
    together = evaluate_triton_with_statistics(op, logits, ids, upstream)
    halves = [
        evaluate_triton_with_statistics(op, logits[i : i + 2], ids[i : i + 2], upstream[i : i + 2])
        for i in (0, 2)
    ]
    for expected, *parts in zip(together, *halves, strict=True):
        assert_bitwise(torch.cat(parts), expected)
    permutation = torch.tensor([3, 1, 0, 2], device="cuda")
    reordered = evaluate_triton_with_statistics(
        op, logits[permutation], ids[permutation], upstream[permutation]
    )
    for result, expected in zip(reordered, together, strict=True):
        assert_bitwise(result, expected[permutation])
    storage = torch.zeros((4, 524288), device="cuda", dtype=dtype)
    view = storage[:, ::2]
    view.copy_(logits)
    for result, expected in zip(
        evaluate_triton_with_statistics(op, view, ids, upstream), together, strict=True
    ):
        assert_bitwise(result, expected)


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab", (31745, 32768, 32769))
def test_triton_production_range_boundary_preserves_output_statistics_and_gradients(dtype, vocab):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
        select_softcapped_logprob_strategy,
        softcapped_logprob_device_key,
    )

    max_rows = 4096
    # Allocate on the GPU directly: inputs use at most about 513 MiB here.
    # Compare each chunk in place rather than concatenating full-sized gradients.
    generator = torch.Generator(device="cuda").manual_seed(431)
    n_rows = max_rows + 1
    logits = torch.randn((n_rows, vocab), generator=generator, device="cuda", dtype=dtype)
    logits.mul_(15.0)
    ids = torch.arange(n_rows, device="cuda", dtype=torch.int64) * 1025 % vocab
    ids[0], ids[-1] = 0, vocab - 1
    upstream = torch.randn(n_rows, generator=generator, device="cuda")
    key = softcapped_logprob_device_key(logits.device)
    assert (
        select_softcapped_logprob_strategy(key, dtype, max_rows, vocab)
        is SoftcappedLogprobStrategy.PARALLEL
    )
    assert (
        select_softcapped_logprob_strategy(key, dtype, n_rows, vocab)
        is SoftcappedLogprobStrategy.ROW
    )
    reference = evaluate_triton_with_statistics(
        partial(
            TritonSoftcappedSelectedLogprobOp(forward_impl=SoftcappedLogprobStrategy.ROW),
            softcap=30.0,
        ),
        logits,
        ids,
        upstream,
    )
    auto = partial(TritonSoftcappedSelectedLogprobOp(), softcap=30.0)
    together = evaluate_triton_with_statistics(auto, logits, ids, upstream)
    for result, expected in zip(together, reference, strict=True):
        assert_bitwise(result, expected)
    del together, result, expected
    # The whole batch selects ROW; both chunks select PARALLEL. Output, saved
    # statistics and gradients must keep the same bytes across this real policy switch.
    for start, stop in ((0, max_rows), (max_rows, n_rows)):
        actual = evaluate_triton_with_statistics(
            auto, logits[start:stop], ids[start:stop], upstream[start:stop]
        )
        for result, expected in zip(actual, reference, strict=True):
            assert_bitwise(result, expected[start:stop])
        del actual, result, expected
