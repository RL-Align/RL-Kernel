# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""GPU launch safety: input device, scratch bounds, tile coverage and 64-bit offsets."""

import math

import pytest
import torch

from rl_engine.contracts.operators.softcapped_selected_logprob import REDUCTION_CONFIG
from tests.helpers.softcapped_checks import (
    DTYPES,
    INDEPENDENT_CHECKS,
    assert_backward,
    assert_bitwise,
    assert_fp32_accuracy,
    fp32_backward_reference,
    fp32_reference,
)


@pytest.mark.parametrize("check", INDEPENDENT_CHECKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab", (1025, 262144))
@pytest.mark.parametrize("forward_impl", (None, "row", "parallel", "row_pipelined"))
@pytest.mark.parametrize("softcap", (30.0, 50.0))
def test_triton_noncurrent_device_forward_backward(dtype, vocab, forward_impl, softcap, check):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA or ROCm GPUs are required for the noncurrent-device regression")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
        _launch_softcapped_selected_logprob_bwd,
    )

    device = torch.device("cuda:1")
    generator = torch.Generator().manual_seed(435)
    # Include copies of strided input tensors while cuda:0 is current.
    storage = (torch.randn((3, 2 * vocab), generator=generator) * 15).to(device=device, dtype=dtype)
    logits = storage[:, ::2].detach().requires_grad_(True)
    token_ids = torch.tensor([0, 0, vocab // 2, 0, vocab - 1, 0], device=device)[::2]
    upstream = torch.tensor([0.25, 0, -2.0, 0, 1.5, 0], device=device)[::2]
    strategy = None if forward_impl is None else SoftcappedLogprobStrategy(forward_impl)
    op = TritonSoftcappedSelectedLogprobOp(forward_impl=strategy)

    with torch.cuda.device(0):
        output = op(logits, token_ids, softcap=softcap)
        assert torch.cuda.current_device() == 0
        assert output.device == device and output.dtype == torch.float32
        saved_logits, saved_ids, row_max, log_sum_exp_shifted = output.grad_fn.saved_tensors
        assert all(
            value.device == device
            for value in (saved_logits, saved_ids, row_max, log_sum_exp_shifted)
        )

        # Call backward directly too: autograd may itself select the forward
        # device, which could otherwise hide a missing guard in the launcher.
        direct_gradient = _launch_softcapped_selected_logprob_bwd(
            logits, token_ids, upstream, row_max, log_sum_exp_shifted, softcap
        )
        assert torch.cuda.current_device() == 0
        (gradient,) = torch.autograd.grad(output, logits, grad_outputs=upstream)
        assert torch.cuda.current_device() == 0
        assert gradient.device == direct_gradient.device == device
        assert gradient.dtype == direct_gradient.dtype == dtype
        if check == "consistency":
            assert_bitwise(gradient, direct_gradient)
        else:
            assert_fp32_accuracy(
                output, fp32_reference(logits, token_ids, softcap), "forward_accuracy"
            )
            assert_fp32_accuracy(
                gradient,
                fp32_backward_reference(logits, token_ids, upstream, softcap=softcap),
                "gradient_accuracy",
            )

        if check == "consistency":
            for context in (torch.no_grad(), torch.inference_mode()):
                with context:
                    inference_output = op(logits, token_ids, softcap=softcap)
                assert torch.cuda.current_device() == 0
                assert inference_output.device == device
                assert_bitwise(inference_output, output.detach())


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab", (2048, 2049, 3073, 4097, 5121, 6145))
@pytest.mark.parametrize("merge_unroll", (1, 3, 4))
def test_triton_parallel_forward_scratch_and_output_bounds(dtype, vocab, merge_unroll):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    import triton

    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        _launch_softcapped_selected_logprob_fwd,
        _math_launch_options,
        _softcapped_selected_logprob_fwd_max_merge_kernel,
        _softcapped_selected_logprob_fwd_max_partial_kernel,
        _softcapped_selected_logprob_fwd_merge_kernel,
        _softcapped_selected_logprob_fwd_partial_kernel,
    )

    n_rows = 4
    n_tiles = triton.cdiv(vocab, REDUCTION_CONFIG.tile_size)
    generator = torch.Generator().manual_seed(427)
    logits = (torch.randn((n_rows, vocab), generator=generator) * 15).to(device="cuda", dtype=dtype)
    math_options = _math_launch_options(logits.device)
    # Include invalid IDs to verify the merge masks selected-logit reads.
    ids = torch.tensor([-1, 1024, vocab - 1, vocab], device="cuda")

    def guarded(shape):
        storage = torch.full((math.prod(shape) + 2,), -123.0, device="cuda")
        view = storage[1:-1].view(shape)
        view.fill_(float("nan"))
        return storage, view

    scratch_storage, scratch = guarded((n_rows, n_tiles))
    output_storage, output = guarded((n_rows,))
    max_storage, row_max = guarded((n_rows,))
    lse_storage, log_sum_exp_shifted = guarded((n_rows,))
    _softcapped_selected_logprob_fwd_max_partial_kernel[(n_rows, n_tiles)](
        logits,
        scratch,
        vocab,
        n_tiles,
        REDUCTION_CONFIG.tile_size,
        num_warps=4,
        **math_options,
    )
    max_scratch_before = scratch.clone()
    for tile in range(n_tiles):
        begin = tile * REDUCTION_CONFIG.tile_size
        expected_tile_max = logits[:, begin : begin + REDUCTION_CONFIG.tile_size].float().amax(-1)
        assert_bitwise(scratch[:, tile], expected_tile_max, "raw-logit scratch maximum")
    _softcapped_selected_logprob_fwd_max_merge_kernel[(n_rows,)](
        scratch, row_max, n_tiles, SOFTCAP=30.0, num_warps=4, **math_options
    )
    assert_bitwise(scratch, max_scratch_before)
    row_max_before = row_max.clone()
    # The sum stage overwrites every entry; no maximum may remain in the scratch.
    scratch.fill_(float("nan"))
    _softcapped_selected_logprob_fwd_partial_kernel[(n_rows, n_tiles)](
        logits,
        row_max,
        scratch,
        vocab,
        n_tiles,
        REDUCTION_CONFIG.tile_size,
        SOFTCAP=30.0,
        MAX_GROUP_SIZE=REDUCTION_CONFIG.group_size,
        SECTION_COUNT=REDUCTION_CONFIG.section_count,
        num_warps=4,
        **math_options,
    )
    scratch_before = scratch.clone()
    _softcapped_selected_logprob_fwd_merge_kernel[(n_rows,)](
        logits,
        ids,
        row_max,
        scratch,
        output,
        log_sum_exp_shifted,
        vocab,
        n_tiles,
        SOFTCAP=30.0,
        MERGE_UNROLL=merge_unroll,
        num_warps=4,
        **math_options,
    )
    expected = _launch_softcapped_selected_logprob_fwd(logits, ids, softcap=30.0)
    assert torch.isfinite(scratch).all()
    assert_bitwise(scratch, scratch_before)
    assert_bitwise(row_max, expected.row_max)
    assert_bitwise(row_max, row_max_before)
    assert_bitwise(log_sum_exp_shifted, expected.log_sum_exp_shifted)
    assert_bitwise(output[1:3], expected.selected_logprob[1:3])
    assert torch.isnan(output[[0, 3]]).all()
    for storage in (scratch_storage, output_storage, max_storage, lse_storage):
        assert torch.equal(storage[[0, -1]], torch.full((2,), -123.0, device="cuda"))


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab", (2048, 2049))
def test_triton_backward_tile_coverage_and_output_bounds(dtype, vocab):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    import triton

    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        _launch_softcapped_selected_logprob_fwd,
        _math_launch_options,
        _softcapped_selected_logprob_bwd_kernel,
    )

    n_rows = 4
    generator = torch.Generator().manual_seed(425)
    logits = (torch.randn((n_rows, vocab), generator=generator) * 15).to(device="cuda", dtype=dtype)
    # Selected tokens exercise both sides of a tile boundary and the last lane.
    ids = torch.tensor([0, 1023, 1024, vocab - 1], device="cuda")
    upstream = torch.tensor([0.25, -2.0, 1.5, -0.75], device="cuda")
    storage = torch.full(
        (n_rows * vocab + 2 * REDUCTION_CONFIG.tile_size,), -123.0, device="cuda", dtype=dtype
    )
    grad_logits = storage[REDUCTION_CONFIG.tile_size : -REDUCTION_CONFIG.tile_size].view(
        n_rows, vocab
    )
    grad_logits.fill_(float("nan"))

    def backward_into_guarded_output(x, token_ids, grad_output):
        result = _launch_softcapped_selected_logprob_fwd(x, token_ids, softcap=30.0)
        _softcapped_selected_logprob_bwd_kernel[
            (n_rows, triton.cdiv(vocab, REDUCTION_CONFIG.tile_size))
        ](
            x,
            token_ids,
            grad_output,
            result.row_max,
            result.log_sum_exp_shifted,
            grad_logits,
            vocab,
            BLOCK_V=REDUCTION_CONFIG.tile_size,
            SOFTCAP=30.0,
            num_warps=4,
            **_math_launch_options(x.device),
        )
        return grad_logits

    assert_backward(backward_into_guarded_output, logits, ids, upstream)
    # NaN initialization detects missing tiles; guards detect out-of-bounds stores.
    assert torch.isfinite(grad_logits).all()
    for guard in (storage[: REDUCTION_CONFIG.tile_size], storage[-REDUCTION_CONFIG.tile_size :]):
        assert torch.equal(guard, torch.full_like(guard, -123.0))


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_row_offsets_across_int32_boundary(dtype):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    import triton
    import triton.language as tl

    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        _math_launch_options,
        _softcapped_selected_logprob_bwd_kernel,
        _softcapped_selected_logprob_fwd_kernel,
    )

    @triton.jit
    def run_last_rows(
        x,
        ids,
        upstream,
        output,
        row_max,
        log_sum_exp_shifted,
        vocab: tl.constexpr,
        START_ROW: tl.constexpr,
        BACKWARD: tl.constexpr,
        BLOCK: tl.constexpr,
        MAX_GROUP_SIZE: tl.constexpr,
        SECTION_COUNT: tl.constexpr,
    ):
        if tl.program_id(0) >= START_ROW:
            base = tl.full((), START_ROW, tl.int64) * vocab
            if BACKWARD:
                _softcapped_selected_logprob_bwd_kernel(
                    x - base,
                    ids - START_ROW,
                    upstream - START_ROW,
                    row_max - START_ROW,
                    log_sum_exp_shifted - START_ROW,
                    output - base,
                    vocab,
                    BLOCK,
                    SOFTCAP=30.0,
                )
            else:
                _softcapped_selected_logprob_fwd_kernel(
                    x - base,
                    ids - START_ROW,
                    output - START_ROW,
                    row_max - START_ROW,
                    log_sum_exp_shifted - START_ROW,
                    vocab,
                    BLOCK,
                    SOFTCAP=30.0,
                    MAX_GROUP_SIZE=MAX_GROUP_SIZE,
                    SECTION_COUNT=SECTION_COUNT,
                )

    vocab = 65537
    start_row = 2**31 // vocab - 1
    generator = torch.Generator().manual_seed(423)
    x = (torch.randn((3, vocab), generator=generator) * 15).to(device="cuda", dtype=dtype)
    ids = torch.tensor([0, 1024, vocab - 1], device="cuda")
    upstream = torch.tensor([0.25, -2.0, 1.5], device="cuda")
    row_max = torch.full((4,), -123.0, device="cuda", dtype=torch.float32)
    log_sum_exp_shifted = torch.full((4,), -123.0, device="cuda", dtype=torch.float32)
    for backward in (False, True):
        shape = (4, vocab) if backward else (4,)
        output = torch.full(
            shape, -123.0, device="cuda", dtype=dtype if backward else torch.float32
        )
        grid = (start_row + 3, triton.cdiv(vocab, REDUCTION_CONFIG.tile_size) if backward else 1)
        run_last_rows[grid](
            x,
            ids,
            upstream,
            output,
            row_max,
            log_sum_exp_shifted,
            vocab,
            START_ROW=start_row,
            BACKWARD=backward,
            BLOCK=REDUCTION_CONFIG.tile_size,
            MAX_GROUP_SIZE=REDUCTION_CONFIG.group_size,
            SECTION_COUNT=REDUCTION_CONFIG.section_count,
            num_warps=4,
            **_math_launch_options(x.device),
        )
        expected = fp32_backward_reference(x, ids, upstream) if backward else fp32_reference(x, ids)
        assert_fp32_accuracy(
            output[:3],
            expected,
            "gradient_accuracy" if backward else "forward_accuracy",
        )
        assert torch.equal(output[3:], torch.full_like(output[3:], -123.0))
        for statistic in (row_max, log_sum_exp_shifted):
            assert torch.equal(statistic[3:], torch.full_like(statistic[3:], -123.0))
