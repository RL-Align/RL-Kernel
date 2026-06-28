# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.testing import (
    compare_selected_logprob_layouts,
    make_padded_batch_layout,
    selected_logprobs_reference,
    summarize_kernel_drift,
)


def _case(*, device="cpu", dtype=torch.float32):
    generator = torch.Generator(device=device).manual_seed(123)
    logits = torch.randn(4, 5, 17, device=device, dtype=dtype, generator=generator)
    token_ids = torch.randint(0, 17, (4, 5), device=device, generator=generator)
    mask = torch.tensor(
        [
            [True, True, False, True, False],
            [True, False, True, True, True],
            [False, True, True, False, True],
            [True, True, True, False, False],
        ],
        device=device,
        dtype=torch.bool,
    )
    return logits, token_ids, mask


def test_selected_logprob_is_invariant_to_batch_position():
    logits, token_ids, mask = _case()
    row_order = torch.tensor([2, 0, 3, 1])

    base = selected_logprobs_reference(logits, token_ids, mask=mask)
    shuffled = selected_logprobs_reference(
        logits[row_order],
        token_ids[row_order],
        mask=mask[row_order],
    )
    restored = torch.empty_like(base)
    restored[row_order] = shuffled

    summary = summarize_kernel_drift(restored, base, mask)
    assert summary["active_count"] == int(mask.sum().item())
    assert summary["max_abs_error"] == 0.0
    assert summary["mean_abs_error"] == 0.0


def test_selected_logprob_is_invariant_to_padding_layout():
    logits, token_ids, mask = _case()
    destination_rows = torch.tensor([4, 0, 2, 5])

    padded_logits, padded_token_ids, padded_mask = make_padded_batch_layout(
        logits,
        token_ids,
        mask,
        destination_rows=destination_rows,
        padded_batch_size=6,
    )

    summary = compare_selected_logprob_layouts(
        logits,
        token_ids,
        mask,
        padded_logits,
        padded_token_ids,
        padded_mask,
        candidate_rows=destination_rows,
    )

    assert summary["active_count"] == int(mask.sum().item())
    assert summary["max_abs_error"] == 0.0
    assert summary["mean_abs_error"] == 0.0


def test_make_padded_batch_layout_rejects_out_of_range_pad_token_id():
    logits, token_ids, mask = _case()

    with pytest.raises(ValueError, match="pad_token_id"):
        make_padded_batch_layout(
            logits,
            token_ids,
            mask,
            destination_rows=torch.tensor([0, 1, 2, 3]),
            padded_batch_size=4,
            pad_token_id=logits.shape[-1],
        )


@pytest.mark.parametrize(
    "candidate_rows",
    [
        torch.tensor([0, 1, 1, 3]),
        torch.tensor([0, -1, 2, 3]),
        torch.tensor([0, 1, 2, 6]),
        torch.tensor([0, 1, 2]),
    ],
)
def test_selected_logprob_layout_compare_rejects_bad_candidate_rows(candidate_rows):
    logits, token_ids, mask = _case()
    padded_logits, padded_token_ids, padded_mask = make_padded_batch_layout(
        logits,
        token_ids,
        mask,
        destination_rows=torch.tensor([0, 1, 2, 3]),
        padded_batch_size=4,
    )

    with pytest.raises(ValueError):
        compare_selected_logprob_layouts(
            logits,
            token_ids,
            mask,
            padded_logits,
            padded_token_ids,
            padded_mask,
            candidate_rows=candidate_rows,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_selected_logprob_padding_layout_cuda_dtype_sweep(dtype):
    logits, token_ids, mask = _case(device="cuda", dtype=dtype)
    destination_rows = torch.tensor([1, 4, 0, 3], device="cuda")

    padded_logits, padded_token_ids, padded_mask = make_padded_batch_layout(
        logits,
        token_ids,
        mask,
        destination_rows=destination_rows,
        padded_batch_size=5,
    )

    summary = compare_selected_logprob_layouts(
        logits,
        token_ids,
        mask,
        padded_logits,
        padded_token_ids,
        padded_mask,
        candidate_rows=destination_rows,
        output_dtype=torch.float32,
    )

    assert summary["active_count"] == int(mask.sum().item())
    assert summary["max_abs_error"] == 0.0
    assert summary["mean_abs_error"] == 0.0
