# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.testing import (
    active_token_count,
    compute_policy_ratio,
    compute_reference_kl,
    masked_mean,
    masked_sum,
    selected_logprobs_reference,
    summarize_kernel_drift,
    teacher_forced_logprobs_reference,
)


def test_selected_logprobs_reference_matches_pytorch():
    logits = torch.tensor([[[1.0, 2.0, 3.0], [0.5, -0.5, 1.5]]])
    token_ids = torch.tensor([[2, 0]])

    actual = selected_logprobs_reference(logits, token_ids)
    expected = (
        torch.log_softmax(logits.float(), dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    )

    assert torch.allclose(actual, expected)
    assert actual.dtype == torch.float32


def test_selected_logprobs_reference_mask_and_dtype():
    logits = torch.randn(2, 3, 5)
    token_ids = torch.tensor([[0, 1, 2], [3, 4, 0]])
    mask = torch.tensor([[True, False, True], [False, True, True]])

    actual = selected_logprobs_reference(logits, token_ids, mask=mask, output_dtype=torch.float16)

    assert actual.dtype == torch.float16
    assert torch.equal(actual[~mask], torch.zeros_like(actual[~mask]))


def test_selected_logprobs_reference_allows_ignore_index_when_masked_out():
    logits = torch.randn(1, 3, 6)
    token_ids = torch.tensor([[1, -100, 2]])
    mask = torch.tensor([[True, False, True]])

    actual = selected_logprobs_reference(logits, token_ids, mask=mask)
    safe_token_ids = token_ids.masked_fill(~mask, 0)
    expected = (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(-1, safe_token_ids.unsqueeze(-1))
        .squeeze(-1)
        .masked_fill(~mask, 0.0)
    )

    assert torch.allclose(actual, expected)
    assert actual[0, 1] == 0.0


def test_selected_logprobs_reference_temperature():
    logits = torch.tensor([[1.0, 3.0, -1.0]])
    token_ids = torch.tensor([1])

    actual = selected_logprobs_reference(logits, token_ids, temperature=0.5)
    expected = (
        torch.log_softmax(logits.float() / 0.5, dim=-1)
        .gather(-1, token_ids.unsqueeze(-1))
        .squeeze(-1)
    )

    assert torch.allclose(actual, expected)


def test_selected_logprobs_reference_rejects_bad_temperature():
    with pytest.raises(ValueError, match="temperature"):
        selected_logprobs_reference(torch.randn(1, 3), torch.tensor([0]), temperature=0.0)


def test_teacher_forced_logprobs_match_unit_temperature_reference():
    logits = torch.tensor([[[2.0, -1.0, 0.5], [0.25, 1.25, -0.5]]])
    token_ids = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, False]])

    actual = teacher_forced_logprobs_reference(logits, token_ids, mask=mask)
    expected = selected_logprobs_reference(logits, token_ids, mask=mask, temperature=1.0)

    assert torch.allclose(actual, expected)
    assert actual[0, 1] == 0.0


def test_teacher_forced_logprobs_do_not_apply_sampling_temperature():
    logits = torch.tensor([[[4.0, 1.0, -1.0], [0.1, 2.0, -0.5]]])
    token_ids = torch.tensor([[0, 1]])

    actual = teacher_forced_logprobs_reference(logits, token_ids)
    wrongly_temperature_scaled = selected_logprobs_reference(logits, token_ids, temperature=0.25)

    assert not torch.allclose(actual, wrongly_temperature_scaled)


def test_teacher_forced_logprobs_preserve_dtype_contract():
    logits = torch.randn(2, 3, 5)
    token_ids = torch.tensor([[0, 1, 2], [3, 4, 0]])

    actual = teacher_forced_logprobs_reference(
        logits,
        token_ids,
        output_dtype=torch.float16,
    )

    assert actual.dtype == torch.float16


def test_masked_reductions_ignore_inactive_tokens():
    values = torch.tensor([[1.0, 100.0], [3.0, 4.0]])
    mask = torch.tensor([[True, False], [True, True]])

    assert torch.equal(active_token_count(mask), torch.tensor(3.0))
    assert torch.equal(masked_sum(values, mask), torch.tensor(8.0))
    assert torch.allclose(masked_mean(values, mask), torch.tensor(8.0 / 3.0))


def test_ratio_kl_and_drift_helpers():
    current = torch.tensor([[0.0, -1.0], [-2.0, -3.0]])
    old = torch.tensor([[-0.5, -1.5], [-2.5, -3.5]])
    ref = torch.tensor([[-0.25, -1.25], [-2.25, -3.25]])
    mask = torch.tensor([[True, False], [True, True]])

    ratio = compute_policy_ratio(current, old, mask)
    kl = compute_reference_kl(current, ref, mask)

    assert torch.equal(ratio[~mask], torch.zeros_like(ratio[~mask]))
    assert torch.equal(kl[~mask], torch.zeros_like(kl[~mask]))

    summary = summarize_kernel_drift(current + 0.1, current, mask)

    assert summary["active_count"] == 3
    assert summary["max_abs_error"] == pytest.approx(0.1, rel=1e-6)
    assert summary["mean_abs_error"] == pytest.approx(0.1, rel=1e-6)
