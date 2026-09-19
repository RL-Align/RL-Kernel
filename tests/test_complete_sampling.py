# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.integrations.sampling import sampling_keep_mask, vocab_parallel_sampling_keep_mask


def test_nucleus_is_not_truncated_to_api_logprob_limit():
    logits = torch.zeros(2, 1024)
    keep = sampling_keep_mask(logits, temperature=0.7, top_p=0.95)
    assert (keep.sum(dim=1) > 900).all()
    assert torch.equal(keep[0], keep[1])


def test_per_request_temperature_and_top_p():
    logits = torch.arange(100, dtype=torch.float32).repeat(3, 1)
    temperature = torch.tensor([0.7, 1.0, 2.0])
    p = torch.tensor([0.8, 0.95, 0.99])
    batched = sampling_keep_mask(logits, temperature=temperature, top_p=p)
    for i in range(3):
        assert torch.equal(
            batched[i : i + 1],
            sampling_keep_mask(logits[i : i + 1], temperature=temperature[i], top_p=p[i]),
        )


def test_top_k_preserves_boundary_ties():
    keep = sampling_keep_mask(torch.tensor([[0.0, 1.0, 1.0, 2.0]]), top_k=2)
    assert keep.tolist() == [[False, True, True, True]]


@pytest.mark.parametrize("top_p,top_k", [(None, None), (0.9, None), (None, 8), (0.95, 20)])
def test_chunking_and_padding_do_not_change_support(top_p, top_k):
    torch.manual_seed(19)
    logits = torch.randn(7, 128, dtype=torch.bfloat16)
    logits[:, 100:] = 1000  # Padding must not affect the nucleus.
    active = torch.tensor([True, False, True, False, True, True, False])
    actual = vocab_parallel_sampling_keep_mask(
        logits,
        real_vocab_size=100,
        temperature=0.7,
        top_p=top_p,
        top_k=top_k,
        active_rows=active,
        chunk_size=2,
    )
    expected = sampling_keep_mask(logits[active, :100], temperature=0.7, top_p=top_p, top_k=top_k)
    assert torch.equal(actual[active, :100], expected)
    assert not actual[active, 100:].any()
    assert actual[~active].all()


def test_sampling_support_has_no_gradient_and_preserves_input():
    x = torch.randn(3, 80, requires_grad=True)
    original = x.detach().clone()
    mask = sampling_keep_mask(x, top_p=0.9, top_k=30)
    assert not mask.requires_grad
    assert torch.equal(x, original)
    loss = x.masked_fill(~mask, float("-inf")).log_softmax(-1)[mask].sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_greedy_requests_ignore_truncation_without_dividing_by_zero():
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    mask = sampling_keep_mask(logits, temperature=torch.tensor([0.0, 0.7]), top_p=0.5, top_k=1)
    assert mask.tolist() == [[True, True, True], [False, False, True]]
