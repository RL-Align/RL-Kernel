# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import math

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


def _reference_row_support(logits, temperature, top_p, top_k):
    """Independent scalar reference for the ascending top-k/top-p contract."""
    values = logits.tolist()
    if temperature < 1e-5 or (top_p is None and top_k is None):
        return [math.isfinite(value) for value in values]
    values = [value / temperature for value in values]
    ordered = sorted(range(len(values)), key=values.__getitem__)
    if top_k is not None and top_k > 0:
        threshold = values[ordered[-min(top_k, len(values))]]
        ordered = [index for index in ordered if values[index] >= threshold]
    keep = [False] * len(values)
    weights = [math.exp(values[index] - values[ordered[-1]]) for index in ordered]
    total = sum(weights)
    cumulative = 0.0
    for index, weight in zip(ordered, weights, strict=True):
        cumulative += weight / total
        keep[index] = top_p is None or cumulative > 1 - top_p
    keep[ordered[-1]] = True
    return keep


def _assert_wrapper_support(parameters, active_mode, chunk_size):
    # Repeated logits make differences in per-original-row parameters observable.
    logits = torch.arange(13, dtype=torch.float32).repeat(7, 1) / 4
    logits[:, 11:] = 1000  # Padding must not dominate global sorting.
    logits.requires_grad_()
    original = logits.detach().clone()
    active = {
        "all": None,
        "subset": torch.tensor([True, False, True, True, False, True, True]),
        "none": torch.zeros(7, dtype=torch.bool),
    }[active_mode]
    expected = torch.ones_like(logits, dtype=torch.bool)
    for row in range(logits.size(0)):
        if active is not None and not active[row]:
            continue
        row_parameters = {}
        for name, value in parameters.items():
            if value is None:
                row_parameters[name] = None
            else:
                tensor = torch.as_tensor(value).reshape(-1)
                row_parameters[name] = tensor[0 if tensor.numel() == 1 else row].item()
        expected[row, :11] = torch.tensor(
            _reference_row_support(logits[row, :11], **row_parameters)
        )
        expected[row, 11:] = False
    actual = vocab_parallel_sampling_keep_mask(
        logits,
        real_vocab_size=11,
        active_rows=active,
        chunk_size=chunk_size,
        **parameters,
    )
    assert torch.equal(actual, expected)
    assert torch.equal(logits, original)
    assert not actual.requires_grad


@pytest.mark.parametrize("chunk_size", [1, 3, 32])
@pytest.mark.parametrize("active_mode", ["all", "subset", "none"])
@pytest.mark.parametrize("per_row", ["temperature", "top_p", "top_k", "all"])
def test_chunked_per_row_parameters_follow_original_rows(per_row, active_mode, chunk_size):
    parameters = {"temperature": 0.7, "top_p": 0.9, "top_k": 4}
    vectors = {
        "temperature": torch.tensor([0.0, 0.35, 1.7, 0.0, 2.2, 0.55, 1.1]),
        "top_p": torch.tensor([0.1, 0.95, 0.75, 0.25, 0.9, 0.6, 1.0]),
        "top_k": torch.tensor([1, 6, 4, 2, 0, -1, 5]),
    }
    parameters.update(vectors if per_row == "all" else {per_row: vectors[per_row]})
    _assert_wrapper_support(parameters, active_mode, chunk_size)


@pytest.mark.parametrize("parameter_type", [list, tuple, "column"])
def test_chunked_per_row_parameters_accept_array_like_values(parameter_type):
    parameters = {
        "temperature": [0.0, 0.35, 1.7, 0.0, 2.2, 0.55, 1.1],
        "top_p": [0.1, 0.95, 0.75, 0.25, 0.9, 0.6, 1.0],
        "top_k": [1, 6, 4, 2, 0, -1, 5],
    }
    parameters = {
        name: (
            torch.tensor(value).reshape(-1, 1)
            if parameter_type == "column"
            else parameter_type(value)
        )
        for name, value in parameters.items()
    }
    _assert_wrapper_support(parameters, "subset", 3)


@pytest.mark.parametrize("parameter_type", ["scalar", "scalar_tensor", "singleton_tensor"])
@pytest.mark.parametrize(
    "temperature,top_p,top_k", [(0.7, 0.9, 4), (0.0, 0.5, 1), (1.0, None, None)]
)
def test_chunked_scalar_parameters_still_broadcast(parameter_type, temperature, top_p, top_k):
    parameters = {"temperature": temperature, "top_p": top_p, "top_k": top_k}
    if parameter_type != "scalar":
        parameters = {
            name: (
                None
                if value is None
                else torch.tensor([value] if parameter_type == "singleton_tensor" else value)
            )
            for name, value in parameters.items()
        }
    _assert_wrapper_support(parameters, "subset", 3)


@pytest.mark.parametrize("name", ["temperature", "top_p", "top_k"])
@pytest.mark.parametrize("active", [True, False])
def test_per_row_parameter_length_must_match_original_rows(name, active):
    with pytest.raises(ValueError, match=rf"{name}.*7"):
        vocab_parallel_sampling_keep_mask(
            torch.zeros(7, 11),
            real_vocab_size=11,
            active_rows=torch.full((7,), active),
            chunk_size=2,
            **{name: [1, 1]},
        )


def test_empty_batch_accepts_empty_per_row_parameters():
    logits = torch.empty(0, 11, requires_grad=True)
    actual = vocab_parallel_sampling_keep_mask(
        logits,
        real_vocab_size=9,
        temperature=torch.empty(0),
        top_p=torch.empty(0),
        top_k=torch.empty(0, dtype=torch.long),
        chunk_size=3,
    )
    assert actual.shape == logits.shape
    assert actual.dtype == torch.bool
    assert not actual.requires_grad
