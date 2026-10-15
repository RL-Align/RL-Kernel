# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU checks of the strict shared-forward boundary, without a CPU fallback."""

import pytest
import torch

from rl_engine.models.qwen3_next import qwen3_next_forward as provider


@pytest.mark.parametrize(
    "call",
    [
        lambda: provider.shared_linear(torch.zeros(1, 8), torch.zeros(8, 8)),
        lambda: provider.stable_top10_routes(torch.zeros(1, 512)),
        lambda: provider.combine_routes(torch.zeros(1, 10, 8), torch.zeros(1, 10)),
    ],
)
def test_required_cuda_primitives_reject_cpu(call):
    with pytest.raises(ValueError, match="no CPU fallback"):
        call()


def test_shared_linear_rejects_an_unpinned_vllm(monkeypatch):
    provider._vllm_linear.cache_clear()
    monkeypatch.setattr(provider, "version", lambda package: "0.30.1")
    with pytest.raises(RuntimeError, match="pinned vLLM 0.30.0"):
        provider._vllm_linear()
    provider._vllm_linear.cache_clear()


def _rows(count, width, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((count, width), generator=generator, dtype=torch.float32) * 4


@pytest.mark.parametrize("width", [1, 2, 7, 10, 512])
def test_fixed_order_row_sum_matches_sequential_double_sum(width):
    values = _rows(5, width, 901)
    actual = provider.fixed_order_row_sum(values)
    assert actual.shape == (5, 1)
    reference = values.double().sum(-1, keepdim=True).float()
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("width", [10, 512])
def test_fixed_order_row_sum_is_independent_of_row_count(width):
    values = _rows(32, width, 902)
    whole = provider.fixed_order_row_sum(values)
    for count in (1, 8, 17):
        assert torch.equal(provider.fixed_order_row_sum(values[:count]), whole[:count])


def test_fixed_order_softmax_rows_do_not_depend_on_batch():
    logits = _rows(32, 512, 903)
    whole = provider.fixed_order_softmax(logits)
    torch.testing.assert_close(whole.double().sum(-1), torch.ones(32, dtype=torch.float64))
    for count in (1, 8):
        assert torch.equal(provider.fixed_order_softmax(logits[:count]), whole[:count])
    # The row sum is a fixed sequential chain, so each result is a pure function of
    # its own row and cannot depend on how many rows share the call.
