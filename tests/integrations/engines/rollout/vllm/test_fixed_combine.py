# SPDX-License-Identifier: Apache-2.0
"""Regression: exact v37 layer-1 token-147/channel-1209 expert operands."""

import itertools
import struct

import pytest
import torch

from rl_engine.integrations.engines.rollout.vllm.fixed_combine import _FixedReduce

VALUES = [
    -0.08740234375,
    0.01092529296875,
    -1.709850039333105e-9,
    0.00775146484375,
    0.031982421875,
    0.01470947265625,
]
IDS = [7, 31, 37, 38, 126, 169]


def test_saved_v37_values_expose_order_sensitive_bf16_midpoint():
    def add32(a, b):
        return struct.unpack("f", struct.pack("f", a + b))[0]

    results = set()
    for order in itertools.permutations(range(6)):
        total = 0.0
        for i in order:
            total = add32(total, VALUES[i])
        results.add(
            float(
                torch.tensor(total).bfloat16()
                + torch.tensor(-0.01226806640625).bfloat16()
            )
        )
    assert results == {-0.0341796875, -0.034423828125}


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="ROCm GPU required"
)
@pytest.mark.parametrize("num_tokens", [162, 210, 324])
def test_exact_v37_routed_values_match_actor_across_original_b1_b2_layouts(num_tokens):
    ids = (
        torch.tensor(IDS, device="cuda", dtype=torch.int32)
        .expand(num_tokens, -1)
        .clone()
    )
    order = ids.reshape(-1).argsort(stable=True)
    tokens = (
        torch.arange(num_tokens, device="cuda")[:, None]
        .expand_as(ids)
        .reshape(-1)[order]
    )
    source = torch.zeros((num_tokens * 6, 4096), device="cuda", dtype=torch.float32)
    selected = torch.nonzero(tokens == 147).flatten()
    source[selected, 1209] = torch.tensor(VALUES, device="cuda")
    expected = torch.tensor(-0.02203369140625, device="cuda", dtype=torch.float32)
    for _ in range(3):
        output = torch.zeros((num_tokens, 4096), device="cuda", dtype=torch.float32)
        mode = _FixedReduce(ids)
        with mode:
            output.index_add_(0, tokens, source)
        assert mode.calls == 1
        assert torch.equal(
            output[147, 1209].view(torch.int32), expected.view(torch.int32)
        )
        final = (
            output[147, 1209].bfloat16()
            + torch.tensor(-0.01226806640625, device="cuda").bfloat16()
        )
        assert float(final) == -0.0341796875


def test_rejects_non_top6_before_execution():
    with pytest.raises(ValueError, match="Top6"):
        _FixedReduce(torch.zeros((2, 4), dtype=torch.int32))


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="ROCm GPU required"
)
def test_explicit_inactive_profile_rows_have_no_expert_contribution():
    ids = torch.full((2, 6), -1, device="cuda", dtype=torch.int32)
    tokens = torch.arange(2, device="cuda").repeat_interleave(6)
    output = torch.ones((2, 32), device="cuda")
    with _FixedReduce(ids) as mode:
        output.index_add_(0, tokens, torch.full((12, 32), float("nan"), device="cuda"))
    assert mode.calls == 1 and torch.count_nonzero(output) == 0
