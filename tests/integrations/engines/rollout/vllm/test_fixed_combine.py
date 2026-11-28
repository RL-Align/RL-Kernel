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


def test_scoped_rewrite_keeps_other_zero_buffers_and_restores_context():
    from contextvars import ContextVar
    from rl_engine.integrations.engines.rollout.vllm.fixed_combine import (
        _scoped_reduction,
    )

    def upstream(values, tokens):
        other = torch.zeros((2, 3))
        other.add_(2)

        def nested():
            return torch.zeros(1).add_(3)

        acc = torch.zeros((2, 3))
        acc.index_add_(0, tokens, values)
        return acc + other + nested()

    class Mode:
        calls = 0

        def reduce(self, target, *args, **kwargs):
            self.calls += 1
            return target.index_add_(*args, **kwargs)

    context = ContextVar("test_fixed_mode")
    scoped = _scoped_reduction(upstream, context)
    mode = Mode()
    token = context.set(mode)
    values, tokens = torch.ones((2, 3)), torch.tensor([0, 1])
    try:
        assert torch.equal(scoped(values, tokens), upstream(values, tokens))
        assert mode.calls == 1
    finally:
        context.reset(token)
    with pytest.raises(LookupError):
        scoped(values, tokens)


def test_unsupported_expert_count_fails_at_binding():
    from types import SimpleNamespace
    from rl_engine.integrations.engines.rollout.vllm.fixed_combine import (
        bind_fixed_combine,
    )

    experts = SimpleNamespace(
        w13_weight=torch.empty(128, 0, 0), w2_weight=torch.empty(128, 0, 0)
    )
    ffn = SimpleNamespace(experts=SimpleNamespace(routed_experts=experts))
    with pytest.raises(ValueError, match="exactly 256 local experts"):
        bind_fixed_combine(ffn)


def test_scoped_rewrite_rejects_preseeded_native_accumulator():
    from contextvars import ContextVar
    from rl_engine.integrations.engines.rollout.vllm.fixed_combine import (
        _scoped_reduction,
    )

    def preseeded(values, tokens):
        acc = torch.ones((2, 3))
        acc.index_add_(0, tokens, values)
        return acc

    def intervening_write(values, tokens):
        acc = torch.zeros((2, 3))
        acc.add_(1)
        acc.index_add_(0, tokens, values)
        return acc

    for function in (preseeded, intervening_write):
        with pytest.raises(ValueError, match="freshly zeroed native accumulator"):
            _scoped_reduction(function, ContextVar("unsupported_native_accumulator"))
