# SPDX-License-Identifier: Apache-2.0
"""MI300X/MI355X: fixed P6 permute VJP is bytewise repeatable; RLK_REQUIRE_ROCM=1 forbids skips."""

import hashlib
import os
from types import SimpleNamespace

import pytest
import torch

from rl_engine.integrations.engines.train.megatron import native_combine

moe_utils = pytest.importorskip("megatron.core.transformer.moe.moe_utils")
permute = moe_utils.permute
unpermute = moe_utils.unpermute

REPEATS = 50
HIDDEN = 4096


@pytest.fixture(scope="module", autouse=True)
def require_device():
    available = bool(torch.version.hip and torch.cuda.is_available())
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] if available else "none"
    if arch not in ("gfx942", "gfx950"):
        reason = f"fixed permute VJP requires gfx942/gfx950, found {arch}"
        if os.getenv("RLK_REQUIRE_ROCM") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


class Dispatcher:
    def __init__(self, routing_map):
        self.routing_map = routing_map
        self.num_out_tokens = routing_map.shape[0] * 6

    def dispatch_preprocess(self, hidden_states, probs):
        tokens, permuted_probs, self.sorted_indices, _, _ = permute(
            hidden_states, self.routing_map, probs=probs, num_out_tokens=self.num_out_tokens
        )
        return tokens, permuted_probs

    def combine_postprocess(self, hidden_states):
        return unpermute(hidden_states, self.sorted_indices, None, routing_map=self.routing_map)


def routing(t, seed):
    g = torch.Generator().manual_seed(seed)
    experts = torch.rand(t, 256, generator=g).topk(6, dim=1).indices
    return torch.zeros(t, 256, dtype=torch.bool).scatter_(1, experts, True)


def fixed_reference(dy, routing_map):
    """CPU oracle: per token, FP32 sum of its rows by ascending expert, one RTNE cast."""
    experts, tokens = routing_map.T.nonzero(as_tuple=True)
    order = (tokens * 256 + experts).argsort(stable=True)  # token-major, expert-ascending
    rows = dy.float().index_select(0, order).reshape(routing_map.shape[0], 6, -1)
    acc = torch.zeros(rows.shape[0], rows.shape[2], dtype=torch.float32)
    for slot in range(6):
        acc = acc + rows[:, slot]
    return acc.to(dy.dtype)


def raw_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("t", [128, 2048, 8192])
def test_fixed_permute_vjp_top6_bf16_repeats_bytewise_and_matches_reference(t):
    routing_cpu = routing(t, seed=t)
    g = torch.Generator().manual_seed(t + 1)
    hidden = torch.randn(t, HIDDEN, generator=g).bfloat16().cuda()
    probs = torch.rand(t, 256, generator=g).cuda() * routing_cpu.cuda()
    dy_cpu = (torch.randn(t * 6, HIDDEN, generator=g) * 4).bfloat16()
    dy, routing_map = dy_cpu.cuda(), routing_cpu.cuda()

    dispatcher = Dispatcher(routing_map)
    native_combine.bind_native_combine(SimpleNamespace(token_dispatcher=dispatcher))
    native = Dispatcher(routing_map)
    expected_forward = native.dispatch_preprocess(hidden, probs)[0]

    expected = fixed_reference(dy_cpu, routing_cpu)
    first = None
    for _ in range(REPEATS):
        x = hidden.clone().requires_grad_()
        tokens, _ = dispatcher.dispatch_preprocess(x, probs)
        assert raw_equal(tokens.detach(), expected_forward)
        tokens.backward(dy)
        grad = x.grad.cpu()
        if first is None:
            first = grad
            assert raw_equal(first, expected), "fixed VJP differs from FP32 expert-ascending oracle"
        assert raw_equal(grad, first), "fixed VJP bytes changed between repeats"

    # Evidence only: the native BF16 atomic index_add_ VJP over the same rows.
    native_digests = set()
    for _ in range(REPEATS):
        x = hidden.clone().requires_grad_()
        Dispatcher(routing_map).dispatch_preprocess(x, probs)[0].backward(dy)
        native_digests.add(hashlib.sha256(x.grad.cpu().view(torch.uint8).numpy()).hexdigest())
    distinct = len(native_digests)
    print(f"T={t}: native index_select VJP distinct byte patterns in {REPEATS} repeats: {distinct}")
