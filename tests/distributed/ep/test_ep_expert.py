# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Task 38 on the threaded transport with a stand-in provider.

See check_ep_expert_rccl.py for the real kernels on real RCCL.
"""

import pytest
import torch

from rl_engine.distributed.ep import EpExchange, EpExchangeError, EpIdentity, EpPlacement
from rl_engine.moe.ep_expert import (
    receive_expert_input,
    routed_expert_forward_ep,
    select_local_weights,
)
from rl_engine.moe.mx_format import mx_dequantize, mx_quantize
from tests.distributed.ep.test_ep_exchange import run_ranks, split

pytestmark = pytest.mark.unit

E = 256
K = 6
H = 64
F = 32
IDENTITY = EpIdentity("run", "model-v1", "place-v1")


def placement(ep):
    return EpPlacement.from_owners([e * ep // E for e in range(E)], ep)


def inputs(total, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(total, H, generator=g).to(torch.bfloat16)
    p_s = torch.rand(total, K, generator=g)
    ids = torch.rand(total, E, generator=g).argsort(dim=1)[:, :K]
    return x, p_s, ids


class RecordingProvider:
    """Depends only on each row and its expert, like a batch-invariant kernel."""

    numeric_profile = "test"

    def __init__(self):
        self.batch = None
        self.x_q = None

    def forward(self, batch, x_q):
        self.batch, self.x_q = batch, x_q
        experts = torch.repeat_interleave(
            torch.arange(batch.expert_offsets.numel() - 1),
            batch.expert_offsets[1:] - batch.expert_offsets[:-1],
        )
        tag = batch.w1.codes[experts, 0, 0].to(torch.int16)[:, None]
        return (batch.x.view(torch.int16) ^ tag).view(torch.bfloat16)


def weights():
    g = torch.Generator().manual_seed(9)
    w1 = mx_quantize(torch.randn(E, 2, H, generator=g).to(torch.bfloat16), "e2m1")
    w2 = mx_quantize(torch.randn(E, 1, 32, generator=g).to(torch.bfloat16), "e2m1")
    return w1, w2


def global_rows(ep, x):
    """(source, token) -> global token, to compare received rows with the input."""
    sizes = split(x.shape[0], ep)
    return torch.tensor([sum(sizes[:r]) for r in range(ep)]), sizes


@pytest.mark.parametrize("ep", [1, 2, 4, 8])
def test_receiver_keeps_raw_bytes_and_quantizes_locally(ep):
    x, p_s, ids = inputs(40)
    begins, sizes = global_rows(ep, x)
    xs, ws, idss = x.split(sizes), p_s.split(sizes), ids.split(sizes)

    def body(rank, transport):
        exchange = EpExchange(None, placement(ep), transport=transport)
        route = exchange.build_route(idss[rank], IDENTITY)
        got = receive_expert_input(exchange, route, xs[rank], ws[rank])
        recv = route.recv
        order = recv.to_expert
        tokens = begins[recv.sources[order]] + recv.tokens[order]
        want = x[tokens]
        assert torch.equal(got.x.view(torch.int16), want.view(torch.int16))
        assert torch.equal(got.p_s, p_s[tokens, recv.slots[order]])
        reference = mx_quantize(want, "e4m3")
        assert torch.equal(got.x_q.codes, reference.codes)
        assert torch.equal(got.x_q.scales, reference.scales)
        assert got.expert_offsets.dtype == got.output_slot.dtype == torch.int32
        assert torch.equal(got.output_slot.to(torch.int64), order)
        return True

    assert run_ranks(ep, body) == [True] * ep


def test_quantization_does_not_depend_on_row_grouping():
    x, _, _ = inputs(40)
    full = mx_quantize(x, "e4m3")
    for row in (0, 7, 39):
        one = mx_quantize(x[row : row + 1], "e4m3")
        assert torch.equal(one.codes[0], full.codes[row])
        assert torch.equal(one.scales[0], full.scales[row])


def test_output_is_byte_identical_across_ep_sizes():
    x, p_s, ids = inputs(97)
    w1, w2 = weights()
    outputs = {}
    for ep in (1, 2, 4, 8):
        sizes = split(97, ep)
        xs, ws, idss = x.split(sizes), p_s.split(sizes), ids.split(sizes)

        def body(rank, transport):
            place = placement(ep)
            exchange = EpExchange(None, place, transport=transport)
            route = exchange.build_route(idss[rank], IDENTITY)
            local1 = select_local_weights(w1, place, rank)
            local2 = select_local_weights(w2, place, rank)
            return routed_expert_forward_ep(
                exchange, route, RecordingProvider(), xs[rank], ws[rank], local1, local2
            )

        results = run_ranks(ep, body)
        outputs[ep] = torch.cat([o.rows for o in results]).view(torch.int16)
    for ep in (2, 4, 8):
        assert torch.equal(outputs[ep], outputs[1]), f"ep={ep}"


def test_local_weights_follow_local_expert_ids():
    w1, _ = weights()
    place = EpPlacement.from_owners([e % 4 for e in range(E)], 4)
    local = select_local_weights(w1, place, 2)
    assert place.local_expert_ids(2).tolist() == list(range(2, E, 4))
    assert local.shape == (E // 4, 2, H)
    assert torch.equal(local.codes[1], w1.codes[6])
    assert torch.equal(local.scales[1], w1.scales[6])


def test_lora_input_is_the_raw_row_not_a_decoded_quantized_row():
    x, p_s, ids = inputs(8)

    def body(rank, transport):
        exchange = EpExchange(None, placement(1), transport=transport)
        route = exchange.build_route(ids, IDENTITY)
        got = receive_expert_input(exchange, route, x, p_s)
        assert not torch.equal(got.x.to(torch.float32), mx_dequantize(got.x_q))
        return True

    assert run_ranks(1, body) == [True]


@pytest.mark.parametrize(
    ("x_dtype", "p_dtype"),
    [(torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16)],
)
def test_wrong_dtype_on_one_rank_fails_all(x_dtype, p_dtype):
    x, p_s, ids = inputs(8)

    def body(rank, transport):
        exchange = EpExchange(None, placement(2), transport=transport)
        route = exchange.build_route(ids, IDENTITY)
        if rank == 1:
            return receive_expert_input(exchange, route, x.to(x_dtype), p_s.to(p_dtype))
        return receive_expert_input(exchange, route, x, p_s)

    results = run_ranks(2, body)
    assert isinstance(results[1], TypeError)
    assert isinstance(results[0], EpExchangeError)


def test_non_finite_input_fails_every_rank_without_deadlock():
    x, p_s, ids = inputs(40)
    # Every token reaches expert 200, owned by the last rank; rank 0 sends the NaN.
    ids[:, 0] = 200
    bad = x.clone()
    bad[3, 5] = float("nan")

    def body(rank, transport):
        exchange = EpExchange(None, placement(2), transport=transport)
        route = exchange.build_route(ids, IDENTITY)
        return receive_expert_input(exchange, route, bad if rank == 0 else x, p_s)

    results = run_ranks(2, body)
    assert all(isinstance(r, Exception) for r in results)


def test_provider_failure_fails_every_rank():
    x, p_s, ids = inputs(8)
    w1, w2 = weights()

    class Broken(RecordingProvider):
        def forward(self, batch, x_q):
            raise NotImplementedError("LoRA is not supported")

    def body(rank, transport):
        place = placement(2)
        exchange = EpExchange(None, place, transport=transport)
        route = exchange.build_route(ids, IDENTITY)
        provider = Broken() if rank == 1 else RecordingProvider()
        return routed_expert_forward_ep(
            exchange,
            route,
            provider,
            x,
            p_s,
            select_local_weights(w1, place, rank),
            select_local_weights(w2, place, rank),
        )

    results = run_ranks(2, body)
    assert isinstance(results[1], NotImplementedError)
    assert isinstance(results[0], EpExchangeError)
