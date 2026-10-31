# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Task 37 on a threaded transport that fills receive segments in shuffled order.

See check_ep_exchange_rccl.py for real RCCL.
"""

import random
import threading

import pytest
import torch

from rl_engine.distributed.ep import (
    EpExchange,
    EpExchangeError,
    EpIdentity,
    EpPlacement,
    TorchAllToAll,
)

pytestmark = pytest.mark.unit

E = 256
K = 6
H = 8
IDENTITY = EpIdentity("run", "model-v1", "place-v1")


class Fabric:
    def __init__(self, size, seed=0):
        self.size = size
        self.barrier = threading.Barrier(size, timeout=20)
        self.slots = [None] * size
        self.rng = random.Random(seed)
        self.moves = 0


class ThreadTransport:
    device = torch.device("cpu")

    def __init__(self, fabric, rank):
        self.fabric, self.rank, self.size = fabric, rank, fabric.size

    def _swap(self, item):
        self.fabric.slots[self.rank] = item
        self.fabric.barrier.wait()
        peers = list(self.fabric.slots)
        self.fabric.barrier.wait()
        return peers

    def gather_signature(self, signature):
        return torch.stack(self._swap(signature))

    def all_to_all(self, output, send, recv_counts, send_counts):
        peers = self._swap((send, send_counts))
        for source in self.fabric.rng.sample(range(self.size), self.size):
            peer_send, peer_counts = peers[source]
            start = sum(peer_counts[: self.rank])
            at = sum(recv_counts[:source])
            n = recv_counts[source]
            assert n == peer_counts[self.rank]
            output[at : at + n] = peer_send[start : start + n]
        # Blocking like the real collective: nobody returns, and so nobody may reuse a
        # send buffer, until every peer has finished reading it.
        self.fabric.barrier.wait()
        self.fabric.moves += 1


def run_ranks(size, body, seed=0):
    """Run ``body(rank, transport)`` on each rank; returns results or raised exceptions."""
    fabric = Fabric(size, seed)
    results = [None] * size

    def target(rank):
        try:
            results[rank] = body(rank, ThreadTransport(fabric, rank))
        except Exception as exc:  # noqa: BLE001
            results[rank] = exc

    threads = [threading.Thread(target=target, args=(r,)) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
        assert not t.is_alive(), "rank deadlocked"
    return results


def placement(ep):
    return EpPlacement.from_owners([e * ep // E for e in range(E)], ep)


def fake_kernel(rows, experts):
    return (rows.view(torch.int16) ^ (experts[:, None].to(torch.int16) + 1)).view(rows.dtype)


def inputs(total, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-(2**15), 2**15, (total, H), generator=g, dtype=torch.int16)
    ids = torch.rand(total, E, generator=g).argsort(dim=1)[:, :K]
    return x.view(torch.bfloat16), ids


def split(total, ep):
    if ep == 1:
        return [total]
    parts = [total // (ep - 1)] * ep
    parts[1] = 0
    parts[0] += total - sum(parts)
    return parts


def round_trip(ep, x, ids, *, strict=False, seed=0):
    sizes = split(x.shape[0], ep)
    xs, idss = x.split(sizes), ids.split(sizes)

    def body(rank, transport):
        exchange = EpExchange(None, placement(ep), strict=strict, transport=transport)
        route = exchange.build_route(idss[rank], IDENTITY)
        rows = exchange.dispatch_tokens(route, xs[rank])
        experts = route.recv.to_expert_order(route.recv.experts)
        return exchange.return_outputs(route, fake_kernel(rows, experts))

    return run_ranks(ep, body, seed=seed)


@pytest.mark.parametrize("ep", [1, 2, 4, 8])
@pytest.mark.parametrize("strict", [False, True])
def test_round_trip_is_byte_identical_under_shuffled_arrival(ep, strict):
    x, ids = inputs(97)
    expected = fake_kernel(x.repeat_interleave(K, 0), ids.reshape(-1)).view(torch.int16)
    for seed in range(4):
        outputs = round_trip(ep, x, ids, strict=strict, seed=seed)
        got = torch.cat([o.rows for o in outputs]).view(torch.int16)
        assert torch.equal(got, expected)


def test_every_slot_stays_a_separate_row_with_its_identity():
    x, ids = inputs(40)
    outputs = round_trip(4, x, ids)
    offset = 0
    for size, out in zip(split(40, 4), outputs, strict=True):
        assert out.rows.shape == (size * K, H)
        assert out.token_indices.tolist() == [t for t in range(size) for _ in range(K)]
        assert out.slot_indices.tolist() == list(range(K)) * size
        assert torch.equal(out.expert_indices, ids[offset : offset + size].reshape(-1))
        offset += size


def test_empty_expert_and_all_tokens_on_one_expert():
    x, _ = inputs(16)
    ids = torch.full((16, K), 200, dtype=torch.int64)  # owned by rank 3 of 4
    outputs = round_trip(4, x, ids)
    expected = fake_kernel(x.repeat_interleave(K, 0), ids.reshape(-1)).view(torch.int16)
    assert torch.equal(torch.cat([o.rows for o in outputs]).view(torch.int16), expected)


def test_all_ranks_empty_still_completes():
    x, ids = inputs(0)
    assert all(o.rows.shape == (0, H) for o in round_trip(4, x, ids))


def test_slot_values_travel_with_the_same_route():
    _, ids = inputs(30)
    weights = torch.rand(30, K)
    sizes = split(30, 4)
    idss, ws = ids.split(sizes), weights.split(sizes)

    def body(rank, transport):
        exchange = EpExchange(None, placement(4), transport=transport)
        route = exchange.build_route(idss[rank], IDENTITY)
        got = exchange.dispatch_slots(route, ws[rank])
        recv = route.recv
        order = recv.to_expert
        for i, row in enumerate(order.tolist()):
            source = int(recv.sources[row])
            want = ws[source][recv.tokens[row], recv.slots[row]]
            assert got[i].item() == want.item()
        return True

    assert run_ranks(4, body) == [True] * 4


def assert_all_raise(results, match):
    for result in results:
        assert isinstance(result, EpExchangeError), result
        assert match in str(result)


def test_one_rank_with_bad_route_fails_all_without_deadlock():
    _, ids = inputs(8)

    def body(rank, transport):
        exchange = EpExchange(None, placement(4), transport=transport)
        bad = ids + E if rank == 2 else ids
        return exchange.build_route(bad, IDENTITY)

    assert_all_raise(run_ranks(4, body), "ranks [2] failed")


def test_run_identity_mismatch_fails_all():
    _, ids = inputs(8)

    def body(rank, transport):
        identity = EpIdentity("other", "m", "p") if rank == 1 else IDENTITY
        return EpExchange(None, placement(4), transport=transport).build_route(ids, identity)

    assert_all_raise(run_ranks(4, body), "disagree")


def test_payload_width_mismatch_fails_all():
    x, ids = inputs(8)

    def body(rank, transport):
        exchange = EpExchange(None, placement(4), transport=transport)
        route = exchange.build_route(ids, IDENTITY)
        return exchange.dispatch_tokens(route, x[:, :-1] if rank == 3 else x)

    assert_all_raise(run_ranks(4, body), "disagree")


def test_operation_order_mismatch_fails_all():
    x, ids = inputs(8)

    def body(rank, transport):
        exchange = EpExchange(None, placement(2), transport=transport)
        route = exchange.build_route(ids, IDENTITY)
        if rank == 0:
            return exchange.dispatch_tokens(route, x)
        return exchange.return_outputs(route, torch.zeros(route.recv.num_rows, H))

    assert_all_raise(run_ranks(2, body), "disagree")


def test_kernel_failure_inside_guard_reaches_every_rank():
    x, ids = inputs(8)

    def body(rank, transport):
        exchange = EpExchange(None, placement(4), transport=transport)
        route = exchange.build_route(ids, IDENTITY)
        rows = exchange.dispatch_tokens(route, x)
        with exchange.guard():
            if rank == 1:
                raise RuntimeError("kernel OOM")
        return exchange.return_outputs(route, rows)

    results = run_ranks(4, body)
    assert isinstance(results[1], RuntimeError) and "kernel OOM" in str(results[1])
    for rank in (0, 2, 3):
        assert isinstance(results[rank], EpExchangeError)
        assert "ranks [1] failed" in str(results[rank])


def test_poisoned_exchange_refuses_further_work_locally():
    _, ids = inputs(8)

    def body(rank, transport):
        exchange = EpExchange(None, placement(2), transport=transport)
        with pytest.raises(EpExchangeError):
            exchange.build_route(ids + E, IDENTITY)
        moves = transport.fabric.moves
        with pytest.raises(EpExchangeError, match="poisoned"):
            exchange.build_route(ids, IDENTITY)
        assert transport.fabric.moves == moves
        return True

    assert run_ranks(2, body) == [True, True]


@pytest.mark.parametrize("phase", ["route", "dispatch", "return"])
def test_out_of_memory_on_one_rank_fails_all_before_the_collective(phase):
    x, ids = inputs(8)

    def body(rank, transport):
        exchange = EpExchange(None, placement(2), transport=transport)

        def arm():
            if rank == 1:

                def no_memory(shape, dtype):
                    raise torch.OutOfMemoryError("simulated")

                exchange._allocate = no_memory

        if phase == "route":
            arm()
        route = exchange.build_route(ids, IDENTITY)
        if phase == "dispatch":
            arm()
        rows = exchange.dispatch_tokens(route, x)
        if phase == "return":
            arm()
        return exchange.return_outputs(route, rows)

    assert_all_raise(run_ranks(2, body), "ranks [1] failed")


def test_returned_route_cannot_be_reused():
    x, ids = inputs(8)

    def body(rank, transport):
        exchange = EpExchange(None, placement(2), transport=transport)
        route = exchange.build_route(ids, IDENTITY)
        rows = exchange.dispatch_tokens(route, x)
        exchange.return_outputs(route, rows)
        return exchange.dispatch_tokens(route, x)

    assert_all_raise(run_ranks(2, body), "failed")


def test_route_from_another_exchange_is_rejected():
    x, ids = inputs(8)

    def body(rank, transport):
        first = EpExchange(None, placement(2), transport=transport)
        route = first.build_route(ids, IDENTITY)
        second = EpExchange(None, placement(2), transport=transport)
        return second.dispatch_tokens(route, x)

    assert_all_raise(run_ranks(2, body), "failed")


def test_payload_requiring_grad_is_rejected_not_silently_detached():
    x, ids = inputs(8)
    grad = torch.zeros(8, H, requires_grad=True)

    def body(rank, transport):
        exchange = EpExchange(None, placement(2), transport=transport)
        route = exchange.build_route(ids, IDENTITY)
        return exchange.dispatch_tokens(route, grad)

    assert_all_raise(run_ranks(2, body), "failed")


def test_strict_mode_detects_rows_corrupted_on_the_expert_side():
    x, ids = inputs(8)

    def body(rank, transport):
        exchange = EpExchange(None, placement(2), strict=True, transport=transport)
        route = exchange.build_route(ids, IDENTITY)
        if rank == 1 and route.recv.num_rows:
            route.recv.tokens[0] = route.recv.tokens[0] + 1
        rows = exchange.dispatch_tokens(route, x)
        return exchange.return_outputs(route, rows)

    assert_all_raise(run_ranks(2, body), "failed")


def test_placement_must_match_group_size():
    fabric = Fabric(2)
    with pytest.raises(EpExchangeError, match="placement size"):
        EpExchange(None, placement(4), transport=ThreadTransport(fabric, 0))


def test_torch_transport_requires_an_initialized_group():
    with pytest.raises(EpExchangeError, match="process group"):
        TorchAllToAll(None)
