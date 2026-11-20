# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tasks 35 and 37 on real RCCL.

Run: PYTHONPATH=. torchrun --nproc-per-node=N <this file> [T] [H]

Tokens are split unevenly with rank 1 empty and inputs include NaN patterns. Output
is compared byte for byte with a single-process reference in (token, slot) order.
"""

import hashlib
import os
import random
import sys
import time

import torch
import torch.distributed as dist

from rl_engine.distributed.ep import (
    EpExchange,
    EpExchangeError,
    EpIdentity,
    EpPlacement,
)

E = 256
K = 6
IDENTITY = EpIdentity("rccl-check", "model-v1", "place-v1")


def fake_kernel(rows, experts):
    return (rows.view(torch.int16) ^ (experts[:, None].to(torch.int16) + 1)).view(rows.dtype)


def sizes(total, world):
    if world == 1:
        return [total]
    parts = [total // (world - 1)] * world
    parts[1] = 0
    parts[0] += total - sum(parts)
    return parts


def placement(world):
    return EpPlacement.from_owners([e * world // E for e in range(E)], world)


def make_inputs(total, hidden, seed=35):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-(2**15), 2**15, (total, hidden), generator=g, dtype=torch.int16)
    return x.view(torch.bfloat16), torch.rand(total, E, generator=g).argsort(dim=1)[:, :K]


def digest(t):
    return hashlib.sha256(t.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:12]


def round_trip(exchange, x, ids, stagger=None):
    def pause():
        if stagger is not None:
            time.sleep(stagger.random() * 0.02)

    pause()
    route = exchange.build_route(ids, IDENTITY)
    pause()
    rows = exchange.dispatch_tokens(route, x)
    weights = exchange.dispatch_slots(route, torch.ones((ids.shape[0], K), device=x.device))
    assert bool((weights == 1).all())
    experts = route.recv.to_expert_order(route.recv.experts)
    outputs = fake_kernel(rows, experts)
    pause()
    return exchange.return_outputs(route, outputs)


def check_equal(rank, label, out, x, ids):
    expected = fake_kernel(x.repeat_interleave(K, 0), ids.reshape(-1))
    if not torch.equal(out.rows.view(torch.int16).cpu(), expected.view(torch.int16)):
        raise SystemExit(f"rank {rank}: {label}: output differs from reference")


def scenario(group, total, hidden, label, *, strict=False, skew=False):
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    x_all, ids_all = make_inputs(total, hidden)
    if skew:
        ids_all = torch.full_like(ids_all, E - 1)
    begin, count = sum(sizes(total, world)[:rank]), sizes(total, world)[rank]
    x_cpu, ids_cpu = x_all[begin : begin + count], ids_all[begin : begin + count]
    exchange = EpExchange(group, placement(world), strict=strict)
    x, ids = x_cpu.cuda(), ids_cpu.cuda()

    first = round_trip(exchange, x, ids)
    check_equal(rank, label, first, x_cpu, ids_cpu)
    second = round_trip(exchange, x, ids, random.Random(rank * 1000 + 7))
    if not torch.equal(first.rows.view(torch.int16), second.rows.view(torch.int16)):
        raise SystemExit(f"rank {rank}: {label}: staggered run differs")
    return digest(first.rows)


def expect_all_fail(group, label, body):
    exchange = EpExchange(group, placement(dist.get_world_size(group)))
    try:
        body(exchange, dist.get_rank(group))
    except EpExchangeError:
        return
    raise SystemExit(f"rank {dist.get_rank(group)}: {label}: expected failure, none raised")


def failure_scenarios(group):
    world = dist.get_world_size(group)
    x, ids = make_inputs(16, 32)
    x, ids = x.cuda(), ids.cuda()

    def bad_identity(exchange, rank):
        identity = EpIdentity("other", "m", "p") if rank == world - 1 else IDENTITY
        exchange.build_route(ids, identity)

    def bad_expert(exchange, rank):
        exchange.build_route(ids + E if rank == 0 else ids, IDENTITY)

    def bad_width(exchange, rank):
        route = exchange.build_route(ids, IDENTITY)
        exchange.dispatch_tokens(route, x[:, :-1] if rank == 0 else x)

    def kernel_failure(exchange, rank):
        route = exchange.build_route(ids, IDENTITY)
        rows = exchange.dispatch_tokens(route, x)
        try:
            with exchange.guard():
                if rank == world - 1:
                    raise RuntimeError("simulated kernel failure")
        except RuntimeError as exc:
            raise EpExchangeError(str(exc)) from exc
        exchange.return_outputs(route, rows)

    for label, body in (
        ("identity", bad_identity),
        ("expert", bad_expert),
        ("width", bad_width),
        ("kernel", kernel_failure),
    ):
        # A lone rank has no peer to disagree with.
        if world > 1 or label not in ("identity", "width"):
            expect_all_fail(group, label, body)


def timing(group, total_per_rank, hidden, rounds=6):
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    tokens = 0 if rank == 1 and world > 1 else total_per_rank
    g = torch.Generator().manual_seed(rank)
    ids = torch.rand(tokens, E, generator=g).argsort(dim=1)[:, :K].cuda()
    x = torch.randn(tokens, hidden, generator=g).to(torch.bfloat16).cuda()
    exchange = EpExchange(group, placement(world))
    samples = []
    for _ in range(rounds):
        marks = [time.perf_counter()]
        route = exchange.build_route(ids, IDENTITY)
        marks.append(time.perf_counter())
        rows = exchange.dispatch_tokens(route, x)
        torch.cuda.synchronize()
        marks.append(time.perf_counter())
        out = exchange.return_outputs(route, rows)
        torch.cuda.synchronize()
        marks.append(time.perf_counter())
        samples.append([(b - a) * 1e3 for a, b in zip(marks, marks[1:])])
    assert out.rows.shape[0] == tokens * K
    return route.recv.num_rows, samples[-1]


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    group = dist.group.WORLD
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    hidden = int(sys.argv[2]) if len(sys.argv) > 2 else 4096

    digests = {
        "plain": scenario(group, total, hidden, "plain"),
        "strict": scenario(group, total, hidden, "strict", strict=True),
        "skew": scenario(group, 256, hidden, "skew", skew=True),
        "empty": scenario(group, 0, hidden, "empty"),
    }
    if world >= 4:
        # A stage-local EP group: group ranks differ from global ranks.
        half = world // 2
        groups = [dist.new_group(list(range(half))), dist.new_group(list(range(half, world)))]
        sub = groups[0] if rank < half else groups[1]
        digests["subgroup"] = scenario(sub, 1024, 256, "subgroup")
    failure_scenarios(group)
    inbound, warm = timing(group, total, hidden)
    rows = [None] * world
    dist.all_gather_object(rows, (rank, inbound, warm), group=group)
    if rank == 0:
        print(f"PASS ranks={world} T={total} H={hidden} (all scenarios byte-exact)")
        print("6th round, T per rank: rank inbound | route dispatch return ms")
        for r in rows:
            print(f"{r[0]} {r[1]} | " + " ".join(f"{v:.1f}" for v in r[2]))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
