# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tasks 37 and 38 with the real fused kernels on real RCCL.

Run: PYTHONPATH=. torchrun --nproc-per-node=N <this file> [T] [H] [F]

Rank 0 also computes a reference with no exchange at all: one batch holding every
expert, sorted by plain argsort. The gathered EP output must equal it byte for byte,
and its printed sha must be the same for every N.
"""

import hashlib
import os
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
from rl_engine.moe.contract import ExpertBatch
from rl_engine.moe.ep_expert import receive_expert_input, routed_expert_forward_ep
from rl_engine.moe.mx_format import MXTensor, mx_quantize
from rl_engine.moe.triton_fused_mlp import TritonFusedMoeMlp

E = 256
K = 6
IDENTITY = EpIdentity("expert-check", "model-v1", "place-v1")


def sizes(total, world):
    if world == 1:
        return [total]
    parts = [total // (world - 1)] * world
    parts[1] = 0
    parts[0] += total - sum(parts)
    return parts


def make_inputs(total, hidden, seed=38):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(total, hidden, generator=g) * 0.7).to(torch.bfloat16)
    p_s = torch.softmax(torch.randn(total, K, generator=g), dim=1)
    ids = torch.rand(total, E, generator=g).argsort(dim=1)[:, :K]
    return x, p_s, ids


def expert_weights(expert_ids, hidden, ffn):
    """Each expert's bytes depend only on its global id."""
    w1, w2 = [], []
    for e in expert_ids:
        g = torch.Generator().manual_seed(1000 + e)
        a = (torch.randn(2 * ffn, hidden, generator=g) / hidden**0.5).to(torch.bfloat16)
        b = (torch.randn(hidden, ffn, generator=g) / ffn**0.5).to(torch.bfloat16)
        w1.append(mx_quantize(a.cuda(), "e2m1"))
        w2.append(mx_quantize(b.cuda(), "e2m1"))

    def stack(parts):
        return MXTensor(
            torch.stack([p.codes for p in parts]),
            torch.stack([p.scales for p in parts]),
            "e2m1",
            (len(parts), *parts[0].shape),
        )

    return stack(w1), stack(w2)


def digest(rows):
    return hashlib.sha256(rows.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]


def reference(provider, x, p_s, ids, hidden, ffn):
    """Single batch, no plan, no exchange."""
    w1, w2 = expert_weights(range(E), hidden, ffn)
    flat = ids.reshape(-1).cuda()
    order = torch.argsort(flat, stable=True)
    tokens = torch.div(order, K, rounding_mode="floor")
    rows = x.cuda()[tokens]
    weights = p_s.reshape(-1).cuda()[order]
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int32, device="cuda"),
         torch.bincount(flat, minlength=E).cumsum(0).to(torch.int32))
    )
    batch = ExpertBatch(
        x=rows,
        expert_offsets=offsets,
        p_s=weights,
        w1=w1,
        w2=w2,
        lora=None,
        output_slot=torch.arange(rows.shape[0], dtype=torch.int32, device="cuda"),
        numeric_profile=provider.numeric_profile,
    )
    y = provider.forward(batch, mx_quantize(rows, "e4m3"))
    out = torch.empty_like(y)
    out[order] = y
    return out.cpu()


def fail(rank, message):
    raise SystemExit(f"rank {rank}: {message}")


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    group = dist.group.WORLD
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    hidden = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    ffn = int(sys.argv[3]) if len(sys.argv) > 3 else 256

    provider = TritonFusedMoeMlp()
    place = EpPlacement.from_owners([e * world // E for e in range(E)], world)
    w1, w2 = expert_weights(place.local_expert_ids(rank).tolist(), hidden, ffn)

    x_all, p_all, ids_all = make_inputs(total, hidden)
    parts = sizes(total, world)
    begins = [sum(parts[:r]) for r in range(world)]
    begin, count = begins[rank], parts[rank]
    x = x_all[begin : begin + count].cuda()
    p_s = p_all[begin : begin + count].cuda()
    ids = ids_all[begin : begin + count].cuda()
    exchange = EpExchange(group, place)

    # Task 38: raw bytes on arrival, quantization derived on the receiver.
    route = exchange.build_route(ids, IDENTITY)
    got = receive_expert_input(exchange, route, x, p_s)
    recv = route.recv
    order = recv.to_expert.cpu()
    tokens = torch.tensor(begins)[recv.sources[order].cpu()] + recv.tokens[order].cpu()
    want = x_all[tokens]
    if not torch.equal(got.x.view(torch.int16).cpu(), want.view(torch.int16)):
        fail(rank, "raw BF16 changed in transit")
    if not torch.equal(got.p_s.cpu(), p_all[tokens, recv.slots[order].cpu()]):
        fail(rank, "p_s changed in transit")
    quantized = mx_quantize(want.cuda(), "e4m3")
    if not (torch.equal(got.x_q.codes, quantized.codes)
            and torch.equal(got.x_q.scales, quantized.scales)):
        fail(rank, "receiver quantization differs from quantizing the source row")

    # Tasks 37 and 38: the whole layer against a reference with no exchange.
    route = exchange.build_route(ids, IDENTITY)
    before = digest(x)
    out = routed_expert_forward_ep(exchange, route, provider, x, p_s, w1, w2)
    torch.cuda.synchronize()
    if digest(x) != before:
        fail(rank, "the raw input was modified")
    if not (torch.equal(out.token_indices.cpu(), torch.arange(count).repeat_interleave(K))
            and torch.equal(out.slot_indices.cpu(), torch.arange(K).repeat(count))
            and torch.equal(out.expert_indices.cpu(), ids_all[begin : begin + count].reshape(-1))):
        fail(rank, "slot identity is wrong")

    gathered = [None] * world
    dist.all_gather_object(gathered, out.rows.cpu(), group=group)
    sha = digest(torch.cat(gathered))
    if rank == 0:
        ref = reference(provider, x_all, p_all, ids_all, hidden, ffn)
        if digest(ref) != sha:
            fail(rank, f"EP output {sha} differs from no-exchange reference {digest(ref)}")

    # All tokens on the last expert: every other rank runs the kernels with zero rows.
    skew = torch.full_like(ids, E - 1)
    route = exchange.build_route(skew, IDENTITY)
    skewed = routed_expert_forward_ep(exchange, route, provider, x, p_s, w1, w2)
    if skewed.rows.shape != (count * K, hidden):
        fail(rank, "skewed output has the wrong shape")

    # One rank's non-finite input must fail every rank, not hang them.
    if world > 1:
        bad = x
        if rank == 0:
            bad = x.clone()
            bad[0, 0] = float("nan")
        doomed = EpExchange(group, place)
        try:
            route = doomed.build_route(torch.full_like(ids, 0), IDENTITY)
            routed_expert_forward_ep(doomed, route, provider, bad, p_s, w1, w2)
            fail(rank, "non-finite input was accepted")
        except (ValueError, EpExchangeError):
            pass
    dist.barrier(group=group)

    # Timing of the full layer with the real kernels.
    tokens_t = 0 if rank == 1 and world > 1 else total
    g = torch.Generator().manual_seed(rank)
    x_t = (torch.randn(tokens_t, hidden, generator=g) * 0.7).to(torch.bfloat16).cuda()
    p_t = torch.softmax(torch.randn(tokens_t, K, generator=g), dim=1).cuda()
    ids_t = torch.rand(tokens_t, E, generator=g).argsort(dim=1)[:, :K].cuda()
    laps = []
    for _ in range(5):
        torch.cuda.synchronize()
        marks = [time.perf_counter()]
        route = exchange.build_route(ids_t, IDENTITY)
        marks.append(time.perf_counter())
        received = receive_expert_input(exchange, route, x_t, p_t)
        torch.cuda.synchronize()
        marks.append(time.perf_counter())
        batch = ExpertBatch(
            x=received.x,
            expert_offsets=received.expert_offsets,
            p_s=received.p_s,
            w1=w1,
            w2=w2,
            lora=None,
            output_slot=received.output_slot,
            numeric_profile=provider.numeric_profile,
        )
        y = provider.forward(batch, received.x_q)
        torch.cuda.synchronize()
        marks.append(time.perf_counter())
        exchange.return_outputs(route, y)
        torch.cuda.synchronize()
        marks.append(time.perf_counter())
        laps.append([(b - a) * 1e3 for a, b in zip(marks, marks[1:])])
    rows = [None] * world
    dist.all_gather_object(rows, (rank, route.recv.num_rows, laps[-1]), group=group)
    if rank == 0:
        print(f"PASS ranks={world} T={total} H={hidden} F={ffn} sha={sha}")
        print("last of 5, T per rank: rank inbound | route input+quant kernel return ms")
        for r in rows:
            print(f"{r[0]} {r[1]} | " + " ".join(f"{v:.1f}" for v in r[2]))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
