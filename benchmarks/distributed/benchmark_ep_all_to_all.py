# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark the Nemotron EP exchange against native NCCL ``all_to_all_single``.

Example, with the GPUs to itself (no filler or other job running):

    PYTHONPATH=. torchrun --nproc-per-node=8 --master-addr=127.0.0.1 --master-port=29500 \
      benchmarks/distributed/benchmark_ep_all_to_all.py --output ep_all_to_all_h20.json

Forward only (``--forward-tokens``, rollout decode and prefill sizes), under ``no_grad``:

* ``native_forward``: plain ``torch.distributed`` with no checks: route (counts, expert
  ids, orderings), pack and dispatch, return and restore ``(token, slot)``.
* ``candidate_strict_forward`` / ``candidate_fast_forward``: ``dispatch`` then
  ``return_outputs``, with and without the return-trip echo. Both include the input checks,
  agreements and route building that the public entry point always runs.

Training (``--tokens``), both sides given the same ``dy`` for the returned rows:

* ``native_train``: the native forward, then ``dy`` to the owners and ``dX`` back with the
  FP32 slot sum.
* ``candidate_train``: the public training path, ``dispatch``, ``return_outputs`` and
  ``ep_backward``; ``candidate_train / native_train`` is the training cost. The expert is
  the identity on ``x_rows``, and the loss is a cotangent injector (``SeedGrad``) whose
  backward hands ``dy`` to the rows unchanged, so neither side times a full-size multiply,
  reduction or their backward. The injector's node and scalar remain on this side.
* ``candidate_direct``: the same sequence of agreements and all-to-alls, called on the
  exchange directly under ``no_grad``. It is not the training path: it skips the call
  layer's checks on the returned outputs, agrees on the backward with the route's tag
  instead of a digest of all calls, enters with the no-grad mode, and has no graph walk,
  anchor, link or injector. ``candidate_direct / native_train`` compares this direct
  implementation with the hand-written native one, extra checks, metadata and allocation
  included; ``candidate_train - candidate_direct`` is an approximate increment of the
  public path over the direct one, a difference of medians that can be noisy or negative.

Plus ``transport``, one payload move with the exchange's counts. Before timing, every
scope's outputs and ``dX`` are compared bit for bit with the native side, and the gradient
reaching the candidate's rows with ``dy``. Every round starts behind a barrier and a
device synchronize; each reported time is the slowest rank's for that round. Full scopes
synchronize only at the end. Phase scopes synchronize at every phase boundary, which
removes overlap between phases, so their sum can exceed the full time. The first round
after warm-up is reported apart as ``first_ms``. ``--backend gloo`` runs the same checks
on CPU as a smoke test; its times mean nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from rl_engine.distributed.ep.cuda_transport import CudaNcclAllToAll
from rl_engine.distributed.ep.exchange import TorchAllToAll
from rl_engine.distributed.ep.nemotron import (
    _OP_BACKWARD,
    HIDDEN_SIZE,
    NUM_EXPERTS,
    TOP_K,
    EpInvocation,
    NemotronEpExchange,
    ep_backward,
)
from rl_engine.distributed.ep.plan import EpIdentity, EpPlacement

DECODE_AND_PREFILL = [1, 16, 64, 256, 2048, 8192, 32768]


class GlooAllToAll(TorchAllToAll):
    """CPU smoke of this script only; says nothing about NCCL or its timing."""

    backend_id = "gloo-cpu-smoke"

    def __init__(self, group=None) -> None:
        self.group = group
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        self.device = torch.device("cpu")

    def gather_signature(self, signature: torch.Tensor) -> torch.Tensor:
        peers = [torch.empty_like(signature) for _ in range(self.size)]
        dist.all_gather(peers, signature, group=self.group)
        return torch.stack(peers)


def synchronize() -> None:
    if dist.get_backend() == "nccl":
        torch.cuda.synchronize()


class Clock:
    """Wall time on this rank between synchronized marks."""

    def __init__(self) -> None:
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        synchronize()
        self.marks[name] = time.perf_counter()

    def spans(self, spans: dict[str, tuple[str, str]]) -> dict[str, float]:
        return {k: self.marks[b] - self.marks[a] for k, (a, b) in spans.items()}


def slowest_rank(values: dict[str, float], device) -> dict[str, float]:
    names = sorted(values)
    tensor = torch.tensor([values[n] for n in names], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return dict(zip(names, tensor.tolist()))


def summarize(samples: list[float]) -> dict:
    ms = [s * 1e3 for s in samples]
    return {
        "median_ms": statistics.median(ms),
        "min_ms": min(ms),
        "max_ms": max(ms),
        "samples_ms": ms,
    }


def timed(fn, spans, device, rounds, warmup) -> dict:
    """``fn(index, clock)`` marks ``start`` .. ``end``; ``spans`` names what to report."""
    collected: dict[str, list[float]] = {name: [] for name in spans}
    first: dict[str, float] = {}
    for index in range(warmup + 1 + rounds):
        dist.barrier()
        clock = Clock()
        clock.mark("start")
        fn(index, clock)
        values = slowest_rank(clock.spans(spans), device)
        if index == warmup:
            first = values
        elif index > warmup:
            for name, value in values.items():
                collected[name].append(value)
    return {name: {**summarize(s), "first_ms": first[name] * 1e3} for name, s in collected.items()}


def inputs(tokens: int, rank: int, device):
    g = torch.Generator(device=device).manual_seed(1000 + rank)
    x = torch.randn(tokens, HIDDEN_SIZE, generator=g, device=device).to(torch.bfloat16)
    ids = (
        torch.rand(tokens, NUM_EXPERTS, generator=g, device=device)
        .argsort(dim=1)[:, :TOP_K]
        .sort(dim=1)
        .values
    )
    dy = torch.randn(tokens * TOP_K, HIDDEN_SIZE, generator=g, device=device)
    return x, ids, dy.to(torch.bfloat16)


class NativeRoute:
    """The exchange's semantics in plain ``torch.distributed``, without any checks."""

    def __init__(self, ids, owner, local, size) -> None:
        experts = ids.reshape(-1)
        destinations = owner[experts]
        self.send_order = torch.argsort(destinations, stable=True)
        counts = torch.bincount(destinations, minlength=size)
        recv_counts = torch.empty_like(counts)
        dist.all_to_all_single(recv_counts, counts, [1] * size, [1] * size)
        self.send, self.recv = counts.tolist(), recv_counts.tolist()
        arrived = torch.empty(sum(self.recv), dtype=torch.int64, device=ids.device)
        dist.all_to_all_single(arrived, experts[self.send_order], self.recv, self.send)
        self.to_expert = torch.argsort(local[arrived], stable=True)
        self.to_arrival = torch.empty_like(self.to_expert)
        self.to_arrival[self.to_expert] = torch.arange(self.to_expert.numel(), device=ids.device)
        self.to_canonical = torch.argsort(self.send_order)

    def to_owners(self, rows: torch.Tensor) -> torch.Tensor:
        received = rows.new_empty((sum(self.recv), rows.shape[1]))
        dist.all_to_all_single(received, rows, self.recv, self.send)
        return received[self.to_expert]

    def to_sources(self, rows: torch.Tensor) -> torch.Tensor:
        home = rows.new_empty((sum(self.send), rows.shape[1]))
        dist.all_to_all_single(home, rows[self.to_arrival], self.send, self.recv)
        return home.index_select(0, self.to_canonical)

    def dispatch(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_owners(x.index_select(0, self.send_order // TOP_K))

    def dispatch_dy(self, dy: torch.Tensor) -> torch.Tensor:
        return self.to_owners(dy.index_select(0, self.send_order))

    def return_dx(self, dx_rows: torch.Tensor) -> torch.Tensor:
        lanes = self.to_sources(dx_rows).view(-1, TOP_K, dx_rows.shape[1])
        total = torch.zeros_like(lanes[:, 0], dtype=torch.float32)
        for slot in range(TOP_K):
            total = total + lanes[:, slot].to(torch.float32)
        return total.to(dx_rows.dtype)


def native_forward(x, ids, owner, local, size, clock=None):
    mark = clock.mark if clock else (lambda _: None)
    route = NativeRoute(ids, owner, local, size)
    mark("route")
    rows = route.dispatch(x)
    mark("dispatch")
    out = route.to_sources(rows)
    mark("return")
    return route, out


def native_train(x, ids, dy, owner, local, size, clock=None):
    mark = clock.mark if clock else (lambda _: None)
    route, out = native_forward(x, ids, owner, local, size, clock)
    dy_rows = route.dispatch_dy(dy)
    mark("dy")
    dx = route.return_dx(dy_rows)
    mark("end")
    return out, dx


class SeedGrad(torch.autograd.Function):
    """Cotangent injector: a scalar whose backward hands ``dy`` to ``rows`` unchanged.

    Only for an unscaled ``backward()`` of this scalar, whose incoming gradient is 1.
    """

    @staticmethod
    def forward(ctx, rows, dy):
        ctx.dy = dy
        return rows.new_zeros((), dtype=torch.float32)

    @staticmethod
    def backward(ctx, _grad):
        return ctx.dy, None


def candidate_train(exchange, x, ids, dy, invocation, clock=None, seen=None):
    """One training step through the public path with an identity expert."""
    mark = clock.mark if clock else (lambda _: None)
    leaf = x.detach().requires_grad_(True)
    call = exchange.dispatch(leaf, ids, invocation)
    mark("dispatch")
    rows = call.return_outputs(call.x_rows)
    mark("return")
    loss = SeedGrad.apply(rows, dy)
    if seen is not None:
        rows.register_hook(seen.append)
    if clock:
        rows.grad_fn.register_prehook(lambda *_: clock.mark("dy_start"))
        rows.grad_fn.register_hook(lambda *_: clock.mark("dy_end"))
        call.x_rows.grad_fn.register_prehook(lambda *_: clock.mark("dx_start"))
        call.x_rows.grad_fn.register_hook(lambda *_: clock.mark("dx_end"))
    mark("loss")
    ep_backward(exchange, loss, [call])
    mark("end")
    return rows.detach(), leaf.grad


def candidate_direct(exchange, x, ids, dy, invocation):
    """``candidate_train``'s agreements and all-to-alls in the same order, without autograd.

    Not the training path; see the module docstring for what it skips.
    """
    inner = exchange.exchange
    with torch.no_grad():
        call = exchange.dispatch(x, ids, invocation)
        rows = inner.return_outputs(call.route, call.x_rows).rows
        saved = inner.save_route(call.route)
        inner.agree(_OP_BACKWARD, run_tag=call.route.send.run_tag, width=1)
        dx = inner.return_dx(saved, inner.dispatch_dy(saved, dy), x.dtype)
    return rows, dx


def candidate_forward(exchange, x, ids, invocation):
    with torch.no_grad():
        call = exchange.dispatch(x, ids, invocation)
        return call.return_outputs(call.x_rows)


def same_bits(pairs, device) -> list[int]:
    """Every rank's verdict on each pair; one rank's mismatch fails all ranks."""
    verdict = [int(torch.equal(a.view(torch.int16), b.view(torch.int16))) for a, b in pairs]
    flags = torch.tensor(verdict, dtype=torch.int32, device=device)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    return flags.tolist()


def peak_memory(device) -> int:
    if device.type != "cuda":
        return 0
    peak = torch.tensor([torch.cuda.max_memory_allocated(device)], device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    return int(peak.item())


def transport_scope(exchange, x, ids, device, run) -> dict:
    with torch.no_grad():
        route = exchange.dispatch(x, ids, EpInvocation(9, 0, 0)).route
    send, recv = route.send_counts, route.recv_counts
    payload = route.send.pack_token_rows(x)
    output = torch.empty((sum(recv), HIDDEN_SIZE), dtype=x.dtype, device=device)
    return timed(
        lambda i, c: (
            dist.all_to_all_single(output, payload, recv, send),
            c.mark("end"),
        ),
        {"full": ("start", "end")},
        **run,
    )


def exchanges(placement, device):
    if device.type == "cuda":
        transport = CudaNcclAllToAll(None)
    else:
        transport = GlooAllToAll(None)
    base = dict(model_version="bench", placement_version="bench", transport=transport)
    strict = NemotronEpExchange(placement, **base)
    fast = NemotronEpExchange(placement, strict=False, **base)
    return strict, fast


def header(tokens: int, size: int) -> dict:
    return {
        "tokens_per_rank": tokens,
        "ep_size": size,
        "rows_per_rank": tokens * TOP_K,
        "payload_bytes_per_rank": tokens * TOP_K * HIDDEN_SIZE * 2,
    }


def measure_forward(tokens: int, placement, owner, local, device, args) -> dict:
    """Rollout: forward only, native against the exchange with and without the echo."""
    rank, size = dist.get_rank(), dist.get_world_size()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    x, ids, _ = inputs(tokens, rank, device)
    strict, fast = exchanges(placement, device)
    with torch.no_grad():
        _, native_out = native_forward(x, ids, owner, local, size)
    strict_out = candidate_forward(strict, x, ids, EpInvocation(0, 0, 0))
    fast_out = candidate_forward(fast, x, ids, EpInvocation(0, 0, 0))
    same = same_bits([(strict_out, native_out), (fast_out, native_out)], device)
    if not all(same):
        raise AssertionError(f"forward tokens={tokens}: native and candidate differ {same}")

    full = {"full": ("start", "end")}
    run = dict(device=device, rounds=args.rounds, warmup=args.warmup)
    row = header(tokens, size)
    row["transport"] = transport_scope(strict, x, ids, device, run)

    def native(i, c):
        with torch.no_grad():
            native_forward(x, ids, owner, local, size)
        c.mark("end")

    row["native_forward"] = timed(native, full, **run)
    row["candidate_strict_forward"] = timed(
        lambda i, c: (
            candidate_forward(strict, x, ids, EpInvocation(1, i, 0)),
            c.mark("end"),
        ),
        full,
        **run,
    )
    row["candidate_fast_forward"] = timed(
        lambda i, c: (
            candidate_forward(fast, x, ids, EpInvocation(2, i, 0)),
            c.mark("end"),
        ),
        full,
        **run,
    )
    row["peak_allocated_bytes_max_rank"] = peak_memory(device)
    return row


def measure_train(tokens: int, placement, owner, local, device, args) -> dict:
    """Training: forward and backward, both sides given the same ``dy``."""
    rank, size = dist.get_rank(), dist.get_world_size()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    x, ids, dy = inputs(tokens, rank, device)
    strict, _ = exchanges(placement, device)

    native_out, native_dx = native_train(x, ids, dy, owner, local, size)
    seen: list[torch.Tensor] = []
    out, dx = candidate_train(strict, x, ids, dy, EpInvocation(0, 0, 0), seen=seen)
    direct_out, direct_dx = candidate_direct(strict, x, ids, dy, EpInvocation(0, 0, 1))
    same = same_bits(
        [
            (seen[0], dy),
            (out, native_out),
            (dx, native_dx),
            (direct_out, native_out),
            (direct_dx, native_dx),
        ],
        device,
    )
    if not all(same):
        raise AssertionError(f"train tokens={tokens}: native and candidate differ {same}")

    full = {"full": ("start", "end")}
    run = dict(device=device, rounds=args.rounds, warmup=args.warmup)
    row = header(tokens, size)
    row["transport"] = transport_scope(strict, x, ids, device, run)
    row["native_train"] = timed(
        lambda i, c: (native_train(x, ids, dy, owner, local, size), c.mark("end")),
        full,
        **run,
    )
    row["native_train_phases"] = timed(
        lambda i, c: native_train(x, ids, dy, owner, local, size, c),
        {
            "route": ("start", "route"),
            "dispatch": ("route", "dispatch"),
            "return": ("dispatch", "return"),
            "dy": ("return", "dy"),
            "dx": ("dy", "end"),
        },
        **run,
    )
    row["candidate_train"] = timed(
        lambda i, c: (
            candidate_train(strict, x, ids, dy, EpInvocation(1, i, 0)),
            c.mark("end"),
        ),
        full,
        **run,
    )
    row["candidate_train_phases"] = timed(
        lambda i, c: candidate_train(strict, x, ids, dy, EpInvocation(2, i, 0), c),
        {
            "dispatch": ("start", "dispatch"),
            "return": ("dispatch", "return"),
            "backward_preflight": ("loss", "dy_start"),
            "dy": ("dy_start", "dy_end"),
            "dx": ("dx_start", "dx_end"),
            "backward": ("loss", "end"),
        },
        **run,
    )
    row["candidate_direct"] = timed(
        lambda i, c: (
            candidate_direct(strict, x, ids, dy, EpInvocation(3, i, 0)),
            c.mark("end"),
        ),
        full,
        **run,
    )
    row["candidate_build_route"] = timed(
        lambda i, c: (
            strict.exchange.build_route(ids, EpIdentity(f"bench/route/{i}", "bench", "bench")),
            c.mark("end"),
        ),
        full,
        **run,
    )
    row["peak_allocated_bytes_max_rank"] = peak_memory(device)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="*", default=[64, 1024, 8192, 32768])
    parser.add_argument("--forward-tokens", type=int, nargs="*", default=DECODE_AND_PREFILL)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.backend == "nccl":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(args.backend, timeout=timedelta(minutes=10))
    rank, size = dist.get_rank(), dist.get_world_size()
    if args.backend == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    placement = EpPlacement.from_owners([e * size // NUM_EXPERTS for e in range(NUM_EXPERTS)], size)
    owner = placement.expert_owner.to(device)
    local = placement.expert_local.to(device)
    report = {
        "ep_size": size,
        "backend": args.backend,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "host": platform.node(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": (".".join(map(str, torch.cuda.nccl.version())) if device.type == "cuda" else None),
        "rounds": args.rounds,
        "warmup": args.warmup,
        "timing": "slowest rank per round; first round after warm-up reported apart",
        "routing": "uniform random over all experts, contiguous placement",
    }
    for key, tokens_list, measure in (
        ("forward", args.forward_tokens, measure_forward),
        ("train", args.tokens, measure_train),
    ):
        rows = []
        for tokens in tokens_list:
            row = measure(tokens, placement, owner, local, device, args)
            rows.append(row)
            if rank == 0:
                summary = {
                    k: {p: round(v["median_ms"], 3) for p, v in row[k].items()}
                    for k in row
                    if isinstance(row[k], dict)
                }
                print(key, tokens, json.dumps(summary), flush=True)
        report[key] = rows

    if rank == 0 and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
