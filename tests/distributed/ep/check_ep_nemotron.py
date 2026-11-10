# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Nemotron EP on a real process group; every rank compares bits with EP size 1 locally.

    PYTHONPATH=. torchrun --nproc-per-node=8 --master-addr=127.0.0.1 --master-port=29500 \
        tests/distributed/ep/check_ep_nemotron.py

``--scale`` adds the F6 cases, which need many GPU gigabytes. The router cases need a CUDA
SM90 device. A failure raises on the rank that sees it; a hang means a real defect, so run
this under ``timeout``.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from rl_engine.distributed.ep import EpPlacement, TorchAllToAll
from rl_engine.distributed.ep.cuda_transport import CudaNcclAllToAll
from rl_engine.distributed.ep.nemotron import MAX_TOKENS, NUM_EXPERTS, TOP_K, EpInvocation
from tests.distributed.ep.nemotron_support import (
    H,
    LoopbackTransport,
    bits,
    contiguous_placement,
    expert_scales,
    global_inputs,
    make_exchange,
    rank_slice,
    reference_forward,
    run_layer,
    shuffled_placement,
    split_sizes,
)


class GlooAllToAll(TorchAllToAll):
    """CPU smoke of this script only; says nothing about NCCL."""

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


def expect_rejected(name, body, needle):
    try:
        body()
    except Exception as exc:  # noqa: BLE001
        assert needle in str(exc), f"{name}: {exc}"
        return
    raise AssertionError(f"{name}: not rejected")


class Check:
    def __init__(self, args):
        self.args = args
        self.rank, self.ep = dist.get_rank(), dist.get_world_size()
        # The exchange compares devices exactly; a bare "cuda" is not "cuda:0".
        if args.backend == "nccl":
            self.device = torch.device("cuda", torch.cuda.current_device())
        else:
            self.device = torch.device("cpu")
        self.scales = expert_scales().to(self.device)
        self.results = {}

    def transport(self):
        return CudaNcclAllToAll(None) if self.args.backend == "nccl" else GlooAllToAll()

    def exchange(self, placement=None):
        placement = contiguous_placement(self.ep) if placement is None else placement
        return make_exchange(placement, self.transport())

    def local_inputs(self, total, sizes):
        return [rank_slice(t, sizes, self.rank).to(self.device) for t in global_inputs(total)]

    def forward(self, exchange, x, ids, weights, target):
        with torch.no_grad():
            return run_layer(exchange, x, ids, weights, target, self.scales)[1]

    def ep1(self, x, ids, weights, target):
        exchange = make_exchange(contiguous_placement(1), LoopbackTransport(self.device))
        return self.forward(exchange, x, ids, weights, target)

    def compare(self, name, placement, sizes):
        x, ids, weights, target = self.local_inputs(self.args.tokens, sizes)
        out = self.forward(self.exchange(placement), x, ids, weights, target)
        assert torch.equal(bits(out), bits(self.ep1(x, ids, weights, target))), name
        self.results[name] = "pass"

    def f1_topologies(self):
        tokens = self.args.tokens
        balanced = split_sizes(tokens, self.ep, empty_rank=None)
        cases = {
            "f1_contiguous": (contiguous_placement(self.ep), balanced),
            "f4_shuffled": (shuffled_placement(self.ep), balanced),
            "f4_empty_source": (
                contiguous_placement(self.ep),
                split_sizes(tokens, self.ep),
            ),
            "f4_one_owner": (
                EpPlacement.from_owners([0] * NUM_EXPERTS, self.ep),
                balanced,
            ),
            "f4_one_source": (
                contiguous_placement(self.ep),
                [tokens] + [0] * (self.ep - 1),
            ),
        }
        for name, (placement, sizes) in cases.items():
            self.compare(name, placement, sizes)

    def f1_reference(self):
        sizes = split_sizes(self.args.tokens, self.ep)
        x, ids, weights, target = self.local_inputs(self.args.tokens, sizes)
        with torch.no_grad():
            _, out, _ = run_layer(self.exchange(), x, ids, weights, target, self.scales)
        ref = reference_forward(x, ids, weights, self.scales)
        assert torch.equal(bits(out), bits(ref)), "f1: reference"
        self.results["f1_reference"] = "pass"

    def f2_position(self):
        tokens = self.args.tokens
        x, ids, weights, target = global_inputs(tokens)
        perm = torch.randperm(tokens, generator=torch.Generator().manual_seed(3))
        sizes = split_sizes(tokens, self.ep, empty_rank=None)
        shuffled = (x[perm], ids[perm], weights[perm], target[perm])
        local = [rank_slice(t, sizes, self.rank).to(self.device) for t in shuffled]
        out = self.forward(self.exchange(), *local)
        one = perm[sum(sizes[: self.rank]) : sum(sizes[: self.rank + 1])]
        ref_out = []
        for token in one.tolist():
            single = [t[token : token + 1].to(self.device) for t in (x, ids, weights, target)]
            ref_out.append(self.ep1(*single))
        assert torch.equal(bits(out), bits(torch.cat(ref_out))), "f2: forward"
        self.results["f2_batch_and_position"] = "pass"

    def f3_bit_patterns(self):
        patterns = torch.tensor(
            [0x7FC0, 0xFFC1, 0x7F80, 0xFF80, 0x8000, 0x0000, 0x0001, 0x8001],
            dtype=torch.uint16,
        ).view(torch.int16)
        total = 16
        x_bits = patterns[torch.arange(total * H) % patterns.numel()].reshape(total, H)
        sizes = split_sizes(total, self.ep)
        x = rank_slice(x_bits, sizes, self.rank).to(self.device)
        ids = rank_slice(global_inputs(total)[1], sizes, self.rank).to(self.device)
        with torch.no_grad():
            call = self.exchange().dispatch(x.view(torch.bfloat16), ids, EpInvocation(0, 0, 0))
            rows = call.return_outputs(call.x_rows).view(-1, TOP_K, H)
        assert torch.equal(rows.view(torch.int16), x[:, None].expand(-1, TOP_K, H))
        self.results["f3_bit_patterns"] = "pass"

    def f7_faults(self):
        tokens = self.args.tokens
        sizes = split_sizes(tokens, self.ep, empty_rank=None)
        x, ids, *_ = self.local_inputs(tokens, sizes)
        bad = self.ep - 1

        def wrong_dtype():
            with torch.no_grad():
                self.exchange().dispatch(
                    x.float() if self.rank == bad else x, ids, EpInvocation(0, 0, 0)
                )

        if self.ep > 1:
            expect_rejected("f7_wrong_dtype", wrong_dtype, "failed")
            self.results["f7_one_rank_wrong_dtype"] = "pass"

    def router_comparison(self):
        """F5 against the real SM90 router, one local device, EP size 1."""
        if self.args.backend != "nccl":
            self.results["f5_router"] = "skipped: needs SM90"
            return
        from rl_engine.runtime.registry import KernelRegistry

        tokens = self.args.tokens
        g = torch.Generator().manual_seed(21)
        x = torch.randn(tokens, H, generator=g).to(self.device, torch.bfloat16)
        w = torch.randn(NUM_EXPERTS, H, generator=g).to(self.device, torch.float32)
        bias = torch.zeros(NUM_EXPERTS, device=self.device)
        with torch.no_grad():
            op = KernelRegistry().get_op("nemotron_router_dispatch", device=x.device)
            ids, _, _, _, packed = op(x, w, bias)
            exchange = make_exchange(contiguous_placement(1), LoopbackTransport(self.device))
            call = exchange.dispatch(x, ids, EpInvocation(0, 0, 0))
        assert torch.equal(bits(call.x_rows), bits(packed)), "f5"
        self.results["f5_router_packed"] = "pass"

    def f6_bits(self, rank):
        g = torch.Generator(device=self.device).manual_seed(31 + rank)
        shape = (MAX_TOKENS, H)
        return torch.randint(
            -(2**15), 2**15, shape, generator=g, device=self.device, dtype=torch.int16
        )

    @staticmethod
    def f6b_pattern(rows, rank, device):
        """Distinct per row and column, so a short or shifted copy cannot match."""
        r = torch.arange(rows, dtype=torch.int32, device=device)[:, None] * 2689
        c = torch.arange(H, dtype=torch.int32, device=device)[None, :]
        return (r + c + rank * 31).to(torch.int16)

    def f6_scale(self):
        """Largest legal Nemotron load, all into one receiver, and a >2 GiB pair on the wire."""
        transport = self.transport()
        placement = EpPlacement.from_owners([0] * NUM_EXPERTS, self.ep)
        ids = torch.arange(TOP_K, device=self.device).expand(MAX_TOKENS, TOP_K)
        x_bits = self.f6_bits(self.rank)
        with torch.no_grad():
            call = make_exchange(placement, transport).dispatch(
                x_bits.view(torch.bfloat16), ids, EpInvocation(0, 0, 0)
            )
            if self.rank == 0:
                # Expert e is slot e of every token: (local expert, source, token) order.
                received = call.x_rows.view(torch.int16).view(TOP_K, self.ep, MAX_TOKENS, H)
                for source in range(self.ep):
                    expected = self.f6_bits(source)
                    for expert in range(TOP_K):
                        assert torch.equal(received[expert, source], expected), "f6: arrival"
                    del expected
                del received
            rows = call.return_outputs(call.x_rows).view(-1, TOP_K, H)
            assert torch.equal(rows.view(torch.int16), x_bits[:, None].expand(-1, TOP_K, H))
        pair = MAX_TOKENS * TOP_K * H * 2
        self.results["f6_max_legal_scale"] = {
            "pair_bytes": pair,
            "source_bytes": pair,
            "receiver_bytes": pair * self.ep,
        }
        del call, rows, x_bits

        if self.ep > 1:
            rows = 66577 * TOP_K
            peer = self.rank ^ 1
            send_counts = [rows if r == peer else 0 for r in range(self.ep)]
            payload = self.f6b_pattern(rows, self.rank, self.device)
            output = torch.full_like(payload, -1)
            transport.all_to_all(
                output.view(torch.bfloat16),
                payload.view(torch.bfloat16),
                send_counts,
                send_counts,
            )
            del payload
            expected = self.f6b_pattern(rows, peer, self.device)
            assert torch.equal(output, expected), "f6b: pair above 2 GiB"
            self.results["f6b_pair_above_2gib"] = {"pair_bytes": rows * H * 2}
        if self.args.backend == "nccl":
            peak = torch.tensor([torch.cuda.max_memory_allocated()], device=self.device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            self.results["f6_peak_allocated_bytes_max_rank"] = int(peak.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--scale", action="store_true", help="add the large F6 cases")
    args = parser.parse_args()

    if args.backend == "nccl":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(args.backend, timeout=timedelta(minutes=5))
    check = Check(args)
    check.f1_reference()
    check.f1_topologies()
    check.f2_position()
    check.f3_bit_patterns()
    check.f7_faults()
    check.router_comparison()
    if args.scale:
        check.f6_scale()
    dist.barrier()
    if check.rank == 0:
        print(json.dumps({"ep": check.ep, "backend": args.backend, **check.results}, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
