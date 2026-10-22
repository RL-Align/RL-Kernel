# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Run the H3 parallel block on a TP x CP grid under ``srun`` (one task per GPU, any node count).

    srun -N2 --ntasks-per-node=4 python tools/validation/models/h3_parallel_run.py \\
        --tp 8 --cp 1 --out evidence/ws2_tp8

Every rank writes its rows (and gradients) to ``--out/rank{r}.pt``; rank 0 then recomputes the
single-GPU canonical block and compares the assembled forward, input gradient and weight-gradient
shards bitwise, and writes ``report.json``. ``--local-size`` is the number of GPUs per node for a
TP group that spans nodes (selects ``hierarchical_ipc_nccl_tree``); a TP group inside one node
uses the CUDA-IPC fixed tree.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist

from rl_engine.validation.models import h3_tp
from rl_engine.validation.models.h3_weights import load_h3_block_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--layout", default="tiny")
    parser.add_argument("--local-size", type=int, default=4, help="GPUs per node")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(local)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ.get('MASTER_PORT', '29511')}",
        rank=rank,
        world_size=world,
        device_id=torch.device("cuda", local),
    )
    if world != args.tp * args.cp:
        raise SystemExit(f"world {world} != tp {args.tp} x cp {args.cp}")
    probe = torch.ones(1, device="cuda") * rank
    dist.all_reduce(probe)  # NCCL connectivity across nodes
    spans_nodes = args.tp > args.local_size
    grid = h3_tp.make_grid(args.tp, args.cp, local_size=args.local_size if spans_nodes else None)
    weights = load_h3_block_weights("cuda")
    kwargs = dict(sp=args.sp, seq_len=args.seq_len, layout=args.layout)
    result = h3_tp.block_case(weights, grid, backward=True, **kwargs)
    # timing of the forward alone
    torch.cuda.synchronize()
    dist.barrier()
    begin = time.perf_counter()
    for _ in range(args.iters):
        h3_tp.block_case(weights, grid, backward=False, **kwargs)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - begin) / args.iters
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in result.items()}, args.out / f"rank{rank}.pt")
    dist.barrier()
    timings = [None] * world
    dist.all_gather_object(
        timings, {"rank": rank, "host": socket.gethostname(), "forward_s": seconds}
    )
    if rank == 0:
        ref = h3_tp.block_case(
            weights, h3_tp.Grid(1, 1, 0, 0), backward=True, **{**kwargs, "sp": False}
        )
        ranks = [torch.load(args.out / f"rank{r}.pt") for r in range(world)]

        def assemble(key):
            pieces = {}
            for r in ranks:
                span = tuple(int(v) for v in r["rows"])
                if span in pieces and not torch.equal(pieces[span], r[key]):
                    return None
                pieces[span] = r[key]
            return torch.cat([pieces[s] for s in sorted(pieces)], dim=1)

        out, dx = assemble("out"), assemble("dx")
        checks = {
            "forward_bitwise": out is not None and torch.equal(out, ref["out"].cpu()),
            "input_grad_bitwise": dx is not None and torch.equal(dx, ref["dx"].cpu()),
        }
        if args.cp == 1 and not args.sp:
            for name, axis in (
                ("attn.to_out.0.weight", 1),
                ("ff.net.2.weight", 1),
                ("attn.to_q.weight", 0),
                ("attn.to_k.weight", 0),
                ("attn.to_v.weight", 0),
                ("ff.hidden", 0),
                ("ff.gate", 0),
            ):
                shards = [r["dw." + name] for r in ranks]
                checks["dw." + name] = torch.equal(
                    torch.cat(shards, dim=axis), ref["dw." + name].cpu()
                )
        report = dict(
            tp=args.tp,
            cp=args.cp,
            sp=args.sp,
            world=world,
            hosts=sorted({t["host"] for t in timings}),
            collective="hierarchical_ipc_nccl_tree" if spans_nodes else "cuda_ipc_fixed_tree",
            seq_len=int(ref["out"].shape[1]) if not args.seq_len else args.seq_len,
            checks=checks,
            all_bitwise=all(checks.values()),
            forward_seconds_per_rank=timings,
            torch=torch.__version__,
            device=torch.cuda.get_device_name(),
        )
        (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in ("tp", "cp", "sp", "hosts", "collective", "checks", "all_bitwise")
                },
                indent=1,
            )
        )
    dist.barrier()
    if rank == 0:
        time.sleep(3)  # keep the rendezvous store up until every rank has left
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
