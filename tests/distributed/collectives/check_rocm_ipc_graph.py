# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Run with torchrun --standalone --nproc-per-node=8 on ROCm.

Exercise IPC generations across mixed operations, changing graph inputs,
uneven rank progress, and eager/graph transitions using raw-bit references.
"""

import os
import time

import torch
import torch.distributed as dist

from rl_engine import _C
from rl_engine.distributed.transports.rccl import RCCLDeterministicCollective


def check_group(collective, tp, rank):
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for count in (8, 4104, 32768):
            x = torch.empty(count, device="cuda", dtype=dtype)
            collective.prepare_direct_staging_views([(count,)], dtype=dtype)
            staging = collective.direct_staging_view((count,), dtype=dtype)
            summed = torch.empty_like(x)
            staged_sum = torch.empty_like(x)
            gathered = torch.empty(count * tp, device="cuda", dtype=dtype)
            scattered = torch.empty(count // tp, device="cuda", dtype=dtype)

            def run(
                x=x,
                summed=summed,
                gathered=gathered,
                staged_sum=staged_sum,
                scattered=scattered,
                staging=staging,
            ):
                if tp == 1:
                    # The single-rank implementation does not allocate an IPC handle.
                    collective.all_reduce(x, out=summed)
                    collective.all_gather(x, out=gathered)
                    collective.all_reduce(x, out=staged_sum)
                    collective.reduce_scatter(x, out=scattered)
                    return
                handle = collective._handle
                _C.deterministic_collective_rocm_ipc_all_reduce_input(handle, x, summed)
                _C.deterministic_collective_rocm_ipc_all_gather_input(handle, x, gathered)
                _C.deterministic_collective_rocm_ipc_prepare_staged(handle, staging)
                staging.copy_(x)
                _C.deterministic_collective_rocm_ipc_all_reduce_staged(handle, staging, staged_sum)
                _C.deterministic_collective_rocm_ipc_reduce_scatter_input(handle, x, scattered)

            x.fill_(rank % tp + 1)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graphs = []
            for repeats in (1, 5):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(repeats):
                        run()
                graphs.append(graph)
            for iteration in range(12):
                generator = torch.Generator().manual_seed(987 + iteration)
                peers = [
                    torch.randn(count, generator=generator).to(device="cuda", dtype=dtype)
                    for _ in range(tp)
                ]
                for index, peer in enumerate(peers):
                    peer[0] = -0.0
                    peer[1] = (-1.0) ** index * 128
                expected_gather = torch.cat(peers)
                x.copy_(peers[rank % tp])
                while len(peers) > 1:
                    peers = [peers[index] + peers[index + 1] for index in range(0, len(peers), 2)]
                if rank % tp == iteration % tp:
                    time.sleep(0.001)
                if iteration % 3 == 0:
                    run()
                else:
                    graphs[iteration % 2].replay()
                bits = torch.int32 if dtype == torch.float32 else torch.int16
                for actual, expected in (
                    (summed, peers[0]),
                    (staged_sum, peers[0]),
                    (gathered, expected_gather),
                    (scattered, peers[0].chunk(tp)[rank % tp]),
                ):
                    assert torch.equal(actual.view(bits), expected.view(bits)), (
                        tp,
                        dtype,
                        count,
                        iteration,
                    )


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    world = dist.get_world_size()
    assert torch.version.hip is not None and world in (1, 2, 4, 8)
    for tp in (1, 2, 4, 8):
        if tp > world:
            continue
        for start in range(0, world, tp):
            group = dist.new_group(list(range(start, start + tp)))
            if start <= dist.get_rank() < start + tp:
                own_group = group
        with RCCLDeterministicCollective(
            group=own_group, device=device, max_size_bytes=1024 * 1024
        ) as collective:
            check_group(collective, tp, dist.get_rank())
            dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("ROCM_IPC_MIXED_GRAPH_RESULT=PASS", flush=True)


if __name__ == "__main__":
    main()
