"""torchrun: interleaved RS/AG/AR and graph replay without host barriers."""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from rl_engine.distributed import create_deterministic_collective


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    world = dist.get_world_size()
    with create_deterministic_collective(max_size_bytes=16 * 1024 * 1024) as collective:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for rows, offset in ((1, 1), (257, 1), (32769, 1), (32768, 0), (32769, 0)):
                g = torch.Generator(device="cuda").manual_seed(29)
                leaves = torch.randn(
                    world, world * rows, 7, generator=g, device="cuda", dtype=dtype
                )
                x = leaves[rank].clone()
                storage = torch.empty(rows * 7 + 1, device="cuda", dtype=dtype)
                shard = storage[offset : offset + rows * 7].view(rows, 7)
                full = torch.empty_like(x)
                reduced = torch.empty_like(x)

                def operations():
                    collective.reduce_scatter(x, out=shard, validate_signature=False)
                    collective.all_gather(shard, out=full, validate_signature=False)
                    collective.all_reduce(x, out=reduced, validate_signature=False)

                operations()
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(3):
                        operations()
                for replay in range(6):
                    changed = (leaves.float() + replay * 0.125).to(dtype)
                    parts = list(changed.unbind())
                    while len(parts) > 1:
                        parts = [parts[i] + parts[i + 1] for i in range(0, len(parts), 2)]
                    x.copy_(changed[rank])
                    torch.cuda._sleep(rank * 10000)
                    if replay % 2:
                        graph.replay()
                    else:
                        operations()
                    assert torch.equal(full.view(torch.uint8), parts[0].view(torch.uint8))
                    assert torch.equal(reduced.view(torch.uint8), parts[0].view(torch.uint8))
                    assert torch.equal(
                        shard.view(torch.uint8), parts[0].chunk(world)[rank].view(torch.uint8)
                    )
                if rank == 0:
                    print(
                        json.dumps(
                            dict(
                                world=world,
                                dtype=str(dtype),
                                rows=rows,
                                offset=offset,
                                bitwise_equal=True,
                            )
                        ),
                        flush=True,
                    )
                dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
