"""torchrun --nproc-per-node=4: bitwise graph replay and message-size latency."""

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
    device = torch.device("cuda", rank)
    with create_deterministic_collective(max_size_bytes=64 * 2**20) as collective:
        for length in (
            1,
            257,
            4096,
            16384,
            65537,
            131071,
            131073,
            262144,
            1048576,
            2097153,
            4194304,
            8388608,
            16777216,
        ):
            for misaligned in (
                (False, True) if os.environ.get("CHECK_MISALIGNED", "1") == "1" else (False,)
            ):
                generator = torch.Generator(device=device).manual_seed(1234)
                leaves = torch.randn(
                    world, length, device=device, dtype=torch.bfloat16, generator=generator
                )
                x = leaves[rank].clone()
                storage = torch.empty(length + int(misaligned), device=device, dtype=torch.bfloat16)
                output = storage[int(misaligned) :]
                collective.all_reduce(x, out=output, validate_signature=False)
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(16):
                        collective.all_reduce(x, out=output, validate_signature=False)
                for replay in range(5):
                    changed = (leaves.float() + replay * 0.125).to(torch.bfloat16)
                    x.copy_(changed[rank])
                    graph.replay()
                    parts = list(changed.unbind())
                    while len(parts) > 1:
                        parts = [parts[i] + parts[i + 1] for i in range(0, len(parts), 2)]
                    torch.cuda.synchronize()
                    assert torch.equal(output.view(torch.int16), parts[0].view(torch.int16)), (
                        length,
                        misaligned,
                        replay,
                        rank,
                    )
                dist.barrier()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                begin.record()
                for _ in range(20):
                    graph.replay()
                end.record()
                end.synchronize()
                if rank == 0:
                    print(
                        json.dumps(
                            dict(
                                world=world,
                                elements=length,
                                misaligned=misaligned,
                                graph_us=begin.elapsed_time(end) * 1000 / 320,
                                bitwise_equal=True,
                            )
                        ),
                        flush=True,
                    )
                dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
