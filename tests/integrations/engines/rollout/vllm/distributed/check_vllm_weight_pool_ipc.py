import faulthandler
import gc
import os

import torch
import torch.distributed as dist
from vllm.device_allocator.cumem import CuMemAllocator

from rl_engine.distributed.algorithms.collectives import collective_for_group

rank = int(os.environ["LOCAL_RANK"])
faulthandler.enable()
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
allocator = CuMemAllocator.get_instance()
collective = collective_for_group(dist.group.WORLD)
with allocator.use_memory_pool(tag="weights"):
    weights = torch.full((1024,), rank + 1.0, device="cuda")
    assert collective_for_group(dist.group.WORLD) is collective
    after = torch.ones(1024, device="cuda")
assert allocator.get_current_usage() > 0
before = weights.clone()
payload = torch.full((4,), float(rank), device="cuda")
expected = torch.arange(dist.get_world_size(), device="cuda").float().repeat_interleave(4)
assert torch.equal(collective.all_gather(payload), expected)
allocator.sleep(offload_tags=("weights",))
assert torch.equal(collective.all_gather(payload), expected)
allocator.wake_up(tags=["weights"])
assert torch.equal(weights, before)
assert torch.equal(after, torch.ones_like(after))
assert torch.equal(collective.all_gather(payload), expected)
dist.barrier()
if rank == 0:
    print("PASS: weight-pool sleep/wake preserves weights and resident IPC", flush=True)
# Release tagged tensors while allocator callbacks and CUDA are still alive.
# Leaving them to interpreter teardown can unload vLLM's callback library first.
del weights, after
gc.collect()
torch.cuda.synchronize()
collective.close()
retained_allocators = [entry[1] for entry in allocator.allocator_and_pools.values()]
allocator.allocator_and_pools.clear()
gc.collect()
dist.destroy_process_group()
print(f"rank {rank}: cleanup complete", flush=True)
