"""Run with torchrun and the matching torch-memory-saver LD_PRELOAD library."""
import os
import sys

import torch
import torch.distributed as dist
from torch_memory_saver import torch_memory_saver as saver

from rl_engine.distributed.collectives import DeterministicCollective

if "--with-vime" in sys.argv:
    from vime.backends.megatron_utils.actor import _configure_train_offload

    _configure_train_offload()

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
with saver.disable():
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    dist.barrier()
# Match the training CP arena and exercise lazy initialization after offload.
x = torch.full((256,), rank + 1.0, device="cuda")
torch.cuda.synchronize()
saver.pause()
saver.resume()
collective = DeterministicCollective(max_size_bytes=64 * 1024 * 1024)
x = torch.full((256,), rank + 1.0, device="cuda")
for phase in range(2):
    if phase:
        torch.cuda.synchronize()
        saver.pause()
        saver.resume()
    result = collective.all_reduce(x)
    assert torch.equal(result, torch.full_like(x, 36.0))
    dist.barrier()
if rank == 0:
    print("OFFLOAD_IPC_RESULT=PASS before and after pause/resume", flush=True)
collective.close()

# Ray can enter a later call with hooks disabled while the default allocator
# still caches offloadable VMM blocks from a previous call. Reproduce that
# state before creating another IPC arena; toggling the hook alone is unsafe.
cached = torch.zeros(3 * (64 * 1024 * 1024 + 32), dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()
del cached
binary = saver._impl._binary_wrapper.cdll
binary.tms_set_interesting_region(False)
try:
    inactive = DeterministicCollective(max_size_bytes=64 * 1024 * 1024)
    result = inactive.all_reduce(x)
    assert torch.equal(result, torch.full_like(x, 36.0))
    dist.barrier()
    inactive.close()
finally:
    binary.tms_set_interesting_region(True)
if rank == 0:
    print("OFFLOAD_IPC_INACTIVE_RESULT=PASS with cached VMM allocations", flush=True)
dist.destroy_process_group()
