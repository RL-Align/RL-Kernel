import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from rl_engine.integrations.vllm_pcp import PagedContextParallel
from rl_engine.kernels.ops.cuda.attention.strict_runtime import StrictCUDAAttentionRuntime

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
runtime = StrictCUDAAttentionRuntime()
records = []
for world in (2, 4, 8):
    own = None
    for start in range(0, 8, world):
        group = dist.new_group(list(range(start, start + world)))
        if start <= rank < start + world:
            own = group
    coordinator = SimpleNamespace(
        world_size=world, rank_in_group=rank % world, device_group=own, cpu_group=own
    )
    for interleave in (1, 8, 16):
        cp = PagedContextParallel(coordinator, interleave)
        cp.bind(2 * 3 * 128 * 2 * 128 * 2)
        for qlens in ([13, 7, 1], [1, 1, 1]):
            heads, kvheads, dim, block = 8, 2, 128, 16
            lengths = torch.tensor([33, 79, 6], device="cuda", dtype=torch.int32)
            starts = torch.tensor(
                [0, qlens[0], sum(qlens[:2]), sum(qlens)], device="cuda", dtype=torch.int32
            )
            maximum = 79
            fullpages = (maximum + 15) // 16
            localpages = (fullpages + world - 1) // world
            total = localpages * block * world
            generator = torch.Generator(device="cuda").manual_seed(29)
            k = torch.randn(
                3, total, kvheads, dim, generator=generator, device="cuda", dtype=torch.bfloat16
            )
            v = torch.randn(k.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
            valid = torch.arange(total, device="cuda")[None, :] < lengths[:, None]
            k = torch.where(valid[:, :, None, None], k, 0)
            v = torch.where(valid[:, :, None, None], v, 0)
            pos = torch.arange(localpages * block, device="cuda")
            ids = ((pos // interleave) * world + cp.rank) * interleave + pos % interleave
            kc = k[:, ids].reshape(3 * localpages, block, kvheads, dim).contiguous()
            vc = v[:, ids].reshape_as(kc).contiguous()
            localtable = torch.arange(3 * localpages, device="cuda", dtype=torch.int32).reshape(
                3, localpages
            )
            q = torch.randn(
                sum(qlens), heads, dim, generator=generator, device="cuda", dtype=torch.bfloat16
            )
            metadata = SimpleNamespace(
                num_actual_tokens=q.size(0),
                seq_lens=lengths,
                query_start_loc=starts,
                max_seq_len=maximum,
            )
            impl = SimpleNamespace(
                num_heads=heads, num_kv_heads=kvheads, head_size=dim, scale=dim**-0.5
            )
            fullk = (
                k[:, : fullpages * block].reshape(3 * fullpages, block, kvheads, dim).contiguous()
            )
            fullv = v[:, : fullpages * block].reshape_as(fullk).contiguous()
            restoredk, restoredv, table = cp.materialize(kc, vc, localtable, lengths, maximum)
            for i, length in enumerate(lengths.tolist()):
                assert torch.equal(
                    restoredk.reshape(3, -1, kvheads, dim)[i, :length], k[i, :length]
                )
                assert torch.equal(
                    restoredv.reshape(3, -1, kvheads, dim)[i, :length], v[i, :length]
                )
            rows = torch.arange(q.size(0), device="cuda", dtype=torch.int32)
            req = torch.searchsorted(starts[1:], rows, right=True).long()
            used = (lengths[req] - starts[1:][req] + rows + 1).int()
            expected = runtime.forward_paged_with_lse(
                q.unsqueeze(2),
                fullk,
                fullv,
                page_table=table[req].contiguous(),
                seqused_k=used,
                max_seqlen_k=fullpages * block,
                scale=impl.scale,
            ).out.squeeze(2)
            out = torch.empty_like(q)
            for _ in range(3):
                cp.forward(runtime, impl, q, out, metadata, kc, vc, localtable)
            assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8)), (
                rank,
                world,
                interleave,
                qlens,
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                cp.forward(runtime, impl, q, out, metadata, kc, vc, localtable)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8)), (
                "graph",
                rank,
                world,
                interleave,
                qlens,
            )
            records.append(
                {
                    "cp": world,
                    "interleave": interleave,
                    "queries": sum(qlens),
                    "mismatches": 0,
                    "cuda_graph": True,
                }
            )
            graph.reset()
        dist.barrier(group=own)
    dist.barrier()
    dist.destroy_process_group(own)
if rank == 0:
    print(json.dumps({"passed": len(records), "cases": records}), flush=True)
dist.destroy_process_group()
