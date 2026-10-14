# SPDX-License-Identifier: Apache-2.0
"""Strict attention over vLLM's interleaved context-parallel KV cache.

KV storage is sharded by logical token position. Attention queries are divided
between CP ranks; outputs are gathered without any floating-point reduction.
Every query sees the same logical KV order as CP1.
"""
import torch


def restore_interleaved_pages(gathered, *, interleave, block_size, page_count):
    """[CP, K/V, requests, pages, block, heads, dim] -> ordinary paged KV."""
    cp, planes, requests, pages, block, heads, dim = gathered.shape
    if block != block_size or block % interleave:
        raise ValueError("CP cache interleave must divide the physical block size")
    ordered = gathered.reshape(
        cp, planes, requests, pages, block // interleave, interleave, heads, dim
    )
    ordered = ordered.permute(1, 2, 3, 4, 0, 5, 6, 7).contiguous()
    ordered = ordered.reshape(planes, requests, pages * cp, block, heads, dim)
    ordered = ordered[:, :, :page_count].contiguous()
    return ordered.reshape(planes, requests * page_count, block, heads, dim).unbind(0)


class PagedContextParallel:
    def __init__(self, coordinator, interleave=1):
        self.world = coordinator.world_size
        self.rank = coordinator.rank_in_group
        self.group = coordinator.device_group
        self.interleave = int(interleave)
        self.collective = None
        if self.world < 2 or self.interleave < 1:
            raise ValueError("PCP requires multiple ranks and a positive cache interleave")

    def bind(self, max_size_bytes):
        # Exactly the transport used by CUDAAGRSAttentionCPCommunication in
        # training TP4/CP2. Bind before vLLM starts CUDA Graph capture.
        from rl_engine.distributed.algorithms.collectives import collective_for_group

        self.collective = collective_for_group(self.group, min_size_bytes=max_size_bytes)

    def materialize(self, key_cache, value_cache, block_table, seq_lens, max_seq_len):
        block = key_cache.size(1)
        if block % self.interleave:
            raise ValueError("PCP interleave must divide KV block size")
        full_pages = max(1, (max_seq_len + block - 1) // block)
        local_pages = (full_pages + self.world - 1) // self.world
        if local_pages > block_table.size(1):
            raise ValueError("CP block table cannot cover the declared sequence length")
        requests = seq_lens.numel()
        page_ids = block_table[:requests, :local_pages].long()
        columns = torch.arange(local_pages, device=block_table.device)
        live_pages = columns[None, :] * block * self.world < seq_lens[:, None]
        page_ids = torch.where(live_pages, page_ids, 0)
        # Graph padding can reference unused pages; zero invalid token lanes
        # before communication so stale/uninitialized cache bytes cannot leak.
        positions = torch.arange(local_pages * block, device=block_table.device)
        logical = ((positions // self.interleave) * self.world + self.rank) * self.interleave
        logical += positions % self.interleave
        live = logical.reshape(1, local_pages, block) < seq_lens[:, None, None]
        planes = []
        for cache in (key_cache, value_cache):
            plane = cache.index_select(0, page_ids.flatten()).reshape(
                requests, local_pages, block, cache.size(2), cache.size(3)
            )
            planes.append(torch.where(live[..., None, None], plane, 0))
        if self.collective is None:
            self.bind(sum(p.numel() * p.element_size() for p in planes))
        # vLLM broadcasts the same scheduled batch to every PCP worker.
        # Its first graph-only metadata shape can appear during capture;
        # Python object collectives for signature checks cannot run there.
        gathered = self.collective.all_gather_many(
            tuple(planes), validate_signature=not torch.cuda.is_current_stream_capturing()
        )
        logical = []
        for plane in gathered:
            ordered = plane.reshape(
                self.world,
                requests,
                local_pages,
                block // self.interleave,
                self.interleave,
                key_cache.size(2),
                key_cache.size(3),
            )
            ordered = ordered.permute(1, 2, 3, 0, 4, 5, 6).contiguous()
            ordered = ordered.reshape(
                requests, local_pages * self.world, block, key_cache.size(2), key_cache.size(3)
            )
            logical.append(
                ordered[:, :full_pages]
                .contiguous()
                .reshape(requests * full_pages, block, key_cache.size(2), key_cache.size(3))
            )
        keys, values = logical
        pages = torch.arange(
            requests * full_pages, device=block_table.device, dtype=torch.int32
        ).reshape(requests, full_pages)
        return keys, values, pages

    def forward(self, runtime, impl, query, output, metadata, key_cache, value_cache, block_table):
        count = int(getattr(metadata, "num_actual_tokens", query.size(0)))
        if not count:
            return output.zero_()
        seq_lens = metadata.seq_lens
        maximum = int(
            getattr(metadata, "max_seq_len", block_table.size(1) * key_cache.size(1) * self.world)
        )
        keys, values, pages = self.materialize(
            key_cache, value_cache, block_table, seq_lens, maximum
        )
        width = (count + self.world - 1) // self.world
        begin = self.rank * width
        end = min(count, begin + width)
        local_output = torch.zeros(
            (width, int(impl.num_heads), int(impl.head_size)),
            dtype=query.dtype,
            device=query.device,
        )
        if begin < end:
            rows = torch.arange(begin, end, device=query.device, dtype=torch.int32)
            starts = metadata.query_start_loc.to(device=query.device, dtype=torch.int32)
            requests = torch.searchsorted(starts[1:], rows, right=True).long()
            requests = requests.clamp_max(seq_lens.numel() - 1)
            lengths = (
                seq_lens.index_select(0, requests) - starts[1:].index_select(0, requests) + rows + 1
            )
            lengths = torch.where(rows < starts[-1], lengths, 1).clamp_min(1).int()
            local_pages = pages.index_select(0, requests).contiguous()
            kwargs = dict(
                page_table=local_pages,
                seqused_k=lengths,
                max_seqlen_k=pages.size(1) * keys.size(1),
                scale=float(impl.scale),
            )
            if torch.version.hip is not None:
                kwargs.update(
                    return_lse=False,
                    page_table_validated=True,
                    cu_seqlens_q=torch.arange(
                        end - begin + 1, device=query.device, dtype=torch.int32
                    ),
                    kv_indptr=torch.arange(end - begin + 1, device=query.device, dtype=torch.int32)
                    * pages.size(1),
                )
            result = runtime.forward_paged_with_lse(
                query[begin:end].unsqueeze(2).contiguous(), keys, values, **kwargs
            )
            local_output[: end - begin].copy_(result.out.squeeze(2))
        gathered = self.collective.all_gather(
            local_output, validate_signature=not torch.cuda.is_current_stream_capturing()
        )
        output.view(output.size(0), int(impl.num_heads), int(impl.head_size))[:count].copy_(
            gathered[:count]
        )
        if output.size(0) > count:
            output[count:].zero_()
        return output
