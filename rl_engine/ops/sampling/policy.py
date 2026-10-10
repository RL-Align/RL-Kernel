# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Complete top-k/top-p support, independent of API top-logprob limits.

The support follows vLLM's native ascending-sort contract on the real
vocabulary. Only the boolean selection is detached; scoring retains autograd.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


@torch.no_grad()
def sampling_keep_mask(logits, *, temperature=1.0, top_p=None, top_k=None):
    """Return the full support for scalar or per-row sampling parameters."""
    if logits.ndim != 2:
        raise ValueError("sampling logits must be [rows, real_vocab]")
    # Conversion already owns new storage for BF16/FP16 inputs. A single
    # explicit copy also keeps FP32 callers safe from the in-place masking.
    values = logits.detach().to(dtype=torch.float32, copy=True)
    temp = torch.as_tensor(temperature, dtype=torch.float32, device=values.device)
    greedy = temp < 1e-5
    values.div_(torch.where(greedy, 1.0, temp).reshape(-1, 1))
    if top_p is None and top_k is None:
        return torch.isfinite(values)
    sorted_values, ids = values.sort(dim=-1, descending=False)
    if top_k is not None:
        k = torch.as_tensor(top_k, dtype=torch.long, device=values.device).reshape(-1)
        k = torch.where(k <= 0, values.size(1), k).clamp(max=values.size(1))
        k = k.expand(values.size(0))
        threshold = sorted_values.gather(1, (values.size(1) - k).unsqueeze(1))
        sorted_values.masked_fill_(sorted_values < threshold, float("-inf"))
    if top_p is not None:
        p = torch.as_tensor(top_p, dtype=torch.float32, device=values.device)
        probabilities = sorted_values.softmax(dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        removed = cumulative <= 1 - p.reshape(-1, 1)
        removed[:, -1] = False
        sorted_values.masked_fill_(removed, float("-inf"))
    if values.is_cuda and torch.version.hip is None:
        from rl_engine.backends.cuda.sampling.unique_scatter import scatter_permuted_columns

        mask = scatter_permuted_columns(torch.isfinite(sorted_values), ids)
    else:
        mask = torch.zeros_like(values, dtype=torch.bool).scatter_(
            1, ids, torch.isfinite(sorted_values)
        )
    return torch.where(greedy.reshape(-1, 1), torch.isfinite(values), mask)


@torch.no_grad()
def vocab_parallel_sampling_keep_mask(
    local_logits,
    *,
    real_vocab_size,
    tp_group=None,
    temperature=1.0,
    top_p=None,
    top_k=None,
    active_rows=None,
    chunk_size=None,
):
    """Compute global support in bounded chunks, then retain this TP shard.

    TP ranks have identical token ownership. CP groups deliberately do not
    participate: their token rows are independent. Inactive packed/prompt
    rows stay unmasked so discarded targets do not introduce infinities.
    """
    if chunk_size is None:
        # Each token row is independent. Amortize TP transport and sorting
        # dispatches on CUDA while bounding the larger vocabulary workspace.
        # Keep the existing ROCm/CPU workspace and explicit caller overrides.
        chunk_size = (
            max(32, min(128, (512 * 1024 * 1024) // (28 * real_vocab_size)))
            if local_logits.is_cuda and torch.version.hip is None and real_vocab_size > 0
            else 32
        )
    if chunk_size <= 0:
        raise ValueError("sampling mask chunk size must be positive")
    initialized = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size(tp_group) if initialized else 1
    rank = dist.get_rank(tp_group) if initialized else 0
    width = local_logits.size(1)
    if not 0 < real_vocab_size <= width * world:
        raise ValueError("real vocabulary must fit the TP shards")
    keep = torch.ones_like(local_logits, dtype=torch.bool)
    indices = (
        torch.arange(local_logits.size(0), device=local_logits.device)
        if active_rows is None
        else active_rows.nonzero().reshape(-1)
    )
    for chunk in indices.split(chunk_size):
        local = local_logits.detach().index_select(0, chunk).contiguous()
        if world > 1:
            shards = [torch.empty_like(local) for _ in range(world)]
            dist.all_gather(shards, local, group=tp_group)
            complete = torch.cat(shards, dim=1)[:, :real_vocab_size]
        else:
            complete = local[:, :real_vocab_size]
        mask = sampling_keep_mask(complete, temperature=temperature, top_p=top_p, top_k=top_k)
        start = rank * width
        count = max(0, min(width, real_vocab_size - start))
        if local.is_cuda and torch.version.hip is None and count == width:
            selected = mask[:, start : start + count]
        else:
            selected = torch.zeros_like(local, dtype=torch.bool)
            selected[:, :count] = mask[:, start : start + count]
        if keep.is_cuda and torch.version.hip is None and keep.stride(1) == 1:
            from rl_engine.backends.cuda.sampling.unique_scatter import copy_unique_rows_

            copy_unique_rows_(keep, chunk, selected)
        else:
            keep.index_copy_(0, chunk, selected)
    return keep
