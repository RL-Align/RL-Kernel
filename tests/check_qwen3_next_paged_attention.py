# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Paged-KV component tests; a real vLLM scheduler gate is still required."""

import pytest
import torch

from rl_engine.integrations.qwen3_next_vllm_attention import (
    SharedAttentionImpl,
    SharedAttentionMetadata,
)
from rl_engine.testing.tensor_identity import assert_tensor_bitwise_equal

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("length", [8, 65, 1024])
def test_paged_attention_full_chunk_reorder_and_cache_bits(batch, length):
    torch.manual_seed(777)
    q = torch.randn(batch, length, 4, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, length, 1, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    block = 64
    pages = (length + block - 1) // block
    tables = torch.arange(1, batch * pages + 1, device="cuda").reshape(batch, pages)
    full_cache = torch.zeros(batch * pages + 1, 1, block, 512, device="cuda", dtype=torch.bfloat16)
    chunk_cache = full_cache.clone()
    impl = SharedAttentionImpl(4, 256, 1 / 16, 1)

    def invoke(cache, start, end, order):
        positions = torch.arange(start, end, device="cuda")
        table = tables[order]
        slots = (table[:, positions // block] * block + positions % block).reshape(-1)
        metadata = SharedAttentionMetadata(
            tuple(index * (end - start) for index in range(batch + 1)),
            (end,) * batch,
            table,
            slots,
            batch * (end - start),
        )
        query = q[order, start:end].reshape(-1, 4, 256)
        out = torch.empty_like(query)
        impl.forward(
            None,
            query,
            k[order, start:end].reshape(-1, 1, 256),
            v[order, start:end].reshape(-1, 1, 256),
            cache,
            metadata,
            out,
        )
        return out.reshape(batch, end - start, 4, 256)

    expected = invoke(full_cache, 0, length, list(range(batch)))
    actual = torch.empty_like(expected)
    start = 0
    for end in sorted(
        {1, length, *(point for point in (63, 64, 65, 100, 528, 576) if point < length)}
    ):
        order = list(reversed(range(batch))) if start else list(range(batch))
        result = invoke(chunk_cache, start, end, order)
        actual[order, start:end] = result
        start = end
    assert_tensor_bitwise_equal(actual, expected, name="paged full/chunk/reorder outputs")
    assert_tensor_bitwise_equal(chunk_cache, full_cache, name="complete scheduler-owned KV cache")


@pytest.mark.parametrize("bad_slots", [[-1, 1], [1, 1], [1, 99], [1, 0]])
def test_paged_attention_invalid_indices_do_not_mutate_cache(bad_slots):
    cache = torch.zeros(2, 1, 4, 512, device="cuda", dtype=torch.bfloat16)
    original = cache.clone()
    query = torch.zeros(2, 4, 256, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros(2, 1, 256, device="cuda", dtype=torch.bfloat16)
    metadata = SharedAttentionMetadata(
        (0, 2), (2,), torch.tensor([[0]], device="cuda"), torch.tensor(bad_slots, device="cuda"), 2
    )
    with pytest.raises(ValueError):
        SharedAttentionImpl(4, 256, 1 / 16, 1).forward(
            None, query, key, key, cache, metadata, query.clone()
        )
    assert_tensor_bitwise_equal(cache, original)
