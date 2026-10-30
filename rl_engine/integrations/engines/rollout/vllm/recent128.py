# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Opt-in DSV4 cache boundary; leaves vLLM's allocator and attention kernels intact."""

from rl_engine.contracts.operators.recent128 import CacheRequest, CacheVersion, Recent128Contract
from rl_engine.models.deepseek_v4.recent128 import Recent128Batch


def bind_recent128_metadata(
    swa_metadata,
    *,
    requests: tuple[CacheRequest, ...],
    version: CacheVersion,
    contract: Recent128Contract,
) -> Recent128Batch:
    """Bind DeepseekSparseSWAMetadata once and share across the model's layers.

    Pass the cache writer's *actual* encoding in ``contract``. On gfx942, vLLM's
    fused Q-norm/RoPE C++ path may use FNUZ while its standalone Triton writer
    defaults to OCP. Selecting a backend by architecture alone is not sufficient.
    """
    return Recent128Batch.from_metadata(
        swa_metadata,
        requests=requests,
        version=version,
        contract=contract,
    )


def recent128_cache_update(cache, post_rope_kv, batch, *, workspace=None):
    """Replace only the unfused post-RoPE cache boundary, not a fused RoPE op."""
    return cache.update(post_rope_kv, batch, workspace=workspace)
