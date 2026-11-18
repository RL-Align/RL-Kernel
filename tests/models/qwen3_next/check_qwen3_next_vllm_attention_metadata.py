# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from types import SimpleNamespace

import pytest
import torch

from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm_attention import (
    SharedAttentionBackend,
    SharedAttentionBuilder,
    SharedAttentionImpl,
    validate_boundaries,
)


def _config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(enforce_eager=True),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        speculative_config=None,
    )


def test_real_builder_preserves_mixed_absolute_prefixes_and_ignores_optimistic_bounds():
    builder = SharedAttentionBuilder(None, ["layer3"], _config(), torch.device("cpu"))
    metadata = builder.build(
        0,
        SimpleNamespace(
            num_reqs=3,
            num_actual_tokens=5,
            query_start_loc=torch.tensor([0, 1, 1, 5]),
            seq_lens=torch.tensor([65, 0, 4]),
            seq_lens_cpu_upper_bound=torch.tensor([100, 0, 999]),
            block_table_tensor=torch.tensor([[3, 4], [0, 0], [8, 0]]),
            slot_mapping=torch.tensor([128, 256, 257, 258, 259, -1]),
            causal=True,
        ),
    )
    assert metadata.seq_lens == (65, 0, 4)
    assert metadata.query_start_loc == (0, 1, 1, 5)
    assert metadata.slot_mapping.numel() == 5
    assert not SharedAttentionBackend.supports_batch_invariance()


@pytest.mark.parametrize(
    "starts,lengths,n",
    [
        ((0, 2), (1,), 2),
        ((0, 1), (1,), 2),
        ((1, 2), (2,), 2),
        ((0, 1), (1041,), 1),
        ((0, 2, 1), (2, 1), 1),
    ],
)
def test_invalid_scheduler_boundaries_fail(starts, lengths, n):
    with pytest.raises(ValueError):
        validate_boundaries(starts, lengths, n)


def test_empty_batch_and_profile_guards():
    validate_boundaries((0,), (), 0)
    config = _config()
    config.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="Prefix reuse"):
        SharedAttentionBuilder(None, [], config, torch.device("cpu"))
    with pytest.raises(ValueError, match="four Q heads"):
        SharedAttentionImpl(8, 256, 1 / 16, 1)
    SharedAttentionImpl(4, 256, 1 / 16, 1)
