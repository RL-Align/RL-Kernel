# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.integrations.qwen3_next_logprob import (
    calculate_log_probs_and_entropy,
    qwen_logprob_contract,
)


def test_contract_pins_complete_tp4_vocab_and_original_dtype():
    for rank in range(4):
        contract = qwen_logprob_contract(
            torch.empty(2, 37984, dtype=torch.bfloat16), rank=rank, world_size=4
        )
        assert contract.sharding.vocab_shard_bounds[rank] == (rank * 37984, (rank + 1) * 37984)
        assert contract.sharding.padded_vocab_size == 151936
        assert contract.sharding.cp_world_size == 1
        assert contract.dtype.value == "bf16"


@pytest.mark.parametrize(
    "rank,world,width,dtype",
    [
        (0, 1, 151936, torch.bfloat16),
        (4, 4, 37984, torch.bfloat16),
        (0, 4, 38016, torch.bfloat16),
        (0, 4, 37984, torch.float16),
    ],
)
def test_contract_rejects_profile_drift(rank, world, width, dtype):
    with pytest.raises(ValueError):
        qwen_logprob_contract(torch.empty(0, width, dtype=dtype), rank=rank, world_size=world)


def test_numerical_callback_has_no_fallback_for_mask_or_absent_group():
    logits, tokens = torch.empty(0, 37984), torch.empty(0, dtype=torch.long)
    with pytest.raises(ValueError, match="top-p"):
        calculate_log_probs_and_entropy(logits, tokens, None, log_prob_keep_mask=torch.empty(0))
    with pytest.raises(ValueError, match="explicit initialized"):
        calculate_log_probs_and_entropy(logits, tokens, None)
