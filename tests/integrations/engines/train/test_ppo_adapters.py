# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.integrations.common.ppo import PPOConfig, PPOTrainingWorker
from rl_engine.integrations.engines.train.megatron.ppo import megatron_attention_mask


def test_megatron_mask_preserves_causality_and_left_padding():
    valid = torch.tensor([[False, True, True, True], [True, True, True, False]])
    blocked = megatron_attention_mask(valid)
    assert blocked.shape == (2, 1, 4, 4)
    for batch in range(2):
        for query in range(4):
            assert (~blocked[batch, 0, query]).any()
            for key in range(4):
                if valid[batch, query]:
                    allowed = key <= query and bool(valid[batch, key])
                    assert bool(blocked[batch, 0, query, key]) != allowed
    assert not blocked[0, 0, 0, 0]


def test_engine_callbacks_must_be_paired():
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="provided together"):
        PPOTrainingWorker(
            role="critic",
            model=model,
            forward=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            parameter_layouts={"weight": "replicated", "bias": "replicated"},
            group=None,
            config=PPOConfig(),
            backward=lambda loss: loss.backward(),
        )
