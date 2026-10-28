# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from argparse import Namespace

import pytest
import torch

from rl_engine.validation.models.qwen3_next.qwen3_next_rl_fixture import (
    OPTIMIZER_SETTINGS,
    POLICY_SETTINGS,
    PolicyFixture,
    require_policy_configuration,
)


def fixture():
    return PolicyFixture.freeze(
        torch.arange(320).reshape(4, 80),
        torch.full((4, 16), -3.0),
        baseline_identity="test-baseline",
    )


def test_policy_fixture_preserves_old_policy_and_native_loss_fields():
    value = fixture()
    batch = value.batch([2], device="cpu")
    assert batch["partition"] == [2]
    assert batch["total_lengths"] == [80]
    assert batch["response_lengths"] == [16]
    assert torch.equal(batch["advantages"][0], torch.full((16,), 0.75))
    value.verify_initial_recompute(torch.full((4, 16), -3.0))
    with pytest.raises(AssertionError):
        value.verify_initial_recompute(torch.full((4, 16), -3.01))


def test_policy_fixture_rejects_mutation_and_unfrozen_baseline():
    value = fixture()
    value.old_logprobs[0, 0] = -4
    with pytest.raises(ValueError, match="modified"):
        value.batch([0], device="cpu")
    with pytest.raises(ValueError, match="frozen"):
        PolicyFixture.freeze(
            torch.arange(320).reshape(4, 80),
            torch.zeros(4, 16, requires_grad=True),
            baseline_identity="run",
        )


@pytest.mark.parametrize(
    "changed", ["loss_type", "use_rollout_logprobs", "use_stateless_adam", "optimizer_cpu_offload"]
)
def test_policy_configuration_cannot_fall_back_or_substitute_optimizer(changed):
    settings = {**POLICY_SETTINGS, **OPTIMIZER_SETTINGS}
    require_policy_configuration(Namespace(**settings))
    settings[changed] = "unexpected"
    with pytest.raises(ValueError, match=changed):
        require_policy_configuration(Namespace(**settings))
