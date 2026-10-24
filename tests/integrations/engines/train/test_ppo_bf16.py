# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch
import torch.distributed as dist

from rl_engine.integrations.common.tp_scalar import clip_tp_grad_norm_
from rl_engine.integrations.engines.train.megatron.ppo_precision import FP32MasterWeights


@pytest.fixture
def group(tmp_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1
    )
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


def test_master_gradient_clipping_consumes_fp32_buffer_not_model_grad(group):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.bfloat16))
    master = FP32MasterWeights([parameter], lambda ps: torch.optim.SGD(ps, lr=0.25))
    parameter.grad = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    original_gradient = parameter.grad
    buffers = master.gradients()
    norm = clip_tp_grad_norm_(
        [(parameter, "replicated")],
        gradients=buffers,
        group=group,
        max_norm=1,
        device=parameter.device,
    )
    assert norm.item() == 5
    torch.testing.assert_close(original_gradient.float(), torch.tensor([3.0, 4.0]))
    master.step()
    expected = torch.tensor([1.0, -1.0]) - 0.25 * torch.tensor([3.0, 4.0]) / 5.000001
    torch.testing.assert_close(master.masters[0], expected)
    assert torch.equal(parameter, expected.bfloat16())
    master.zero_grad()
    assert parameter.grad is None and master.masters[0].grad is None


def test_sub_bf16_updates_accumulate_in_fp32_master():
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    master = FP32MasterWeights([parameter], lambda ps: torch.optim.SGD(ps, lr=0.001))
    for _ in range(8):
        master.zero_grad()
        parameter.grad = torch.ones_like(parameter)
        master.gradients()
        master.step()
    torch.testing.assert_close(master.masters[0], torch.tensor([0.992]))
    assert torch.equal(parameter, torch.tensor([0.992]).bfloat16())


def test_master_rejects_step_without_fresh_gradients():
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    master = FP32MasterWeights([parameter], lambda ps: torch.optim.SGD(ps, lr=0.1))
    with pytest.raises(RuntimeError, match="materialize"):
        master.step()


def test_nonfinite_optimizer_buffer_is_rejected(group):
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    parameter.grad = torch.zeros_like(parameter)
    with pytest.raises(ValueError, match="non-finite"):
        clip_tp_grad_norm_(
            [(parameter, "replicated")],
            gradients=[torch.tensor([float("nan")])],
            group=group,
            max_norm=1,
            device=parameter.device,
        )


def test_microbatch_gradients_accumulate_in_fp32_before_clipping():
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    master = FP32MasterWeights([parameter], lambda ps: torch.optim.SGD(ps, lr=0.001))
    for value in [256.0] + [1.0] * 8:
        parameter.grad = torch.full_like(parameter, value)
        master.accumulate()
        assert parameter.grad is None
    assert master.gradients()[0].item() == 264


def test_microbatch_token_weighting_matches_full_minibatch(group):
    from rl_engine.integrations.common.ppo import PPOBatch, PPOConfig, PPOTrainingWorker
    from tests.integrations.common.test_ppo import rollout_fixture

    rollout = rollout_fixture()
    # Deliberately unequal valid completion lengths: weighting sequence means
    # equally would produce the wrong loss and gradient.
    rollout.action_mask[0, 1] = False
    old = torch.zeros_like(rollout.behavior_logps)
    batch = PPOBatch(
        rollout,
        old,
        torch.zeros(2),
        torch.zeros(2),
        old,
        torch.tensor([[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, -1.0, 3.0]]),
        old,
    )
    results = []
    for microbatch in (None, 1):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.01)

        def forward(inputs):
            return -1 + model(inputs.input_ids[:, :-1].float().unsqueeze(-1)).squeeze(-1)

        worker = PPOTrainingWorker(
            role="actor",
            model=model,
            forward=forward,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            parameter_layouts={"weight": "replicated"},
            group=group,
            config=PPOConfig(epochs=1, minibatch_size=2, microbatch_size=microbatch),
        )
        stats = worker.train(batch)
        results.append((model.weight.detach().clone(), stats))
    torch.testing.assert_close(results[0][0], results[1][0])
    assert results[0][1]["loss"] == pytest.approx(results[1][1]["loss"], abs=1e-6)
    assert results[0][1]["grad_norm"] == pytest.approx(results[1][1]["grad_norm"], abs=1e-6)
