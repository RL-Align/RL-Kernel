# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""PPO rollout targets, losses, a sharded example Actor and two-round coordinated PPO."""

from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from examples.post_training.tp_ppo.train import TinyModel, make_example
from rl_engine.integrations.common.ppo import (
    PPOBatch,
    PPOConfig,
    PPORollout,
    build_ppo_batch,
    ppo_policy_loss,
    ppo_value_loss,
)
from rl_engine.integrations.common.tp_scalar import TPModelInputs


def rollout_fixture():
    ids = torch.tensor([[1, 2, 3, 4, 0], [0, 1, 2, 3, 4]])
    attention = torch.tensor([[1, 1, 1, 1, 0], [0, 1, 1, 1, 1]], dtype=torch.bool)
    action = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.bool)
    return PPORollout(
        ids,
        attention,
        (attention.long().cumsum(1) - 1).clamp(min=0),
        action,
        torch.full((2, 4), -1.0),
        torch.tensor([True, False]),
        torch.tensor([19, 7]),
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor([1.0, 0.0, 1.0]),
    )


def test_gae_alignment_terminal_padding_and_snapshots():
    rollout = rollout_fixture()
    rollout.validate(actor_version=0, critic_version=0)
    states = torch.tensor([[99.0, 1.0, 2.0, 8.0, 99.0], [99.0, 99.0, 3.0, 4.0, 5.0]])
    reward = torch.tensor([4.0, 6.0])
    batch = build_ppo_batch(
        rollout,
        state_values=states,
        rewards=reward,
        reference_logps=rollout.behavior_logps,
        config=PPOConfig(gamma=1, gae_lambda=1, kl_coefficient=0),
    )
    torch.testing.assert_close(batch.bootstrap_values, torch.tensor([0.0, 5.0]))
    torch.testing.assert_close(
        batch.returns, torch.tensor([[0.0, 4.0, 4.0, 0.0], [0.0, 0.0, 11.0, 11.0]])
    )
    raw = torch.tensor([3.0, 2.0, 8.0, 7.0])
    torch.testing.assert_close(
        batch.advantages[rollout.action_mask], (raw - raw.mean()) / raw.std(unbiased=False)
    )
    rollout.behavior_logps.fill_(123)
    states.fill_(123)
    reward.fill_(123)
    assert (batch.rollout.behavior_logps == -1).all()
    assert batch.old_values[0, 1] == 1
    assert batch.rewards[0] == 4


def test_clipped_losses_ignore_masked_nan_and_preserve_gradients():
    rollout = rollout_fixture()
    mask = rollout.action_mask
    old = torch.zeros_like(rollout.behavior_logps)
    advantages = torch.zeros_like(old)
    advantages[mask] = torch.tensor([1.0, -1.0, 2.0, -2.0])
    batch = PPOBatch(
        rollout, old, torch.zeros(2), torch.zeros(2), old, advantages, torch.ones_like(old)
    )
    logps = torch.full_like(old, float("nan"))
    ratios = torch.tensor([1.5, 0.5, 1.0, 1.0])
    logps[mask] = -1 + ratios.log()
    logps.requires_grad_()
    loss = ppo_policy_loss(logps, batch, 0.2)
    torch.testing.assert_close(loss, torch.tensor(-0.1))
    loss.backward()
    assert torch.isfinite(logps.grad).all()
    assert (logps.grad[~mask] == 0).all()
    values = torch.full_like(old, float("nan"))
    values[mask] = 0.9
    values.requires_grad_()
    loss = ppo_value_loss(values, batch, 0.2)
    torch.testing.assert_close(loss, torch.tensor(0.32))
    loss.backward()
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"actor_version": torch.tensor(1)}, "actor_version"),
        ({"critic_version": torch.tensor(1)}, "critic_version"),
        ({"sampling": torch.tensor([0.7, 0.0, 1.0])}, "temperature"),
        ({"sampling": torch.tensor([1.0, 3.0, 1.0])}, "temperature"),
        ({"terminated": torch.tensor([1, 0])}, "terminated"),
        ({"action_mask": torch.zeros(2, 4, dtype=torch.bool)}, "nonempty"),
        ({"behavior_logps": torch.full((2, 4), float("nan"))}, "finite"),
        ({"sample_ids": torch.tensor([1, 1])}, "unique"),
    ],
)
def test_rollout_rejects_invalid_inputs(change, match):
    with pytest.raises(ValueError, match=match):
        replace(rollout_fixture(), **change).validate(actor_version=0, critic_version=0)


def test_kl_shaping_and_discounted_bootstrap():
    rollout = rollout_fixture()
    states = torch.ones_like(rollout.input_ids, dtype=torch.float32)
    batch = build_ppo_batch(
        rollout,
        state_values=states,
        rewards=torch.tensor([2.0, 2.0]),
        reference_logps=rollout.behavior_logps - 0.5,
        config=PPOConfig(gamma=0.9, gae_lambda=0.5, kl_coefficient=0.2),
    )
    # terminal deltas [-0.2, 0.9]; cutoff deltas [-0.2, 1.8]
    torch.testing.assert_close(
        batch.returns[rollout.action_mask], torch.tensor([1.205, 1.9, 1.61, 2.8])
    )


def test_training_batch_rejects_live_behavior_graph():
    rollout = rollout_fixture()
    batch = build_ppo_batch(
        rollout,
        state_values=torch.zeros_like(rollout.input_ids, dtype=torch.float32),
        rewards=torch.ones(2),
        reference_logps=rollout.behavior_logps,
        config=PPOConfig(),
    )
    batch.validate()
    batch.rollout.behavior_logps.requires_grad_()
    with pytest.raises(ValueError, match="frozen"):
        batch.validate()


def _init(rank, size, rendezvous, backend):
    torch.set_num_threads(1)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=size,
        init_method=f"file://{rendezvous}",
        timeout=timedelta(seconds=60),
    )


def _actor_worker(rank, size, rendezvous):
    """Sharded example Actor logprobs and gradients match the unsharded Actor."""
    _init(rank, size, rendezvous, "gloo")
    try:
        group = dist.new_group(list(range(size)))
        leader = size - 1
        rollout = rollout_fixture()
        inputs = TPModelInputs(rollout.input_ids, rollout.attention_mask, rollout.position_ids)
        actor = TinyModel(outputs=8, group=group, leader=leader)
        full_actor = TinyModel(outputs=8)
        actor_scores = actor.logps(inputs)
        full_scores = full_actor.logps(inputs)
        torch.testing.assert_close(actor_scores, full_scores)
        (-actor_scores[rollout.action_mask].mean()).backward()
        (-full_scores[rollout.action_mask].mean()).backward()
        for name, axis in (("embedding", 1), ("head", 0)):
            torch.testing.assert_close(
                getattr(actor, name).grad,
                getattr(full_actor, name).grad.chunk(size, dim=axis)[rank],
            )
        if rank == leader:
            torch.testing.assert_close(actor.bias.grad, full_actor.bias.grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [1, 2, 4])
def test_sharded_actor_logps_and_gradients(tmp_path, size):
    """The example Actor's TP logprobs and shard gradients match TP=1."""
    mp.start_processes(
        _actor_worker, args=(size, str(tmp_path / "init")), nprocs=size, start_method="spawn"
    )


def _ppo_worker(rank, size, rendezvous, actor_tp, critic_tp):
    _init(rank, size, rendezvous, "gloo")
    try:
        coordinator, models = make_example(actor_tp, critic_tp, epochs=2)
        initial = {
            role: {name: p.detach().clone() for name, p in model.named_parameters()}
            for role, model in models.items()
        }
        if rank == 0:
            original_sample = coordinator.services.sample

            def checked_sample(actor_version, critic_version):
                result = original_sample(actor_version, critic_version)
                recomputed = models["rollout"].logps(result.inputs)
                torch.testing.assert_close(
                    recomputed[result.action_mask], result.behavior_logps[result.action_mask]
                )
                return result

            coordinator.services.sample = checked_sample
        previous = None
        for iteration in range(2):
            metrics = coordinator.run_round()
            assert metrics["actor_version"] == metrics["critic_version"] == iteration + 1
            batch = coordinator.last_batch
            assert batch.rollout.actor_version == iteration
            assert not batch.old_values.requires_grad
            assert not batch.returns.requires_grad
            if previous is not None:
                torch.testing.assert_close(previous[0].old_values, previous[1])
                torch.testing.assert_close(previous[0].rollout.behavior_logps, previous[2])
            previous = (batch, batch.old_values.clone(), batch.rollout.behavior_logps.clone())
        for role, model in models.items():
            changed = any(
                not torch.equal(p, initial[role][name]) for name, p in model.named_parameters()
            )
            assert changed == (role in ("actor", "critic", "rollout"))
            if role in ("reference", "reward"):
                assert all(p.grad is None and not p.requires_grad for p in model.parameters())

        if rank == 0:

            def empty_sample(actor_version, critic_version):
                sample = original_sample(actor_version, critic_version)
                packet = sample.packet()
                for name in packet:
                    if name not in ("actor_version", "critic_version", "sampling"):
                        packet[name] = packet[name][:0]
                return PPORollout(**packet)

            coordinator.services.sample = empty_sample
        metrics = coordinator.run_round()
        assert metrics["updates"] == 0 and metrics["actor_version"] == 2

        if rank == 0:
            coordinator.services.sample = original_sample
            coordinator.services.install = lambda weights, version: False
        with pytest.raises(ValueError, match="acknowledge"):
            coordinator.run_round()
        assert coordinator.actor_version == 2 and coordinator.critic_version == 2
        with pytest.raises(RuntimeError, match="restore"):
            coordinator.run_round()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("actor_tp,critic_tp", [(2, 1), (1, 2)])
def test_two_round_ppo_separate_tp_groups(tmp_path, actor_tp, critic_tp):
    size = 1 + actor_tp + critic_tp
    mp.start_processes(
        _ppo_worker,
        args=(size, str(tmp_path / "ppo"), actor_tp, critic_tp),
        nprocs=size,
        start_method="spawn",
    )
