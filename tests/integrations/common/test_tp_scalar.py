# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""TP-sharded Reward/Critic forward, rollout packet transport and reward return (#44)."""

import os
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.integrations.common.tp_scalar import (
    TPModelInputs,
    TPScalarExecutor,
    clip_tp_grad_norm_,
)
from rl_engine.integrations.common.weights import TensorPacketBridge, synchronize_validation


class ShardedScalarModel(torch.nn.Module):
    """Feature-sharded embedding and head whose outputs are additive across TP ranks.

    The bias lives only on the TP leader, so summing rank outputs gives the full model.
    """

    def __init__(self, *, width=8, vocab=8, group=None, leader=0, device="cpu"):
        """Build the full model from a fixed seed and keep this rank's feature shard."""
        super().__init__()
        self.group, self.leader = group, leader
        size = dist.get_world_size(group) if group is not None else 1
        rank = dist.get_rank(group) if group is not None else 0
        rng = torch.Generator().manual_seed(1)
        embedding = torch.randn(vocab, width, generator=rng) * 0.2
        head = torch.randn(width, 1, generator=rng) * 0.2
        self.embedding = torch.nn.Parameter(embedding.chunk(size, dim=1)[rank].clone().to(device))
        self.head = torch.nn.Parameter(head.chunk(size, dim=0)[rank].clone().to(device))
        if group is None or dist.get_rank() == leader:
            self.bias = torch.nn.Parameter(torch.zeros(1, device=device))
        else:
            self.register_buffer("bias", torch.zeros(1, device=device))

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        """Return this rank's additive [B, S] contribution."""
        assert not use_cache
        return (self.embedding[input_ids].tanh() @ self.head + self.bias).squeeze(-1)

    @torch.no_grad()
    def export(self):
        """Gather the full parameters on the TP leader; other ranks return None."""
        if self.group is None:
            return dict(self.state_dict())
        size = dist.get_world_size(self.group)
        embeddings = [torch.empty_like(self.embedding) for _ in range(size)]
        heads = [torch.empty_like(self.head) for _ in range(size)]
        dist.all_gather(embeddings, self.embedding, group=self.group)
        dist.all_gather(heads, self.head, group=self.group)
        if dist.get_rank() == self.leader:
            return {
                "embedding": torch.cat(embeddings, dim=1),
                "head": torch.cat(heads, dim=0),
                "bias": self.bias.detach(),
            }
        return None


def model_inputs(device="cpu"):
    """Two sequences, one right- and one left-padded, with their last valid tokens at 3 and 4."""
    ids = torch.tensor([[1, 2, 3, 4, 0], [0, 1, 2, 3, 4]], device=device)
    attention = torch.tensor([[1, 1, 1, 1, 0], [0, 1, 1, 1, 1]], dtype=torch.bool, device=device)
    return TPModelInputs(ids, attention, (attention.long().cumsum(1) - 1).clamp(min=0))


def _init(rank, size, rendezvous, backend):
    """Join a file-rendezvous process group with a short timeout."""
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


def _validation_worker(rank, rendezvous):
    """An empty exception message fails on both ranks, including the healthy peer."""
    _init(rank, 2, rendezvous, "gloo")
    try:
        group = dist.group.WORLD
        synchronize_validation(None, group=group, device=torch.device("cpu"))
        for message in ("", "adapter gradient failure"):
            with pytest.raises(ValueError) as raised:
                synchronize_validation(
                    message if rank == 1 else None, group=group, device=torch.device("cpu")
                )
            assert str(raised.value) == message
    finally:
        dist.destroy_process_group()


def test_validation_propagates_empty_error_to_every_rank(tmp_path):
    """None alone signals success; even str(RuntimeError()) must abort all ranks."""
    mp.start_processes(
        _validation_worker,
        args=(str(tmp_path / "validation"),),
        nprocs=2,
        start_method="spawn",
    )


def _tp_worker(rank, size, rendezvous, backend):
    """Check rewards, values, gradients, clipping and packet transport against TP=1."""
    _init(rank, size, rendezvous, backend)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    try:
        group = dist.new_group(list(range(size)))
        leader = size - 1
        inputs = model_inputs(device)
        model = ShardedScalarModel(group=group, leader=leader, device=device)
        full = ShardedScalarModel(device=device)
        executor = TPScalarExecutor(model, group=group, leader=leader, output_layout="local_sum")
        reference = full(inputs.input_ids)
        reward = executor.rewards(inputs)
        if rank == leader:
            torch.testing.assert_close(
                reward,
                reference[torch.arange(2, device=device), torch.tensor([3, 4], device=device)],
            )
        else:
            assert reward is None
        # Rewards are pooled before communication: one reduce of B float32 scalars.
        assert executor.head_metrics == {
            "reduce_calls": int(size > 1),
            "payload_bytes": 8 * int(size > 1),
        }
        values = executor.values(inputs)
        torch.testing.assert_close(values, reference)
        values.square().mean().backward()
        reference.square().mean().backward()
        for name in ("embedding", "head"):
            axis = 1 if name == "embedding" else 0
            expected = getattr(full, name).grad.chunk(size, dim=axis)[rank]
            torch.testing.assert_close(getattr(model, name).grad, expected)
        if rank == leader:
            torch.testing.assert_close(model.bias.grad, full.bias.grad)
        expected_norm = torch.nn.utils.clip_grad_norm_(full.parameters(), 0.001)
        actual_norm = clip_tp_grad_norm_(
            [(p, "sharded") for p in model.parameters()],
            group=group,
            max_norm=0.001,
            device=device,
        )
        torch.testing.assert_close(actual_norm.float(), expected_norm)
        torch.optim.SGD(model.parameters(), lr=0.1).step()
        torch.optim.SGD(full.parameters(), lr=0.1).step()
        exported = model.export()
        if rank == leader:
            for name, expected in full.state_dict().items():
                torch.testing.assert_close(exported[name], expected)

        replicated = ShardedScalarModel(device=device)
        replicated_executor = TPScalarExecutor(
            replicated, group=group, leader=leader, output_layout="replicated"
        )
        replicated_executor.rewards(inputs)
        replicated_executor.values(inputs).sum().backward()
        local_norm = (
            torch.stack([p.grad.double().square().sum() for p in replicated.parameters()])
            .sum()
            .sqrt()
        )
        norm = clip_tp_grad_norm_(
            [(p, "replicated") for p in replicated.parameters()],
            group=group,
            max_norm=1,
            device=device,
        )
        torch.testing.assert_close(norm, local_norm)
        assert replicated_executor.head_metrics["reduce_calls"] == 0

        shard = torch.nn.Parameter(torch.zeros(1, device=device))
        replica = torch.nn.Parameter(torch.zeros(1, device=device))
        shard.grad = torch.tensor([rank + 1.0], device=device)
        replica.grad = torch.tensor([2.0], device=device)
        mixed_norm = clip_tp_grad_norm_(
            [(shard, "sharded"), (replica, "replicated")],
            group=group,
            max_norm=1,
            device=device,
        )
        expected = torch.tensor(
            sum(i * i for i in range(1, size + 1)) + 4.0, dtype=torch.float64, device=device
        ).sqrt()
        torch.testing.assert_close(mixed_norm, expected)
        torch.testing.assert_close(replica.grad, (2 / (expected + 1e-6)).float().reshape(1))

        bridge = TensorPacketBridge(group=group, source=leader, device=device)
        for shape in ((2, 5), (1, 3), (0, 7)):
            packet = (
                {
                    "tokens": torch.ones(shape, dtype=torch.int64, device=device),
                    "version": torch.tensor(7, device=device),
                }
                if rank == leader
                else None
            )
            result = bridge.broadcast(("tokens", "version"), packet)
            assert result["tokens"].shape == shape
            assert result["version"] == 7
            assert bridge.metrics["payload_bytes"] == result["tokens"].numel() * 8 + 8
        with pytest.raises(ValueError, match="schema"):
            bridge.broadcast(("missing",), {} if rank == leader else None)
        bad_inputs = (
            replace(inputs, attention_mask=inputs.attention_mask.int())
            if rank == leader
            else inputs
        )
        with pytest.raises(ValueError, match="bool"):
            executor.values(bad_inputs)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [1, 2, 4])
def test_tp_rewards_values_gradients_and_packets(tmp_path, size):
    """TP=1/2/4 sharded rewards, values, gradients and updates match the unsharded model."""
    mp.start_processes(
        _tp_worker, args=(size, str(tmp_path / "init"), "gloo"), nprocs=size, start_method="spawn"
    )


def _round_trip_worker(rank, size, rendezvous, rollout_ranks, reward_tp):
    """Rollout ranks -> TP reward group -> rewards back to every rollout rank."""
    _init(rank, size, rendezvous, "gloo")
    try:
        rollout = list(range(rollout_ranks))
        reward = list(range(rollout_ranks, rollout_ranks + reward_tp))
        rollout_leader, reward_leader = rollout[0], reward[-1]
        tp_group = dist.new_group(reward)
        # Every process creates every group, in the same order.
        to_reward = dist.new_group([rollout_leader, *reward])
        to_rollout = dist.new_group(sorted({reward_leader, *rollout}))
        inputs = model_inputs()
        fields = ("input_ids", "attention_mask", "position_ids")

        tokens = None
        if rank in (rollout_leader, *reward):
            packet = (
                {name: getattr(inputs, name) for name in fields} if rank == rollout_leader else None
            )
            tokens = TensorPacketBridge(group=to_reward, source=rollout_leader, device="cpu")
            tokens = tokens.broadcast(fields, packet)

        rewards = None
        if rank in reward:
            received = TPModelInputs(*(tokens[name] for name in fields))
            model = ShardedScalarModel(group=tp_group, leader=reward_leader)
            executor = TPScalarExecutor(
                model, group=tp_group, leader=reward_leader, output_layout="local_sum"
            )
            rewards = executor.rewards(received)
            assert executor.head_metrics["reduce_calls"] == int(reward_tp > 1)

        if rank in (reward_leader, *rollout):
            back = TensorPacketBridge(group=to_rollout, source=reward_leader, device="cpu")
            result = back.broadcast(
                ("rewards",), {"rewards": rewards} if rank == reward_leader else None
            )
            if rank in rollout:
                full = ShardedScalarModel()(inputs.input_ids)
                torch.testing.assert_close(
                    result["rewards"], full[torch.arange(2), torch.tensor([3, 4])]
                )
                # The return packet carries only B scalar rewards.
                assert back.metrics["payload_bytes"] == 2 * 4
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rollout_ranks,reward_tp", [(1, 1), (2, 2), (1, 3)])
def test_rollout_tokens_to_tp_reward_and_rewards_back(tmp_path, rollout_ranks, reward_tp):
    """Token IDs reach a TP reward group and its scalar rewards reach every rollout rank."""
    size = rollout_ranks + reward_tp
    mp.start_processes(
        _round_trip_worker,
        args=(size, str(tmp_path / "init"), rollout_ranks, reward_tp),
        nprocs=size,
        start_method="spawn",
    )


@pytest.mark.skipif(os.environ.get("RUN_TP_CUDA") != "1", reason="opt-in multi-GPU validation")
def test_nccl_tp_rewards_values_gradients_and_packets(tmp_path):
    """The TP=2 checks also pass over NCCL on two GPUs."""
    mp.start_processes(
        _tp_worker, args=(2, str(tmp_path / "nccl"), "nccl"), nprocs=2, start_method="spawn"
    )
