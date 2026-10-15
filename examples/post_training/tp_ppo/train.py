# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Real sampled PPO with separate TP groups, using tiny bigram models.

Run from the repository root:
  torchrun --standalone --nproc-per-node=4 -m examples.post_training.tp_ppo.train \\
    --actor-tp 2 --critic-tp 1
The example intentionally uses small PyTorch models, not a serving framework.
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from rl_engine.integrations.common.ppo import PPOConfig, PPORollout, PPOTrainingWorker
from rl_engine.integrations.common.ppo_coordinator import (
    PPOCoordinator,
    PPOProcessGroups,
    PPOServices,
)
from rl_engine.integrations.common.tp_scalar import (
    TPModelInputs,
    TPScalarExecutor,
    sum_tp_contributions,
)


class TinyModel(torch.nn.Module):
    """Feature-sharded embedding and head; bias belongs only to the TP leader."""

    def __init__(
        self,
        *,
        outputs: int,
        width: int = 8,
        vocab: int = 8,
        group=None,
        leader: int = 0,
        seed: int = 1,
        device="cpu",
    ):
        super().__init__()
        self.group, self.leader = group, leader
        size = dist.get_world_size(group) if group is not None else 1
        rank = dist.get_rank(group) if group is not None else 0
        if width % size:
            raise ValueError("example hidden width must be divisible by TP size")
        rng = torch.Generator().manual_seed(seed)
        embedding = torch.randn(vocab, width, generator=rng) * 0.2
        head = torch.randn(width, outputs, generator=rng) * 0.2
        self.embedding = torch.nn.Parameter(embedding.chunk(size, dim=1)[rank].clone().to(device))
        self.head = torch.nn.Parameter(head.chunk(size, dim=0)[rank].clone().to(device))
        if group is None or dist.get_rank() == leader:
            self.bias = torch.nn.Parameter(torch.zeros(outputs, device=device))
        else:
            self.register_buffer("bias", torch.zeros(outputs, device=device))

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        if use_cache:
            raise ValueError("example does not use a KV cache")
        output = self.embedding[input_ids].tanh() @ self.head + self.bias
        return output.squeeze(-1) if output.shape[-1] == 1 else output

    def logps(self, inputs: TPModelInputs) -> torch.Tensor:
        logits = self(inputs.input_ids)
        if self.group is not None:
            logits = sum_tp_contributions(logits, self.group)
        return logits[:, :-1].log_softmax(-1).gather(-1, inputs.input_ids[:, 1:, None]).squeeze(-1)

    @torch.no_grad()
    def export(self):
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


class TinySampler:
    def __init__(self, device="cpu"):
        self.model = TinyModel(outputs=8, device=device)
        self.version = -1
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(44)

    def install(self, weights, version):
        self.model.load_state_dict(weights)
        self.version = version
        return True

    @torch.no_grad()
    def sample(self, actor_version, critic_version):
        if actor_version != self.version:
            raise ValueError("sampler has stale Actor weights")
        batch, max_tokens = 4, 5
        tokens = torch.zeros(batch, max_tokens, dtype=torch.int64, device=self.device)
        tokens[:, 0] = torch.arange(batch, device=self.device) + 1
        attention = torch.zeros_like(tokens, dtype=torch.bool)
        attention[:, 0] = True
        action_mask = torch.zeros(batch, max_tokens - 1, dtype=torch.bool, device=self.device)
        behavior = torch.zeros(batch, max_tokens - 1, device=self.device)
        terminated = torch.zeros(batch, dtype=torch.bool, device=self.device)
        for t in range(max_tokens - 1):
            logits = self.model(tokens[:, t])
            logps = logits.log_softmax(-1)
            sampled = torch.multinomial(logps.exp(), 1, generator=self.generator).squeeze(1)
            active = ~terminated
            tokens[:, t + 1] = torch.where(active, sampled, 0)
            attention[:, t + 1] = active
            action_mask[:, t] = active
            behavior[:, t] = torch.where(active, logps.gather(1, sampled[:, None]).squeeze(1), 0)
            terminated |= active & (sampled == 0)
        positions = (attention.long().cumsum(1) - 1).clamp(min=0)
        return PPORollout(
            tokens,
            attention,
            positions,
            action_mask,
            behavior,
            terminated,
            torch.arange(batch, device=self.device),
            torch.tensor(actor_version, device=self.device),
            torch.tensor(critic_version, device=self.device),
            torch.tensor([1.0, 0.0, 1.0], device=self.device),
        )


def make_example(actor_tp: int, critic_tp: int, *, device="cpu", epochs=4):
    """Construct groups/services on every rank; also used by integration tests."""
    world = 1 + actor_tp + critic_tp
    if dist.get_world_size() != world:
        raise ValueError("world size must equal 1 + actor_tp + critic_tp")
    control = dist.new_group(list(range(world)))
    rollout_ranks = [0]
    actor_ranks = list(range(1, 1 + actor_tp))
    critic_ranks = list(range(1 + actor_tp, world))
    leaders = [0, actor_ranks[-1], critic_ranks[-1]]
    relay = dist.new_group(leaders)
    role_groups = [dist.new_group(ranks) for ranks in (rollout_ranks, actor_ranks, critic_ranks)]
    rank = dist.get_rank()
    role_index = 0 if rank == 0 else (1 if rank in actor_ranks else 2)
    role = ("rollout", "actor", "critic")[role_index]
    local = role_groups[role_index]
    groups = PPOProcessGroups(control, relay, local, role, *leaders)
    config = PPOConfig(epochs=epochs, minibatch_size=2)
    service = PPOServices()
    models = {}
    if role == "rollout":
        sampler = TinySampler(device)
        service.sample, service.install = sampler.sample, sampler.install
        models["rollout"] = sampler.model
    else:
        model = TinyModel(
            outputs=8 if role == "actor" else 1,
            group=local,
            leader=leaders[role_index],
            device=device,
        )
        models[role] = model
        if role == "actor":
            reference = TinyModel(outputs=8, group=local, leader=leaders[1], device=device)
            reference.requires_grad_(False).eval()
            models["reference"] = reference
            service.reference, service.export = reference.logps, model.export
            service.reference_model = reference
            forward = model.logps
        else:
            reward = TinyModel(outputs=1, group=local, leader=leaders[2], seed=3, device=device)
            models["reward"] = reward
            service.reward = TPScalarExecutor(
                reward, group=local, leader=leaders[2], output_layout="local_sum"
            )
            service.critic = TPScalarExecutor(
                model, group=local, leader=leaders[2], output_layout="local_sum"
            )
            forward = service.critic.values
        service.worker = PPOTrainingWorker(
            role=role,
            model=model,
            forward=forward,
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
            parameter_layouts={name: "sharded" for name, _ in model.named_parameters()},
            group=local,
            config=config,
        )
    coordinator = PPOCoordinator(
        groups=groups,
        services=service,
        config=config,
        weight_fields=("embedding", "head", "bias"),
        device=device,
    )
    return coordinator, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-tp", type=int, default=2)
    parser.add_argument("--critic-tp", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    args = parser.parse_args()
    device = "cpu"
    if args.backend == "nccl":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    dist.init_process_group(args.backend, timeout=timedelta(seconds=120))
    try:
        coordinator, _ = make_example(args.actor_tp, args.critic_tp, device=device)
        for iteration in range(args.rounds):
            metrics = coordinator.run_round()
            print(f"rank={dist.get_rank()} round={iteration} {metrics}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
