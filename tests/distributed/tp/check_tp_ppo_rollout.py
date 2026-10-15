# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Real-framework, independent-role PPO scheduling on shared or dedicated GPUs.

Actor TP=2; Critic TP=1 or 2. Use torchrun with 4 or 6 processes respectively.
Role groups use NCCL; explicit CPU/Gloo transport permits GPU sharing across
roles. This correctness harness does not measure dedicated-GPU performance.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from rl_engine.integrations.common.ppo import PPOConfig, PPORollout, PPOTrainingWorker
from rl_engine.integrations.common.ppo_coordinator import (
    PPOCoordinator,
    PPOProcessGroups,
    PPOServices,
)
from rl_engine.integrations.common.tp_scalar import TPScalarExecutor
from rl_engine.integrations.common.weights import TensorPacketBridge
from rl_engine.integrations.engines.train.deepspeed.ppo import deepspeed_ppo_worker
from rl_engine.integrations.engines.train.megatron.ppo import (
    megatron_attention_mask,
    megatron_parameter_layouts,
)
from tests.distributed.tp.check_tp_ppo_frameworks import (
    initialize_engine,
    make_gpt,
    make_model,
    snapshot,
)


def run(args):
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    rank, world = dist.get_rank(), dist.get_world_size()
    timeout = 600 if args.vllm_python else 120
    hidden_size = 128 if args.vllm_python else 32
    max_sequence_length = 64 if args.vllm_python else 16
    if args.vllm_python and args.critic_tp != 1:
        raise ValueError("vLLM harness currently requires one rollout worker (critic-tp=1)")
    assert world == (4 if args.critic_tp == 1 else 6)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2, distributed_timeout_minutes=timeout // 60
    )
    control = dist.new_group(list(range(world)), backend="gloo", timeout=timedelta(seconds=timeout))
    critic_ranks = list(range(2, 2 + args.critic_tp))
    rollout_ranks = list(range(2 + args.critic_tp, world))
    leaders = dist.new_group(
        [0, 2, rollout_ranks[0]], backend="gloo", timeout=timedelta(seconds=timeout)
    )
    singleton = None
    for i in range(world):
        group = dist.new_group([i], backend="nccl", timeout=timedelta(seconds=timeout))
        if rank == i:
            singleton = group
    role = "actor" if rank < 2 else "critic" if rank in critic_ranks else "rollout"
    local = parallel_state.get_tensor_model_parallel_group()
    if args.critic_tp == 1 and role != "actor":
        local = singleton
    tp = dist.get_world_size(local) if role != "rollout" else 1
    model_group = singleton if role == "rollout" else local
    pgs = ProcessGroupCollection(
        tp=model_group,
        pp=singleton,
        mp=model_group,
        embd=singleton,
        pos_embd=singleton,
        cp=singleton,
        tp_cp=model_group,
        hcp=None,
        ep=singleton,
        expt_tp=model_group,
        tp_ep=model_group,
        tp_ep_pp=model_group,
        tp_dp_cp=model_group,
        dp=singleton,
        dp_cp=singleton,
        expt_dp=singleton,
        intra_dp_cp=singleton,
        intra_expt_dp=singleton,
        inter_dist_opt=singleton,
        intra_dist_opt=singleton,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    config = PPOConfig(epochs=2, minibatch_size=2)
    services = PPOServices()
    installed, sampled, rounds = [], [], []
    behavior_errors = []
    engine = frozen = trainable = vllm_client = None
    if role == "rollout" and args.vllm_python:
        from tests.distributed.tp.vllm_ppo_client import VllmPPOClient

        vllm_client = VllmPPOClient(args.vllm_python, args.output / "vllm", device)
        services.install, services.sample = vllm_client.install, vllm_client.sample
        installed, sampled = vllm_client.installed, vllm_client.sampled
    elif role == "rollout":
        model = make_gpt(1, policy=True, pg_collection=pgs).eval()

        def install(weights, version):
            state = model.state_dict()
            assert set(weights) == {k for k, v in state.items() if isinstance(v, torch.Tensor)}
            state.update(weights)
            model.load_state_dict(state, strict=True)
            for name, value in weights.items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            installed.append(version)
            return True

        @torch.no_grad()
        def sample(actor_version, critic_version):
            assert installed[-1] == actor_version
            sampled.append(actor_version)
            # Different prompt lengths exercise left padding. Four real sampled
            # actions per row; EOS=31 or generation cap determines termination.
            ids = torch.tensor([[0, 1, 2], [1, 3, 4]], device=device)
            mask = ids != 0
            logps = torch.zeros((2, 2), device=device)
            actions = torch.zeros((2, 2), dtype=torch.bool, device=device)
            done = torch.zeros(2, dtype=torch.bool, device=device)
            generator = torch.Generator(device=device).manual_seed(123 + actor_version)
            for _ in range(4):
                positions = (mask.long().cumsum(1) - 1).clamp(min=0)
                logits = model(
                    input_ids=ids,
                    position_ids=positions,
                    attention_mask=megatron_attention_mask(mask),
                )[:, -1].float()
                distribution = logits.log_softmax(-1)
                token = torch.multinomial(distribution.exp(), 1, generator=generator).squeeze(1)
                valid = ~done
                score = distribution.gather(1, token[:, None]).squeeze(1)
                ids = torch.cat((ids, torch.where(valid, token, 0)[:, None]), 1)
                mask = torch.cat((mask, valid[:, None]), 1)
                actions = torch.cat((actions, valid[:, None]), 1)
                logps = torch.cat((logps, score[:, None]), 1)
                done |= valid & (token == 31)
            return PPORollout(
                ids,
                mask,
                (mask.long().cumsum(1) - 1).clamp(min=0),
                actions,
                logps,
                done,
                torch.tensor([13, 7], device=device),
                torch.tensor(actor_version, device=device),
                torch.tensor(critic_version, device=device),
                torch.tensor([1.0, 0.0, 1.0], device=device),
            )

        services.install, services.sample = install, sample
    else:
        trainable = make_model(
            tp,
            policy=role == "actor",
            pg_collection=pgs,
            hidden_size=hidden_size,
            max_sequence_length=max_sequence_length,
        )
        frozen = make_model(
            tp,
            policy=role == "actor",
            pg_collection=pgs,
            hidden_size=hidden_size,
            max_sequence_length=max_sequence_length,
        ).requires_grad_(False)
        initial = snapshot(trainable, local)
        frozen_initial = snapshot(frozen, local)
        engine = (
            initialize_engine(trainable, tp_group=local, dp_group=singleton)
            if args.framework == "deepspeed"
            else None
        )
        training_model = engine if engine is not None else trainable
        if role == "actor":
            forward = training_model
            services.reference, services.reference_model = frozen, frozen
            services.export = lambda: snapshot(trainable.model, local)
        else:
            services.critic = TPScalarExecutor(
                training_model, group=local, leader=2, output_layout="local_sum"
            )
            services.reward = TPScalarExecutor(
                frozen, group=local, leader=2, output_layout="local_sum"
            )
            forward = services.critic.values
        services.worker = (
            deepspeed_ppo_worker(engine, role=role, forward=forward, group=local, config=config)
            if engine is not None
            else PPOTrainingWorker(
                role=role,
                model=trainable,
                forward=forward,
                optimizer=torch.optim.SGD(trainable.parameters(), lr=0.01),
                parameter_layouts=megatron_parameter_layouts(trainable),
                group=local,
                config=config,
            )
        )
        original_train = services.worker.train

        def checked_train(batch):
            if role == "actor":
                with torch.no_grad():
                    actual = trainable(batch.rollout.inputs)
                selected = batch.rollout.action_mask
                behavior_errors.append(
                    float((actual[selected] - batch.rollout.behavior_logps[selected]).abs().max())
                )
                torch.testing.assert_close(
                    actual[selected],
                    batch.rollout.behavior_logps[selected],
                    atol=2e-6,
                    rtol=2e-6,
                )
            frozen_targets = {k: v.clone() for k, v in batch.targets().items()}
            behavior = batch.rollout.behavior_logps.clone()
            metrics = original_train(batch)
            for key, value in frozen_targets.items():
                torch.testing.assert_close(batch.targets()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(batch.rollout.behavior_logps, behavior, rtol=0, atol=0)
            return metrics

        services.worker.train = checked_train

    schema = None
    if rank == 0:
        names = [k for k, v in trainable.model.state_dict().items() if isinstance(v, torch.Tensor)]
        schema = {"names": torch.tensor(list(json.dumps(names).encode()), dtype=torch.uint8)}
    schema = TensorPacketBridge(group=control, source=0, device="cpu").broadcast(("names",), schema)
    weight_fields = tuple(json.loads(bytes(schema["names"].tolist()).decode()))
    coordinator = PPOCoordinator(
        groups=PPOProcessGroups(control, leaders, local, role, rollout_ranks[0], 0, 2),
        services=services,
        weight_fields=weight_fields,
        device=device,
        transport_device="cpu",
        config=config,
    )
    for iteration in range(2):
        started = time.perf_counter()
        metrics = coordinator.run_round()
        torch.cuda.synchronize()
        metrics["round_seconds"] = time.perf_counter() - started
        assert metrics["actor_version"] == metrics["critic_version"] == iteration + 1
        batch = coordinator.last_batch
        assert batch.rollout.sample_ids.tolist() == [13, 7]
        # Check identical packets across every role/rank, including reward return.
        evidence = torch.cat((batch.rollout.input_ids.cpu().flatten().float(), batch.rewards.cpu()))
        copies = [torch.empty_like(evidence) for _ in range(world)]
        dist.all_gather(copies, evidence, group=control)
        for other in copies:
            torch.testing.assert_close(other, evidence, rtol=0, atol=0)
        rounds.append(
            {
                "metrics": metrics,
                "tokens": batch.rollout.input_ids.tolist(),
                "rewards": batch.rewards.tolist(),
                "terminated": batch.rollout.terminated.tolist(),
            }
        )
    report = {
        "framework": args.framework,
        "rollout_backend": "vllm" if args.vllm_python else "megatron",
        "gpu_layout": args.gpu_layout,
        "cuda_device": device.index,
        "hidden_size": hidden_size,
        "behavior_logprob_max_errors": behavior_errors,
        "role": role,
        "rank": rank,
        "actor_tp": 2,
        "critic_tp": args.critic_tp,
        "rounds": rounds,
    }
    if role == "rollout":
        assert installed == [0, 1, 2]
        if rank == rollout_ranks[0]:
            assert sampled == [0, 1]
        report.update(installed=installed, sampled=sampled)
    else:
        final = snapshot(trainable, local)
        delta = max((final[k] - v).abs().max().item() for k, v in initial.items())
        assert delta > 0
        for name, value in snapshot(frozen, local).items():
            torch.testing.assert_close(value, frozen_initial[name], rtol=0, atol=0)
        assert all(p.grad is None for p in frozen.parameters())
        if engine is not None:
            assert engine.global_steps == 4
        report.update(
            parameter_delta=delta,
            frozen_unchanged=True,
            engine_steps=engine.global_steps if engine else None,
        )
        if rank in (0, 2):
            torch.save(final, args.output / f"{role}.pt")
    (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2))
    dist.barrier(group=control)
    if rank == 0:
        reports = [json.loads((args.output / f"rank{i}.json").read_text()) for i in range(world)]
        (args.output / "summary.json").write_text(json.dumps(reports, indent=2))
        print(
            f"PASS {args.framework}: Actor TP=2, Critic TP={args.critic_tp}, "
            "two complete PPO rounds",
            flush=True,
        )
    if vllm_client is not None:
        vllm_client.close()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["megatron", "deepspeed"], required=True)
    parser.add_argument("--critic-tp", type=int, choices=[1, 2], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-python", help="Isolated vLLM 0.15.1 Python executable")
    parser.add_argument("--gpu-layout", choices=["shared", "dedicated"], default="shared")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Each NCCL role group has at most one rank per GPU. Only CPU/Gloo groups
    # span roles; the default NCCL group is never used for tensor collectives.
    rank = int(os.environ["RANK"])
    gpu = rank if args.gpu_layout == "dedicated" else rank % 2
    if args.gpu_layout == "dedicated" and torch.cuda.device_count() < int(os.environ["WORLD_SIZE"]):
        raise ValueError("dedicated layout needs one visible GPU per coordinator rank")
    os.environ["LOCAL_RANK"] = str(gpu)
    torch.cuda.set_device(gpu)
    dist.init_process_group("nccl", timeout=timedelta(seconds=600 if args.vllm_python else 120))
    run(args)


if __name__ == "__main__":
    main()
