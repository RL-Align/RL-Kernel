# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Real-framework, independent-role PPO scheduling on shared or dedicated GPUs.

Pretrained Qwen3 acceptance. Requires frozen numerical thresholds and real vLLM weights.
Role groups use NCCL; explicit CPU/Gloo transport permits GPU sharing across
roles. This correctness harness does not measure dedicated-GPU performance.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from rl_engine.integrations.common.ppo import PPOTrainingWorker
from rl_engine.integrations.common.ppo_coordinator import (
    PPOCoordinator,
    PPOProcessGroups,
    PPOServices,
)
from rl_engine.integrations.common.tp_scalar import TPScalarExecutor
from rl_engine.integrations.common.weights import TensorPacketBridge
from rl_engine.integrations.engines.train.deepspeed.ppo import deepspeed_ppo_worker
from rl_engine.integrations.engines.train.megatron.ppo import megatron_parameter_layouts
from tests.distributed.tp.check_tp_ppo_frameworks import initialize_engine, snapshot


def run(args):
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    rank, world = dist.get_rank(), dist.get_world_size()
    timeout = 600 if args.vllm_python else 120
    hidden_size = 4096
    if args.vllm_python and args.critic_tp != 1:
        raise ValueError("vLLM harness currently requires one rollout worker (critic-tp=1)")
    assert world == args.actor_tp + 2
    critic_leader = args.actor_tp
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.actor_tp, distributed_timeout_minutes=timeout // 60
    )
    control = dist.new_group(list(range(world)), backend="gloo", timeout=timedelta(seconds=timeout))
    critic_ranks = [critic_leader]
    rollout_ranks = [critic_leader + 1]
    leaders = dist.new_group(
        [0, critic_leader, rollout_ranks[0]], backend="gloo", timeout=timedelta(seconds=timeout)
    )
    singleton = None
    for i in range(world):
        group = dist.new_group([i], backend="nccl", timeout=timedelta(seconds=timeout))
        if rank == i:
            singleton = group
    role = "actor" if rank < args.actor_tp else "critic" if rank in critic_ranks else "rollout"
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
    from examples.post_training.tp_ppo.fixed_contract import CONFIG, OPTIMIZER

    config = CONFIG
    services = PPOServices()
    installed, sampled, rounds = [], [], []
    behavior_errors = []
    engine = frozen = trainable = vllm_client = None
    if role == "rollout" and args.vllm_python:
        from tests.distributed.tp.vllm_ppo_client import VllmPPOClient

        vllm_client = VllmPPOClient(
            args.vllm_python,
            args.output / "vllm",
            device,
            checkpoint=args.checkpoint,
            tp=args.vllm_tp,
            cache_logprob_atol=args.thresholds["behavior_logprob"]["atol"],
        )
        services.install, services.sample = vllm_client.install, vllm_client.sample
        installed, sampled = vllm_client.installed, vllm_client.sampled
    else:
        from rl_engine.integrations.engines.train.megatron.qwen3_checkpoint import load_pretrained

        dtype = torch.bfloat16
        trainable = load_pretrained(
            args.checkpoint,
            tp=tp,
            policy=role == "actor",
            dtype=dtype,
            pg_collection=pgs,
            max_sequence_length=256,
        )
        frozen = load_pretrained(
            args.checkpoint,
            tp=tp,
            policy=role == "actor",
            dtype=dtype,
            pg_collection=pgs,
            max_sequence_length=256,
        ).requires_grad_(False)
        initial = snapshot(trainable, local)
        frozen_initial = snapshot(frozen, local)
        engine = (
            initialize_engine(
                trainable,
                tp_group=local,
                dp_group=singleton,
                bf16=True,
                optimizer=torch.optim.AdamW(
                    trainable.parameters(), **OPTIMIZER, foreach=False, fused=False
                ),
            )
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
                training_model, group=local, leader=critic_leader, output_layout="local_sum"
            )
            services.reward = TPScalarExecutor(
                frozen, group=local, leader=critic_leader, output_layout="local_sum"
            )
            forward = services.critic.values
        if engine is not None:
            services.worker = deepspeed_ppo_worker(
                engine, role=role, forward=forward, group=local, config=config
            )
        else:
            from rl_engine.integrations.engines.train.megatron.ppo_precision import (
                FP32MasterWeights,
            )

            master = FP32MasterWeights(
                trainable.parameters(),
                lambda params: torch.optim.AdamW(params, **OPTIMIZER, foreach=False, fused=False),
            )
            services.worker = PPOTrainingWorker(
                role=role,
                model=trainable,
                forward=forward,
                optimizer=master.optimizer,
                parameter_layouts=megatron_parameter_layouts(trainable),
                group=local,
                config=config,
                gradient_state=master.gradient_state,
                backward=torch.Tensor.backward,
                step=master.step,
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
                    atol=args.thresholds["behavior_logprob"]["atol"],
                    rtol=args.thresholds["behavior_logprob"]["rtol"],
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
        groups=PPOProcessGroups(control, leaders, local, role, rollout_ranks[0], 0, critic_leader),
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
        metrics["effective_action_tokens"] = int(batch.rollout.action_mask.sum())
        if rank == 0:
            torch.save(
                {
                    "rollout": {k: v.cpu() for k, v in batch.rollout.packet().items()},
                    "targets": {k: v.cpu() for k, v in batch.targets().items()},
                },
                args.output / f"round-{iteration}-fixed-rollout.pt",
            )
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
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "thresholds": args.thresholds,
        "thresholds_sha256": args.thresholds_sha256,
        "rollout_backend": "vllm" if args.vllm_python else "megatron",
        "gpu_layout": args.gpu_layout,
        "cuda_device": device.index,
        "hidden_size": hidden_size,
        "behavior_logprob_max_errors": behavior_errors,
        "role": role,
        "rank": rank,
        "actor_tp": args.actor_tp,
        "vllm_tp": args.vllm_tp,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "critic_tp": args.critic_tp,
        "rounds": rounds,
    }
    if role == "rollout":
        assert installed == [0, 1, 2]
        if rank == rollout_ranks[0]:
            assert sampled == [0, 1]
        negative_controls = vllm_client._request({"op": "negative_controls"})
        report.update(installed=installed, sampled=sampled, negative_controls=negative_controls)
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
        if rank in (0, critic_leader):
            torch.save(final, args.output / f"{role}.pt")
    (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2))
    dist.barrier(group=control)
    if rank == 0:
        reports = [json.loads((args.output / f"rank{i}.json").read_text()) for i in range(world)]
        (args.output / "summary.json").write_text(json.dumps(reports, indent=2))
        print(
            f"PASS {args.framework}: Actor TP={args.actor_tp}, Critic TP={args.critic_tp}, "
            "two complete PPO rounds",
            flush=True,
        )
    if vllm_client is not None:
        vllm_client.close()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["megatron", "deepspeed"], required=True)
    parser.add_argument("--actor-tp", type=int, choices=[1, 2], required=True)
    parser.add_argument("--critic-tp", type=int, choices=[1], default=1)
    parser.add_argument("--vllm-tp", type=int, choices=[1, 2], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--vllm-python", required=True, help="Isolated vLLM 0.15.1 Python executable"
    )
    parser.add_argument("--gpu-layout", choices=["shared", "dedicated"], default="shared")
    args = parser.parse_args()
    from examples.post_training.tp_ppo.calibrate_qwen3 import sha

    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("formal Qwen3 execution requires clean source")
    manifest_path = (
        Path(__file__).resolve().parents[3] / "examples/post_training/tp_ppo/qwen3-8b-manifest.json"
    )
    for record in json.loads(manifest_path.read_text())["files"]:
        artifact = args.checkpoint / record["path"]
        if artifact.stat().st_size != record["size"] or sha(artifact) != record["sha256"]:
            raise ValueError(f"pretrained checkpoint identity changed: {artifact}")
    args.thresholds_sha256 = sha(args.thresholds)
    args.thresholds = json.loads(args.thresholds.read_text())
    if args.thresholds.get("model_revision") != "b968826d9c46dd6066d109eabc6255188de91218":
        raise ValueError("thresholds are not for the pinned Qwen3 checkpoint")
    if not args.thresholds.get("fp32_reference_sha256"):
        raise ValueError("independent FP32 calibration evidence is required")
    if args.actor_tp + args.critic_tp + args.vllm_tp != 4 or args.gpu_layout != "dedicated":
        raise ValueError("pretrained acceptance requires the dedicated four-GPU topology")
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
