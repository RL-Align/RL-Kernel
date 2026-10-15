# SPDX-License-Identifier: Apache-2.0
"""Replay immutable real sampler inputs through native TP PPO engines."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from examples.post_training.tp_ppo.fixed_contract import (
    CONFIG,
    OPTIMIZER,
    identity,
    load_packet,
    load_targets,
    save_report,
    sha,
    verify_checkpoint,
    write_tensor,
)
from rl_engine.integrations.common.ppo import PPOBatch, PPORollout, PPOTrainingWorker
from rl_engine.integrations.common.tp_scalar import TPScalarExecutor
from rl_engine.integrations.engines.train.deepspeed.ppo import deepspeed_ppo_worker
from rl_engine.integrations.engines.train.megatron.ppo import megatron_parameter_layouts
from rl_engine.integrations.engines.train.megatron.ppo_precision import FP32MasterWeights
from rl_engine.integrations.engines.train.megatron.qwen3_checkpoint import (
    load_pretrained,
    megatron_to_hf,
)
from tests.distributed.tp.check_tp_ppo_frameworks import initialize_engine


def canonical(model, group, config, role, buffers=None):
    """Gather actual optimizer buffers using parameter partition metadata."""
    named = dict(model.named_parameters())
    result = {}
    size = dist.get_world_size(group)
    for name, p in model.state_dict().items():
        if not isinstance(p, torch.Tensor):
            continue
        param = named.get(name)
        if buffers is None:
            value = p.detach()
        elif param is None and name == "bias":
            value = torch.zeros_like(p, dtype=torch.float32)
        else:
            value = buffers.get(id(param))
            if value is None:
                raise ValueError(f"missing optimizer gradient: {name}")
        value = value.detach().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite tensor: {name}")
        if name == "bias":
            value = value.clone()
            dist.all_reduce(value, group=group)
        elif param is not None and getattr(param, "tensor_model_parallel", False):
            shards = [torch.empty_like(value) for _ in range(size)]
            dist.all_gather(shards, value, group=group)
            axis = param.partition_dim
            stride = 2 if name.endswith("mlp.linear_fc1.weight") else param.partition_stride
            parts = [v.chunk(stride, dim=axis) for v in shards]
            value = torch.cat([parts[r][i] for i in range(stride) for r in range(size)], dim=axis)
        elif size > 1:
            leader_value = value.clone()
            dist.broadcast(leader_value, src=0, group=group)
            mismatch = torch.tensor(int(not torch.equal(value, leader_value)), device=value.device)
            dist.all_reduce(mismatch, op=dist.ReduceOp.MAX, group=group)
            if mismatch.item():
                raise ValueError(f"replicated TP tensor differs: {name}")
        result[name] = value.cpu().clone()
    prefix = "model." if role == "actor" else "backbone."
    backbone = {n.removeprefix(prefix): v for n, v in result.items() if n.startswith(prefix)}
    result_hf = megatron_to_hf([backbone], config, policy=role == "actor")
    if role == "critic":
        result_hf.update({"score.weight": result["head.weight"], "score.bias": result["bias"]})
    return result_hf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--acceptance-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--framework", choices=["megatron", "deepspeed"], required=True)
    p.add_argument("--precision", choices=["fp32", "bf16"], required=True)
    p.add_argument("--role", choices=["actor", "critic"], required=True)
    a = p.parse_args()
    rank = int(os.environ["RANK"])
    size = int(os.environ["WORLD_SIZE"])
    if size not in (1, 2):
        raise ValueError("fixed rollout matrix uses TP1/TP2")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=size)
    group = parallel_state.get_tensor_model_parallel_group()
    dp = parallel_state.get_data_parallel_group()
    if rank == 0:
        a.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    report = identity()
    report.update(
        role=a.role,
        precision=a.precision,
        framework=a.framework,
        tp=size,
        rollout_manifest_sha256=sha(a.rollout / "manifest.json"),
        targets_manifest_sha256=sha(a.targets / "targets.json"),
        acceptance_manifest_sha256=sha(a.acceptance_manifest),
    )
    started = time.perf_counter()
    try:
        contract = json.loads(a.acceptance_manifest.read_text())
        if contract["status"] != "frozen" or any(
            contract[k] != report[k] for k in ("rollout_manifest_sha256", "targets_manifest_sha256")
        ):
            raise ValueError("wrong frozen rollout/target contract")
        verify_checkpoint(a.checkpoint)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        packet = {k: v.cuda() for k, v in load_packet(a.rollout).items()}
        targets = {k: v.cuda() for k, v in load_targets(a.targets).items()}
        batch = PPOBatch(PPORollout(**packet), **targets)
        batch.validate()
        dtype = torch.float32 if a.precision == "fp32" else torch.bfloat16
        model = load_pretrained(
            a.checkpoint, tp=size, policy=a.role == "actor", dtype=dtype, max_sequence_length=256
        )
        config = json.loads((a.checkpoint / "config.json").read_text())
        engine = (
            initialize_engine(
                model,
                tp_group=group,
                dp_group=dp,
                bf16=dtype == torch.bfloat16,
                optimizer=torch.optim.AdamW(
                    model.parameters(), **OPTIMIZER, foreach=False, fused=False
                ),
            )
            if a.framework == "deepspeed"
            else None
        )
        trainable = engine if engine is not None else model
        forward = (
            trainable
            if a.role == "actor"
            else TPScalarExecutor(
                trainable, group=group, leader=0, output_layout="local_sum"
            ).values
        )
        if engine is not None:
            worker = deepspeed_ppo_worker(
                engine, role=a.role, forward=forward, group=group, config=CONFIG
            )
        elif dtype == torch.bfloat16:
            master = FP32MasterWeights(
                model.parameters(),
                lambda ps: torch.optim.AdamW(ps, **OPTIMIZER, foreach=False, fused=False),
            )
            worker = PPOTrainingWorker(
                role=a.role,
                model=model,
                forward=forward,
                group=group,
                config=CONFIG,
                parameter_layouts=megatron_parameter_layouts(model),
                optimizer=master.optimizer,
                gradient_state=master.gradient_state,
                backward=torch.Tensor.backward,
                step=master.step,
            )
        else:
            worker = PPOTrainingWorker(
                role=a.role,
                model=model,
                forward=forward,
                group=group,
                config=CONFIG,
                parameter_layouts=megatron_parameter_layouts(model),
                optimizer=torch.optim.AdamW(
                    model.parameters(), **OPTIMIZER, foreach=False, fused=False
                ),
            )
        artifacts = {}
        before = {}

        def observe(event, update, buffers, value):
            nonlocal before
            if event == "before_clip":
                before = canonical(model, group, config, a.role)
                with torch.no_grad():
                    scores = forward(batch.rollout.inputs)
                selected = packet["action_mask"] if a.role == "actor" else packet["attention_mask"]
                # Changing masked token IDs must not alter valid predictions.
                from rl_engine.integrations.common.tp_scalar import TPModelInputs

                perturbed = packet["input_ids"].clone()
                perturbed[~packet["attention_mask"]] = 17
                with torch.no_grad():
                    other = forward(
                        TPModelInputs(perturbed, packet["attention_mask"], packet["position_ids"])
                    )
                torch.testing.assert_close(scores[selected], other[selected], rtol=0, atol=0)
                mapping = {
                    id(p0): (g if buffers is not None else p0.grad)
                    for (p0, _), g in zip(
                        worker.parameters,
                        buffers if buffers is not None else [None] * len(worker.parameters),
                        strict=True,
                    )
                }
                grads = canonical(model, group, config, a.role, mapping)
                padding_ids = packet["input_ids"][~packet["attention_mask"]].unique().cpu()
                valid_ids = packet["input_ids"][packet["attention_mask"]].unique().cpu()
                if any(int(v) in valid_ids.tolist() for v in padding_ids):
                    raise ValueError("padding gradient probe needs distinct padding IDs")
                if padding_ids.numel() and torch.count_nonzero(
                    grads["model.embed_tokens.weight"][padding_ids]
                ):
                    raise ValueError("padding polluted the embedding gradients")
                if rank == 0:
                    artifacts[f"{update}/scores"] = write_tensor(
                        a.output, f"{update}/scores", scores[selected]
                    )
                    artifacts[f"{update}/loss"] = write_tensor(
                        a.output, f"{update}/loss", torch.tensor(value, dtype=torch.float64)
                    )
                    for n, g in grads.items():
                        artifacts[f"{update}/gradient/{n}"] = write_tensor(
                            a.output, f"{update}/gradient/{n}", g
                        )
                del grads
            elif event == "after_clip":
                squared = torch.zeros((), device="cuda", dtype=torch.float64)
                actual_buffers = (
                    buffers if buffers is not None else [p.grad for p, _ in worker.parameters]
                )
                for (parameter, layout), grad in zip(
                    worker.parameters, actual_buffers, strict=True
                ):
                    if grad is not None and (layout == "sharded" or rank == 0):
                        squared += grad.double().square().sum()
                dist.all_reduce(squared, group=group)
                if rank == 0:
                    artifacts[f"{update}/grad_norm"] = write_tensor(
                        a.output, f"{update}/grad_norm", torch.tensor(value, dtype=torch.float64)
                    )
                    artifacts[f"{update}/clipped_norm"] = write_tensor(
                        a.output, f"{update}/clipped_norm", squared.sqrt()
                    )
            elif event == "after_step":
                optimizer = worker.optimizer
                while hasattr(optimizer, "optimizer"):
                    optimizer = optimizer.optimizer
                for state in optimizer.state.values():
                    for tensor in state.values():
                        if (
                            isinstance(tensor, torch.Tensor)
                            and tensor.is_floating_point()
                            and tensor.dtype != torch.float32
                        ):
                            raise ValueError("optimizer state is not FP32")
                after = canonical(model, group, config, a.role)
                if rank == 0:
                    for n, v in after.items():
                        artifacts[f"{update}/update/{n}"] = write_tensor(
                            a.output, f"{update}/update/{n}", v.float() - before[n].float()
                        )
                before = {}

        worker.update_observer = observe
        frozen = {k: v.clone() for k, v in batch.targets().items()}
        metrics = worker.train(batch)
        for k, v in frozen.items():
            torch.testing.assert_close(v, batch.targets()[k], atol=0, rtol=0)
        report.update(
            status="fixed_candidate_complete",
            artifacts=artifacts,
            metrics=metrics,
            effective_action_tokens=int(packet["action_mask"].sum()),
            padding_invariant=True,
        )
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(
            seconds=time.perf_counter() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        (a.output / f"rank{rank}.json").write_text(
            json.dumps({k: v for k, v in report.items() if k != "artifacts"}, indent=2) + "\n"
        )
        if rank == 0:
            save_report(a.output, report)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
