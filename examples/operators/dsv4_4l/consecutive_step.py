# SPDX-License-Identifier: Apache-2.0
"""Collect content/state witnesses around the real TrainingWorker update."""

import os

import torch

from examples.operators.dsv4_4l.actor_service import batch, digest
from examples.operators.dsv4_4l.distributed_evidence import ddp_step_evidence, tensor_record
from examples.operators.dsv4_4l.optimizer_evidence import (
    optimizer_snapshot,
    verify_optimizer_transition,
)
from examples.operators.dsv4_4l.step_evidence import training_metrics
from rl_engine.integrations.engines.train.megatron.checkpoint import (
    export_native_checkpoint_parameters,
)


def consecutive_step(
    worker,
    engine,
    model,
    boundaries,
    record,
    cases,
    *,
    root,
    local_root,
    command,
    rank,
    output_rank,
    previous_optimizer,
    gather,
    save,
):
    version = command["step_version"]
    if engine.rlk_weight_version != version - 1 or version not in (1, 2, 3):
        raise ValueError("consecutive step must advance the actual actor version by one")
    target = root / "steps" / f"v{version}"
    local = local_root / f"step-v{version}"
    local.mkdir()
    target.mkdir(parents=True, exist_ok=True)
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    mapped = {id(p) for group in model._rlk_checkpoint_export_parameters.values() for p in group}
    missing = [name for name, p in parameters.items() if id(p) not in mapped]
    if missing:
        raise RuntimeError(f"trainable parameters absent from actual state export: {missing}")
    before_parameters = {name: tensor_record(p) for name, p in parameters.items()}
    before_state = {
        name: tensor_record(p) for name, p in export_native_checkpoint_parameters(model)
    }
    optimizer_before = optimizer_snapshot(engine)
    with ddp_step_evidence(engine, model) as gradients:
        out = worker.train_batch(batch(cases))
    optimizer_after = optimizer_snapshot(engine)
    verify_optimizer_transition(
        optimizer_before, optimizer_after, version=version, previous=previous_optimizer
    )
    if engine.rlk_weight_version != version:
        raise RuntimeError("real native optimizer failed to advance the actual actor version")
    if (
        set(gradients["local_gradients"]) != set(parameters)
        or set(gradients["reduced_gradients"]) != set(parameters)
        or not all(x["finite"] for x in gradients["local_gradients"].values())
    ):
        raise RuntimeError("trainable parameter gradient coverage/finite check failed")
    after_parameters = {name: tensor_record(p) for name, p in parameters.items()}
    changed_parameters = [
        n for n in parameters if before_parameters[n]["sha256"] != after_parameters[n]["sha256"]
    ]
    if not changed_parameters:
        raise RuntimeError("real optimizer changed no trainable parameters")
    after_state = {name: tensor_record(p) for name, p in export_native_checkpoint_parameters(model)}
    if set(before_state) != set(after_state):
        raise RuntimeError("actual synchronizable state inventory changed")
    gradients["actual_trainable_parameters"] = list(parameters)
    save(local / "gradient-evidence.json", gradients)
    save(local / "pipeline-boundary-evidence.json", boundaries.records)
    state_evidence = dict(before=optimizer_before, after=optimizer_after, continuity_verified=True)
    save(local / "optimizer-evidence.json", state_evidence)
    names = {id(p): n for n, p in model.named_parameters()}
    inventory = {
        key: dict(
            description=value,
            parameters=[names[id(p)] for p in model._rlk_checkpoint_export_parameters.get(key, ())],
        )
        for key, value in after_state.items()
    }
    router_bias = {
        n: tensor_record(b) for n, b in model.named_buffers() if n.endswith("expert_bias")
    }
    router_exports = {k: v for k, v in after_state.items() if k.endswith(".ffn.gate.bias")}
    if model._rlk_checkpoint_stage.layer_ids == (3,) and not router_exports:
        raise RuntimeError("learned-router bias missing from actual synchronization inventory")
    save(
        local / "state-change-evidence.json",
        dict(
            before_parameters=before_parameters,
            after_parameters=after_parameters,
            changed_parameters=changed_parameters,
            before_sync_state=before_state,
            after_sync_state=after_state,
            synchronization_inventory=inventory,
            router_bias_buffers=router_bias,
            router_bias_exports=router_exports,
        ),
    )
    manifest, export = {}, {}
    parameters_out, _ = engine.get_per_tensor_param(base_sync_done=True)
    for key, value in parameters_out:
        cpu = value.detach().cpu().clone()
        manifest[key] = dict(shape=list(cpu.shape), dtype=str(cpu.dtype), sha256=digest(cpu))
        if rank == 0:
            export[key] = cpu
    record.update(
        pid=os.getpid(),
        weight_version=version,
        previous_weight_version=version - 1,
        before=before_parameters[changed_parameters[0]]["sha256"],
        after=after_parameters[changed_parameters[0]]["sha256"],
        changed_parameters=changed_parameters,
        trainable_parameters=len(parameters),
        trainable_elements=sum(p.numel() for p in parameters.values()),
        missing_export_parameters=missing,
        exported_tensors=len(manifest),
        local_sync_state=after_state,
        parameter_export_completed=True,
        source="actual persistent TrainingWorker.train_batch/optimizer.step",
        optimizer_evidence=state_evidence,
        gradient_evidence=str(local.relative_to(root) / "gradient-evidence.json"),
        state_change_evidence=str(local.relative_to(root) / "state-change-evidence.json"),
        gradient_witness=dict(
            backward_executed=True,
            gradient_finite=all(v["finite"] for v in gradients["reduced_gradients"].values()),
            gradient_nonzero=any(v["nonzero"] for v in gradients["reduced_gradients"].values()),
        ),
        metrics=training_metrics(out) if output_rank else None,
        pipeline_boundaries=boundaries.records,
        native_ddp_finish_executed=bool(gradients["native_ddp_finish_calls"]),
        dp_gradient_reduction_verified=engine.get_data_parallel_size() > 1
        and bool(gradients["native_ddp_finish_calls"]),
        cross_replica_reduction_executed=engine.get_data_parallel_size() > 1,
        dp_replicas=gradients["dp_replicas"],
        different_local_gradients=gradients["different_local_gradients"],
    )
    if rank == 0:
        torch.save(
            dict(
                weight_version=version,
                previous_weight_version=version - 1,
                weights=export,
                manifest=manifest,
            ),
            target / "actor-update.pt",
        )
        save(target / "actor-update-manifest.json", manifest)
    stages = gather(record)
    if rank == 0:
        save(target / "actor-distributed-step.json", stages)
        save(
            target / "actor-step-status.json",
            dict(
                optimizer_returned=True,
                parameter_changed=True,
                weight_version=version,
                all_stage_updates_verified=all(r["before"] != r["after"] for r in stages),
            ),
        )
    return record, optimizer_after
