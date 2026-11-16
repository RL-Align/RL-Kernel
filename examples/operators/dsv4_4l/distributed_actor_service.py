# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- install the pinned import ABI adapters before real worker imports.
# flake8: noqa: E402
"""torchrun service using actual verl TrainingWorkers and native PP/DDP scheduling."""

import json
import os
import time
from pathlib import Path

from rl_engine.integrations.orchestrators.verl.native_imports import install_native_import_adapters

install_native_import_adapters()

import torch
import torch.distributed as dist
from megatron.core import parallel_state

from examples.operators.dsv4_4l.actor_service import batch, build_actor_worker, digest
from examples.operators.dsv4_4l.distributed_evidence import (
    PipelineBoundaryEvidence,
    ddp_step_evidence,
    distributed_token_loss,
)
from examples.operators.dsv4_4l.native_trace import install
from examples.operators.dsv4_4l.step_evidence import training_metrics

ROOT = Path("/outputs")


def save(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def gather(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", init_method="env://")
    rank = dist.get_rank()
    local_root = ROOT / f"actor-rank-{rank}"
    local_root.mkdir()
    worker, engine, model = build_actor_worker()
    dp_rank = engine.get_data_parallel_rank()
    output_rank = engine.is_mp_src_rank_with_outputs()
    from rl_engine.integrations.common.device_identity import hip_device_identity

    topology = dict(
        device_identity=hip_device_identity(),
        rank=rank,
        pid=os.getpid(),
        logical_gpu=torch.cuda.current_device(),
        requested_physical_gpu=os.environ["DSV4_ACTOR_PHYSICAL_GPUS"].split(",")[
            int(os.environ["LOCAL_RANK"])
        ],
        pp_rank=parallel_state.get_pipeline_model_parallel_rank(),
        dp_rank=dp_rank,
        pp_ranks=dist.get_process_group_ranks(parallel_state.get_pipeline_model_parallel_group()),
        dp_ranks=dist.get_process_group_ranks(engine.get_data_parallel_group()),
        original_layer_ids=list(model._rlk_checkpoint_stage.layer_ids),
        model_class=str(type(model)),
        weight_version=engine.rlk_weight_version,
    )
    topologies = gather(topology)
    if rank == 0:
        save(ROOT / "actor-topology.json", topologies)
        save(ROOT / "actor-ready.json", dict(ranks=topologies, version=engine.rlk_weight_version))
    boundaries = PipelineBoundaryEvidence(model, dp_rank)

    def loss(model_output, data, dp_group):
        return distributed_token_loss(model_output, data, dp_size=engine.get_data_parallel_size())

    worker.set_loss_fn(loss)
    done = set()
    while True:
        for path in sorted((ROOT / "actor-commands").glob("*.json")):
            if path.name in done:
                continue
            command = json.loads(path.read_text())
            done.add(path.name)
            name, action = command["id"], command["action"]
            if action == "quit":
                dist.barrier()
                dist.destroy_process_group()
                return
            cases = command.get("cases_by_dp", {}).get(str(dp_rank), command["cases"])
            engine.rlk_logical_cases = cases
            boundaries.begin(name)
            record = dict(
                id=name,
                rank=rank,
                pp_rank=topology["pp_rank"],
                dp_rank=dp_rank,
                weight_version=engine.rlk_weight_version,
            )
            if action == "infer":
                stop = install(model, local_root / f"trace-{name}")
                try:
                    out = worker.infer_batch(batch(cases))
                finally:
                    stop()
                selected = None
                if output_rank:
                    raw = out["raw_log_probs"].values().detach().cpu()
                    assert raw.dtype == torch.float32 and bool(torch.isfinite(raw).all())
                    lengths = [
                        len(c["prompt_token_ids"]) + len(c["response_token_ids"]) for c in cases
                    ]
                    pieces = list(raw.split(lengths))
                    selected = [
                        value[
                            len(c["prompt_token_ids"])
                            - 1 : len(c["prompt_token_ids"])
                            - 1
                            + len(c["response_token_ids"])
                        ].clone()
                        for value, c in zip(pieces, cases, strict=True)
                    ]
                    assert all(v.numel() > 0 for v in selected)
                    data = dict(record, cases=cases, raw_log_probs=raw, selected=selected)
                    torch.save(data, local_root / f"actor-{name}.pt")
                    if dp_rank == 0:
                        torch.save(data, ROOT / f"actor-{name}.pt")
                replicas = gather(
                    None if selected is None else [v.view(torch.int32).tolist() for v in selected]
                )
                scores = [v for v in replicas if v is not None]
                assert len(scores) == engine.get_data_parallel_size() and all(
                    v == scores[0] for v in scores
                )
                record.update(
                    source="actual TrainingWorker.infer_batch with native PP",
                    compared_tokens=[len(c["response_token_ids"]) for c in cases],
                    dp_scores_verified=True,
                )
            elif action == "step":
                trainable = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
                covered = {
                    id(p)
                    for group in model._rlk_checkpoint_export_parameters.values()
                    for p in group
                }
                missing = [n for i, n in trainable.items() if i not in covered]
                if missing:
                    raise RuntimeError(f"trainable stage parameters absent from export: {missing}")
                witness = (
                    model.decoder.final_norm.weight
                    if model.post_process
                    else model.decoder.layers[0].hyper_connection.mapping_proj.weight
                )
                before = digest(witness)
                with ddp_step_evidence(engine, model) as gradients:
                    out = worker.train_batch(batch(cases))
                after = digest(witness)
                if before == after:
                    raise RuntimeError("actual optimizer did not change this stage witness")
                save(local_root / "gradient-evidence.json", gradients)
                save(local_root / "pipeline-boundary-evidence.json", boundaries.records)
                parameters, _ = engine.get_per_tensor_param(base_sync_done=True)
                manifest, export = {}, {}
                for key, value in parameters:
                    cpu = value.detach().cpu().clone()
                    manifest[key] = dict(
                        shape=list(cpu.shape), dtype=str(cpu.dtype), sha256=digest(cpu)
                    )
                    if rank == 0:
                        export[key] = cpu
                record.update(
                    weight_version=engine.rlk_weight_version,
                    before=before,
                    after=after,
                    trainable_parameters=len(trainable),
                    missing_export_parameters=missing,
                    exported_tensors=len(manifest),
                    parameter_export_completed=True,
                    source="actual TrainingWorker.train_batch/optimizer.step",
                    gradient_witness=dict(
                        backward_executed=bool(gradients["local_gradients"]),
                        gradient_finite=all(
                            v["finite"] for v in gradients["reduced_gradients"].values()
                        ),
                        gradient_nonzero=any(
                            v["nonzero"] for v in gradients["reduced_gradients"].values()
                        ),
                    ),
                    metrics=training_metrics(out) if output_rank else None,
                    pipeline_boundaries=boundaries.records,
                    dp_gradient_reduction_verified=bool(gradients["native_ddp_finish_calls"]),
                    dp_replicas=gradients["dp_replicas"],
                    different_local_gradients=gradients["different_local_gradients"],
                )
                if rank == 0:
                    torch.save(
                        dict(
                            weight_version=engine.rlk_weight_version,
                            weights=export,
                            manifest=manifest,
                        ),
                        ROOT / "actor-update.pt",
                    )
                    save(ROOT / "actor-update-manifest.json", manifest)
                stages = gather(record)
                if rank == 0:
                    save(ROOT / "actor-distributed-step.json", stages)
                    final = next(r for r in stages if r["metrics"] is not None)
                    record["metrics"] = final["metrics"]
                    record["all_stage_updates_verified"] = all(
                        r["before"] != r["after"] for r in stages
                    )
                    save(
                        ROOT / "actor-step-status.json",
                        dict(
                            optimizer_returned=True,
                            parameter_changed=True,
                            weight_version=engine.rlk_weight_version,
                            all_stage_updates_verified=record["all_stage_updates_verified"],
                        ),
                    )
            else:
                raise ValueError(action)
            save(local_root / f"actor-{name}.json", record)
            dist.barrier()
            if rank == 0:
                save(ROOT / f"actor-{name}.json", record)
        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        save(
            ROOT / f'actor-error-rank-{os.environ.get("RANK", "unknown")}.json',
            dict(type=type(error).__name__, error=str(error)),
        )
        save(
            ROOT / "actor-error.json",
            dict(type=type(error).__name__, error=str(error), rank=os.environ.get("RANK")),
        )
        raise
