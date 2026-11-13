# SPDX-License-Identifier: Apache-2.0
"""Actual TrainingWorker command service; no independent training-engine substitute."""

import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import torch
from tensordict import TensorDict

from rl_engine.integrations.orchestrators.verl.native_engine import RLKNativeMcoreEngineConfig
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.profiler.config import ProfilerConfig
from verl.workers.config import HFModelConfig, McoreOptimizerConfig, TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker

from examples.operators.dsv4_4l.native_trace import install

ROOT = Path("/outputs")
COMMANDS = ROOT / "actor-commands"


def digest(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def batch(cases):
    lengths = [len(c["prompt_token_ids"]) + len(c["response_token_ids"]) for c in cases]
    if len(set(lengths)) != 1:
        raise ValueError("initial padded-free native batch requires equal full lengths")
    inputs = [
        torch.tensor(c["prompt_token_ids"] + c["response_token_ids"], dtype=torch.int64)
        for c in cases
    ]
    masks = [
        torch.cat(
            (
                torch.zeros(len(c["prompt_token_ids"]) - 1),
                torch.ones(len(c["response_token_ids"])),
                torch.zeros(1),
            )
        )
        for c in cases
    ]
    result = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(inputs, layout=torch.jagged),
            "loss_mask": torch.nested.as_nested_tensor(masks, layout=torch.jagged),
            "temperature": torch.ones(len(cases)),
        },
        batch_size=[len(cases)],
    )
    tu.assign_non_tensor(
        result,
        global_token_num=None,
        compute_loss=False,
        disable_auto_offload=True,
        micro_batch_size_per_gpu=len(cases),
    )
    return result


def main():
    os.environ.update(
        WORLD_SIZE="1",
        RANK="0",
        LOCAL_WORLD_SIZE="1",
        LOCAL_RANK="0",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29648",
        DIST_INIT_METHOD="file:///cache/tmp/dsv4-actor-rendezvous",
    )
    cfg = TrainingWorkerConfig(
        model_type="language_model",
        model_config=HFModelConfig(path="/models/4l", use_remove_padding=False),
        engine_config=RLKNativeMcoreEngineConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            context_parallel_size=1,
            sequence_parallel=False,
            param_offload=False,
            optimizer_offload=False,
            forward_only=False,
            use_distributed_optimizer=False,
            use_dynamic_bsz=False,
            micro_batch_size_per_gpu=1,
            infer_micro_batch_size_per_gpu=1,
            override_transformer_config={
                "gradient_accumulation_fusion": False,
                "recompute_granularity": None,
                "apply_rope_fusion": False,
                "moe_permute_fusion": False,
                "moe_shared_expert_overlap": False,
            },
        ),
        optimizer_config=McoreOptimizerConfig(
            lr=1e-4, total_training_steps=2, weight_decay=0.0, clip_grad=1.0
        ),
        checkpoint_config=CheckpointConfig(),
        profiler_config=ProfilerConfig(tool=None, save_path="/outputs", tool_config={}),
    )
    worker = TrainingWorker(cfg)
    worker.reset()
    engine = worker.engine
    model = engine._rlk_models[0]
    from rl_engine.integrations.engines.train.megatron.native_sparse import bind_native_sparse
    from rl_engine.integrations.engines.train.megatron.native_inverse_rope import (
        bind_native_inverse_rope,
    )

    for layer in model.decoder.layers[::2]:
        attention = layer.inner_layer.self_attention
        bind_native_sparse(attention.core_attention)
        bind_native_inverse_rope(attention)
        from rl_engine.integrations.engines.train.megatron.native_compressor_projection import (
            bind_compressor_projection,
        )

        if getattr(attention.core_attention, "compressor", None) is not None:
            bind_compressor_projection(attention.core_attention.compressor)
            from rl_engine.integrations.engines.train.megatron.native_compressor import (
                bind_native_compressor,
            )

            bind_native_compressor(
                attention.core_attention.compressor, attention._rlk_qkv_preprocess
            )
        if getattr(attention.core_attention, "indexer", None) is not None:
            bind_compressor_projection(attention.core_attention.indexer.compressor)
    from rl_engine.integrations.engines.train.megatron.native_hash_route import (
        bind_native_hash_route,
    )

    for layer in model.decoder.layers[1::2]:
        router = layer.inner_layer.mlp.router
        if router.is_hash_layer:
            bind_native_hash_route(router)
        else:
            from rl_engine.integrations.engines.train.megatron.native_learned_route import (
                bind_native_learned_route,
            )

            bind_native_learned_route(router)
    from rl_engine.integrations.engines.train.megatron.native_w4a16 import bind_native_w4a16

    for layer in model.decoder.layers[1::2]:
        bind_native_w4a16(layer.inner_layer.mlp)
        from rl_engine.integrations.engines.train.megatron.native_combine import bind_native_combine

        bind_native_combine(layer.inner_layer.mlp)
    print("ACTUAL_WORKER", type(worker), inspect.getfile(type(worker)), flush=True)
    print("ACTUAL_ENGINE", type(engine), inspect.getfile(type(engine)), flush=True)
    print("ACTUAL_MODEL", type(model), inspect.getfile(type(model)), flush=True)
    (ROOT / "actor-ready.json").write_text(
        json.dumps({"pid": os.getpid(), "version": engine.rlk_weight_version})
    )

    def loss(model_output, data, dp_group):
        raw, mask = model_output["raw_log_probs"].values(), data["loss_mask"].values()
        return -(raw * mask).sum() / mask.sum(), {}

    worker.set_loss_fn(loss)
    done = set()
    while True:
        for command_path in sorted(COMMANDS.glob("*.json")):
            if command_path.name in done:
                continue
            command = json.loads(command_path.read_text())
            name, action = command["id"], command["action"]
            done.add(command_path.name)
            if action == "quit":
                torch.distributed.destroy_process_group()
                return
            cases = command["cases"]
            record = {"id": name, "weight_version": engine.rlk_weight_version}
            if action == "infer":
                stop = install(model, ROOT / f"actor-trace-{name}")
                out = worker.infer_batch(batch(cases))
                stop()
                raw = out["raw_log_probs"].values().detach().cpu()
                assert raw.dtype == torch.float32
                lengths = [len(c["prompt_token_ids"]) + len(c["response_token_ids"]) for c in cases]
                pieces = list(raw.split(lengths))
                selected = [
                    piece[
                        len(c["prompt_token_ids"]) - 1 : len(c["prompt_token_ids"])
                        - 1
                        + len(c["response_token_ids"])
                    ].clone()
                    for c, piece in zip(cases, pieces, strict=True)
                ]
                assert all(v.numel() > 0 and bool(torch.isfinite(v).all()) for v in selected)
                torch.save(
                    dict(record, cases=cases, raw_log_probs=raw, selected=selected),
                    ROOT / f"actor-{name}.pt",
                )
                record.update(
                    compared_tokens=[v.numel() for v in selected],
                    source="actual TrainingWorker.infer_batch",
                )
            elif action == "step":
                before = digest(model.decoder.final_norm.weight)
                out = worker.train_batch(batch(cases))
                metrics = out["metrics"].data
                (ROOT / "actor-step-metrics.json").write_text(
                    json.dumps(metrics, indent=2, default=str) + "\n"
                )
                grad_norm = metrics.get("grad_norm", metrics.get("actor/grad_norm"))
                if grad_norm is not None and not (
                    float(grad_norm) > 0 and float(grad_norm) < float("inf")
                ):
                    raise RuntimeError("actual optimizer gradient norm must be finite and nonzero")
                after = digest(model.decoder.final_norm.weight)
                if before == after:
                    raise RuntimeError(
                        "real optimizer step did not change the tracked trainable final norm"
                    )
                weights, _ = engine.get_per_tensor_param(base_sync_done=True)
                export = {k: v.detach().cpu().clone() for k, v in weights}
                torch.save(
                    {"weight_version": engine.rlk_weight_version, "weights": export},
                    ROOT / "actor-update.pt",
                )
                record.update(
                    weight_version=engine.rlk_weight_version,
                    before=before,
                    after=after,
                    exported_tensors=len(export),
                    source="actual TrainingWorker.train_batch/optimizer.step",
                    metrics=metrics,
                )
            else:
                raise ValueError(action)
            (ROOT / f"actor-{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        (ROOT / "actor-error.json").write_text(
            json.dumps({"type": type(error).__name__, "error": str(error)})
        )
        raise
