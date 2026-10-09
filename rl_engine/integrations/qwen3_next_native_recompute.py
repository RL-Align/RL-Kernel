# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Independent native VIME full-model recomputation from frozen tokens only."""

import hashlib
import os
import sys
import traceback
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.integrations import qwen3_next_subnode as subnode
from rl_engine.testing.qwen3_next_l2 import (
    embedding_digest,
    layer_digest,
    load_tensor_artifact,
    save_tensor_artifact,
    validate_model_artifact,
    validate_native_worker_traces,
)
from rl_engine.testing.tensor_identity import assert_tensor_bitwise_equal as exact


def response_rows(value, request):
    """Keep original BF16 values at P-1..P+14 for every packed sample."""
    batch = len(request["prompt_tokens"])
    length = len(request["consumed_input_tokens"][0])
    if value.dtype not in (torch.bfloat16, torch.int64) or value.numel() == 0:
        raise ValueError("Response traces must preserve nonempty original BF16/int64 storage")
    value = value.reshape(-1, value.shape[-1])
    if value.shape[0] < batch * length:
        raise ValueError("Packed model output does not cover every frozen sample")
    positions = torch.tensor(request["prediction_positions"], device=value.device)
    if positions.shape != (16,) or positions[-1] != length - 1:
        raise ValueError("Response positions must end before consuming the last target token")
    return torch.stack([value[index * length + positions] for index in range(batch)])


def _rank_main(rank, cases, staging):
    os.environ.update(LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE="4")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{staging}/rendezvous",
        rank=rank,
        world_size=4,
        timeout=timedelta(minutes=30),
        device_id=torch.device("cuda", rank),
    )
    try:
        from megatron.core import mpu
        from megatron.core.packed_seq_params import PackedSeqParams
        from qwen3_next_contract import atomic_json

        from rl_engine.integrations.qwen3_next_binary import imported_provider_identity
        from rl_engine.integrations.qwen3_next_forward import shared_logprob
        from rl_engine.integrations.qwen3_next_logprob import qwen_logprob_contract
        from rl_engine.integrations.qwen3_next_vime_runtime import create_actor

        first = cases[0][0]
        args, model, optimizer, scheduler = create_actor(
            first["identity"]["checkpoint"], training=False
        )
        if optimizer is not None or scheduler is not None or len(model) != 1:
            raise ValueError("L2 recompute requires one native actor without an optimizer")
        raw = model[0]
        while hasattr(raw, "module"):
            raw = raw.module
        if len(raw.decoder.layers) != 48:
            raise ValueError("Native recompute must execute all 48 decoder layers")
        model[0].eval()
        provider_binary = imported_provider_identity()
        group = mpu.get_tensor_model_parallel_group()
        for case_index, (request, _output) in enumerate(cases):
            if request["identity"] != first["identity"] or request["role"] != "recompute":
                raise ValueError("A native actor batch cannot mix model identities or roles")
            if rank == 0:
                atomic_json(
                    Path(staging).parent / "active-case.json",
                    {
                        "case_id": request["case_id"],
                        "phase": "native_recompute",
                        "request": str(Path(_output).with_suffix(".request.json")),
                    },
                )
            consumed = torch.tensor(
                request["consumed_input_tokens"], device="cuda", dtype=torch.long
            )
            batch, length = consumed.shape
            if length != request["state_position"] + 1:
                raise ValueError("Recompute recurrent-state position is inconsistent")
            cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * length
            packed = PackedSeqParams(
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=length,
                max_seqlen_kv=length,
                qkv_format="thd",
            )
            traces, handles = {}, []
            local_logits = []

            def capture_output(index):
                def hook(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    traces.setdefault(index, {})["output"] = (
                        response_rows(hidden, request).detach().cpu()
                    )

                return hook

            def capture_state(index):
                def hook(state, indices, _cu):
                    slots = indices[:batch].long()
                    if (index + 1) % 4:
                        values = {
                            "convolution": state.convolution[slots],
                            "recurrent_state": state.recurrent[slots],
                        }
                    else:
                        chosen = slots.cpu().tolist()
                        values = {
                            "key": torch.cat([state.keys[i] for i in chosen]),
                            "value": torch.cat([state.values[i] for i in chosen]),
                        }
                    traces.setdefault(index, {})["states"] = {
                        key: value.detach().cpu() for key, value in values.items()
                    }

                return hook

            def capture_logits(_module, _inputs, output):
                if local_logits:
                    raise ValueError(
                        "Native L2 recompute must run exactly one full forward per case"
                    )
                local_logits.append(response_rows(output[0], request).detach())

            def capture_subnode(index, name, value):
                nodes = traces.setdefault(index, {}).setdefault("subnodes", {})
                if name in nodes:
                    raise ValueError(f"Sub-node {name} of layer {index} was observed twice")
                nodes[name] = response_rows(value, request).detach().cpu()

            subnode.install(raw.decoder.layers, subnode.vime_modules, capture_subnode)
            embedding = raw.embedding.word_embeddings

            def capture_embedding(_name, value):
                if "embedding" in traces.setdefault("model", {}):
                    raise ValueError("The embedding was observed twice in one forward")
                traces["model"]["embedding"] = response_rows(value, request).detach().cpu()

            embedding.subnode_hook = capture_embedding
            for index, layer in enumerate(raw.decoder.layers):
                layer.self_attention.trace_state_hook = capture_state(index)
                handles.append(layer.register_forward_hook(capture_output(index)))
            handles.append(raw.output_layer.register_forward_hook(capture_logits))
            try:
                with torch.no_grad():
                    model[0](
                        input_ids=consumed.reshape(1, -1),
                        position_ids=None,
                        attention_mask=None,
                        labels=None,
                        packed_seq_params=packed,
                    )
                if len(local_logits) != 1 or len(traces) != 49:
                    raise ValueError("Native model omitted an output or diagnostic hook")
                local = local_logits[0].reshape(-1, 37984)
                targets = torch.tensor(
                    request["response_tokens"], device="cuda", dtype=torch.long
                ).reshape(-1)
                selected, _lse, _entropy = shared_logprob(
                    local,
                    targets,
                    contract=qwen_logprob_contract(local, rank=rank, world_size=4),
                    tp_group=group,
                    with_entropy=True,
                )
                peers = [torch.empty_like(local) for _ in range(4)]
                dist.all_gather(peers, local.contiguous(), group=group)
                record = {
                    "rank": rank,
                    "request_sha256": request["request_sha256"],
                    "role": "recompute",
                    "execution": request["execution"],
                    "provider_binary": provider_binary,
                    "layers": traces,
                    "local_logits": local.reshape(batch, 16, 37984).cpu(),
                    "responses": targets.reshape(batch, 16).cpu(),
                    "logprobs": selected.reshape(batch, 16).cpu(),
                }
                if rank == 0:
                    record["logits"] = torch.cat(peers, dim=-1).reshape(batch, 16, 151936).cpu()
                save_tensor_artifact(Path(staging) / f"case-{case_index}-rank-{rank}.pt", record)
                del peers, local
                local_logits.clear()
            finally:
                for handle in handles:
                    handle.remove()
                for layer in raw.decoder.layers:
                    layer.self_attention.trace_state_hook = None
                subnode.remove(raw.decoder.layers, subnode.vime_modules)
                embedding.subnode_hook = None
            dist.barrier()
    except BaseException:
        # One rank must not take the others down by deadlock: tearing down the
        # process group while its peers sit in a collective can block forever.
        # Report, then exit so the spawning parent terminates
        # the sibling ranks and the stage fails instead of running out its budget.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(3)
    finally:
        dist.destroy_process_group()


def _assemble(request, ranks, records):
    if len(ranks) != 4 or [item.get("rank") for item in ranks] != list(range(4)):
        raise ValueError("Recompute artifacts must cover all four distinct TP ranks")
    result = {
        "metadata": {
            key: request[key]
            for key in ("schema_version", "case_id", "role", "identity", "request_sha256")
        },
        "execution": request["execution"],
        "prompt_tokens": torch.tensor(request["prompt_tokens"], dtype=torch.int64),
        "response_tokens": torch.tensor(request["response_tokens"], dtype=torch.int64),
        "prediction_positions": torch.tensor(request["prediction_positions"], dtype=torch.int64),
        "logits": ranks[0]["logits"],
        "logprobs": ranks[0]["logprobs"],
        "layer_digests": [],
        "embedding_digest": embedding_digest(ranks[0]["layers"]["model"]["embedding"]),
        "debug_tensors": {},
        "worker_traces": records,
        "provider_binary": ranks[0]["provider_binary"],
    }
    for peer in ranks[1:]:
        exact(
            peer["layers"]["model"]["embedding"],
            ranks[0]["layers"]["model"]["embedding"],
            name="TP replicated embedding",
        )
    for peer in ranks:
        if peer.get("provider_binary") != result["provider_binary"]:
            raise ValueError("Recompute TP ranks loaded different CUDA provider binaries")
        if peer["request_sha256"] != request["request_sha256"]:
            raise ValueError("Rank result belongs to a different request")
        exact(peer["logprobs"], result["logprobs"], name="TP selected logprob replica")
    for layer in range(48):
        hidden = ranks[0]["layers"][layer]["output"]
        states = {}
        for rank, peer in enumerate(ranks):
            exact(peer["layers"][layer]["output"], hidden, name=f"TP layer {layer} replica")
            states.update(
                {
                    f"tp{rank}.{name}": value
                    for name, value in peer["layers"][layer]["states"].items()
                }
            )
        nodes = ranks[0]["layers"][layer]["subnodes"]
        for peer in ranks[1:]:
            for name in subnode.SUBNODES:
                exact(peer["layers"][layer]["subnodes"][name], nodes[name], name=f"TP {name}")
        nodes = {name: nodes[name] for name in subnode.SUBNODES}
        result["layer_digests"].append(
            layer_digest(layer, hidden, states, request=request, subnodes=nodes)
        )
        if layer in request["trace_layers"]:
            result["debug_tensors"][f"layer.{layer}.output"] = hidden
            result["debug_tensors"].update(
                {f"layer.{layer}.subnode.{name}": value for name, value in nodes.items()}
            )
            result["debug_tensors"].update(
                {f"layer.{layer}.{name}": value for name, value in states.items()}
            )
    validate_model_artifact(result, request)
    return result


def run_recompute_cases(cases):
    cases = [(request, str(output)) for request, output in cases]
    if not cases:
        raise ValueError("Native recompute batch cannot be empty")
    staging = Path(cases[0][1]).parent / f"native-recompute-{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    # Spawn preserves the current isolated-venv interpreter. Each rank loads
    # the real model once, and all ranks exit before another GPU role starts.
    mp.spawn(_rank_main, args=(cases, str(staging)), nprocs=4, join=True)
    for index, (request, output) in enumerate(cases):
        files = [staging / f"case-{index}-rank-{rank}.pt" for rank in range(4)]
        records = []
        for rank, path in enumerate(files):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            records.append(
                {
                    "rank": rank,
                    "path": str(path.absolute()),
                    "sha256": digest,
                    "request_sha256": request["request_sha256"],
                }
            )
        artifact = _assemble(request, [load_tensor_artifact(path) for path in files], records)
        validate_native_worker_traces(artifact, request, artifact_root=staging.parent)
        # Retain original per-rank hidden/state tensors for historical validation
        # and first-divergence diagnosis after old HF exports have been retired.
        save_tensor_artifact(Path(output), artifact)


def run_recompute_case(request, output):
    run_recompute_cases([(request, output)])
