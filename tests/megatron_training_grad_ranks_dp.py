# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P3 (WS2) on real GPUs: cross-rank gradient evidence on Megatron DDP.

One process per GPU runs every configuration in turn (one interpreter start):
a tiny Megatron GPT scored through VIME's LM-head capture and the real strict
``linear_logp`` route, real Megatron DDP, two micro-batches with different data
on each DP rank, then ``finish_grad_sync`` and one optimizer step. Each rank
records the step with ``GradientStepObserver(observe_sync=True)`` and, under
the distributed optimizer, its owned ranges; rank 0 validates the gathered set.

With ``overlap_grad_reduce`` the reduction starts during the last backward, so
the pre-sync snapshot and the numeric check below are skipped for that config.

Test-side checks that are not evidence: the observed step's synchronised
gradients are bitwise equal to an unobserved step's (read-only), the
synchronised shard matches the sum of every rank's pre-sync bucket within FP32
reordering tolerance, and parameters agree across DP replicas after the step.

    torchrun --standalone --nproc_per_node=4 tests/megatron_training_grad_ranks_dp.py \
        --out runs/<id>/grad_ranks.json
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SEQ, VOCAB_PADDED, VOCAB_REAL, HIDDEN = 32, 512, 500, 128
BUCKET_SIZE = 100_000
CONFIGS = (
    {"name": "tp1_dp4_distopt", "tp": 1, "distributed_optimizer": True},
    {"name": "tp1_dp4_allreduce", "tp": 1, "distributed_optimizer": False},
    {"name": "tp2_dp2_distopt", "tp": 2, "distributed_optimizer": True},
    # Overlapped reduction keeps bucket_size, so the buffer spans several buckets; the sync
    # starts from backward hooks on the last micro-batch (the first runs under no_sync).
    {"name": "tp1_dp4_distopt_overlap", "tp": 1, "distributed_optimizer": True, "overlap": True},
)


def _sha(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().cpu().reshape(-1)
    return hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()


def _build_model(tp: int):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from vime.backends.megatron_utils.model import _install_linear_logp_capture

    config = TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        tensor_model_parallel_size=tp,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    torch.manual_seed(0)
    model = GPTModel(
        config,
        get_gpt_layer_local_spec(),
        vocab_size=VOCAB_PADDED,
        max_sequence_length=SEQ,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
    ).cuda()
    model = Float16Module(config, model)
    _install_linear_logp_capture(
        [model], argparse.Namespace(padded_vocab_size=VOCAB_PADDED, vocab_size=VOCAB_REAL)
    )
    return model, config


def _tokens(shift: int) -> torch.Tensor:
    return ((torch.arange(SEQ, device="cuda") * 7 + 3 + shift) % VOCAB_REAL).reshape(1, SEQ)


def _request(model, tokens):
    """VIME's get_log_probs_and_entropy request for one packed sample (CP=1)."""

    from megatron.core import parallel_state as mpu
    from vime.backends.megatron_utils.linear_logp_provider import LinearLogpRequest, TokenLayout
    from vime.backends.megatron_utils.model import _take_linear_logp_context

    positions = torch.arange(SEQ, device="cuda").reshape(1, SEQ)
    logits = model(tokens, positions, None).squeeze(0).contiguous()
    targets = torch.zeros(SEQ, dtype=torch.long, device="cuda")
    targets[: SEQ - 1] = tokens[0, 1:]
    return LinearLogpRequest(
        logits=logits,
        target_ids=targets,
        tensor_parallel_group=mpu.get_tensor_model_parallel_group(),
        token_layout=TokenLayout(world_size=1, rank=0, layout="single"),
        with_entropy=False,
        with_entropy_grad=False,
        chunk_size=1024,
        log_prob_keep_mask=None,
        context=_take_linear_logp_context(model),
        temperature=1.0,
        metadata={
            "logits_are_temperature_scaled": True,
            "complete_sampling_support": False,
            "real_vocab_size": VOCAB_REAL,
            "padded_vocab_size": VOCAB_PADDED,
            "tp_rank": mpu.get_tensor_model_parallel_rank(),
            "tp_world_size": mpu.get_tensor_model_parallel_world_size(),
        },
    )


def _step(cfg: dict[str, Any], *, observe: bool) -> dict[str, Any]:
    from megatron.core import parallel_state as mpu
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    from rl_engine.integrations.megatron_training_gradients import (
        GradientStepObserver,
        observe_optimizer_ownership,
    )
    from rl_engine.integrations.vime.linear_logp_provider import provider

    distributed = cfg["distributed_optimizer"]
    overlap = bool(cfg.get("overlap", False))
    model, config = _build_model(cfg["tp"])
    for param in model.module.embedding.parameters():
        param.requires_grad_(False)  # frozen base
    ddp = DistributedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            overlap_grad_reduce=overlap,
            use_distributed_optimizer=distributed,
            grad_reduce_in_fp32=True,
            bucket_size=BUCKET_SIZE,
        ),
        module=model,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam", lr=1e-3, bf16=True, use_distributed_optimizer=distributed
        ),
        [ddp],
    )
    observer = GradientStepObserver(ddp, observe_sync=True)
    if observe:
        observer.install()
    ddp.zero_grad_buffer()
    optimizer.zero_grad()
    mask = torch.zeros(SEQ, dtype=torch.bool, device="cuda")
    mask[SEQ // 2 : SEQ - 1] = True
    dp_rank = mpu.get_data_parallel_rank()
    for micro_batch in range(2):
        with ddp.no_sync() if overlap and micro_batch == 0 else contextlib.nullcontext():
            result = provider(_request(ddp.module, _tokens(3 * dp_rank + micro_batch)))
            (-(result.logp.squeeze(-1)[mask]).sum()).backward()
    buckets = [bucket for buffer in ddp.buffers for bucket in buffer.buckets]
    pre_sync = None if overlap else [bucket.grad_data.detach().cpu().clone() for bucket in buckets]
    ddp.finish_grad_sync()
    out: dict[str, Any] = {}
    if observe:
        out["evidence"] = observer.end_step()
        observer.uninstall()
        out["ownership"] = observe_optimizer_ownership(optimizer, ddp) if distributed else None
    size, rank = mpu.get_data_parallel_world_size(), mpu.get_data_parallel_rank()
    synced = []
    for bucket in buckets:
        if distributed:
            shard = bucket.grad_data.numel() // size
            synced.append(
                bucket.grad_data[rank * shard : (rank + 1) * shard].detach().cpu().clone()
            )
        else:
            synced.append(bucket.grad_data.detach().cpu().clone())
    out["pre_sync"] = pre_sync
    out["synced"] = synced
    out["scaling"] = [float(bucket.gradient_scaling_factor) for bucket in buckets]
    out["synced_sha"] = [_sha(item) for item in synced]
    update_ok, _, _ = optimizer.step()
    out["update_ok"] = bool(update_ok)
    out["params_sha"] = _sha(
        torch.cat([p.detach().float().reshape(-1) for p in ddp.module.parameters()])
    )
    return out


def _numeric_check(results, gloo, cfg) -> dict[str, Any]:
    """Synced shard vs the sum of every DP rank's scaled pre-sync bucket (not evidence)."""

    from megatron.core import parallel_state as mpu

    if results["pre_sync"] is None:
        return {"max_abs_diff": None, "ok": True, "skipped": "overlap_grad_reduce"}
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, results["pre_sync"], group=gloo)
    dp_ranks = dist.get_process_group_ranks(mpu.get_data_parallel_group())
    size, rank = len(dp_ranks), mpu.get_data_parallel_rank()
    max_diff = 0.0
    for index, local in enumerate(results["synced"]):
        scale = results["scaling"][index]
        total = sum(gathered[peer][index] * scale for peer in dp_ranks)
        if cfg["distributed_optimizer"]:
            shard = total.numel() // size
            total = total[rank * shard : (rank + 1) * shard]
        max_diff = max(max_diff, float((total - local).abs().max()))
    return {"max_abs_diff": max_diff, "ok": max_diff <= 1e-5}


def _negatives(readbacks, topology) -> dict[str, str]:
    from rl_engine.alignment.cross_config.adapters.megatron_training import T05ContractError
    from rl_engine.alignment.cross_config.adapters.megatron_training_gradient_ranks import (
        GradientRankReadback,
        validate_gradient_rank_set,
    )

    payload = [item.to_dict() for item in readbacks]
    cases = {}

    def drop(items):
        items.pop()

    def duplicate(items):
        items.append(copy.deepcopy(items[0]))

    def wrong_collective(items):
        for call in items[-1]["evidence"]["sync"]["calls"]:
            call["kind"] = "all_reduce" if call["kind"] == "reduce_scatter" else "reduce_scatter"

    cases["drop_last_rank"] = (drop, "MISSING_RANK")
    cases["duplicate_rank"] = (duplicate, "IDENTITY_DRIFT")
    cases["wrong_collective"] = (wrong_collective, "SILENT_FALLBACK")
    if payload[0]["ownership"] is not None:

        def drop_owned(items):
            owner = next(item for item in items if item["ownership"]["ranges"])
            owner["ownership"]["ranges"].pop()

        cases["drop_owned_range"] = (drop_owned, "INCOMPLETE_ARTIFACT")
    result = {}
    for name, (mutate, expected) in cases.items():
        items = copy.deepcopy(payload)
        mutate(items)
        try:
            validate_gradient_rank_set(
                [GradientRankReadback.from_dict(item) for item in items], topology
            )
            result[name] = f"PASSED (expected {expected})"
        except T05ContractError as error:
            result[name] = error.status
    result["ok"] = all(result[name] == expected for name, (_, expected) in cases.items())
    return result


def run_config(cfg: dict[str, Any], gloo) -> dict[str, Any]:
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from rl_engine.alignment.cross_config.adapters.megatron_training_gradient_ranks import (
        GradientRankReadback,
        gather_gradient_readbacks,
        validate_gradient_rank_set,
    )
    from rl_engine.alignment.cross_config.adapters.megatron_training_ranks import (
        ExpectedRankTopology,
        observe_rank_coordinates,
    )

    world = dist.get_world_size()
    mpu.initialize_model_parallel(tensor_model_parallel_size=cfg["tp"])
    model_parallel_cuda_manual_seed(1234)
    try:
        plain_a = _step(cfg, observe=False)
        plain_b = _step(cfg, observe=False)
        observed = _step(cfg, observe=True)
        coords = observe_rank_coordinates(
            tp_group=mpu.get_tensor_model_parallel_group(),
            dp_group=mpu.get_data_parallel_group(with_context_parallel=True),
        )
        readbacks = gather_gradient_readbacks(
            GradientRankReadback.from_evidence(observed["evidence"], coords, observed["ownership"]),
            gloo,
        )
        numeric = _numeric_check(observed, gloo, cfg)
        local = {
            "global_rank": dist.get_rank(),
            "tp_rank": mpu.get_tensor_model_parallel_rank(),
            "reproducible": plain_a["synced_sha"] == plain_b["synced_sha"],
            "read_only": plain_a["synced_sha"] == observed["synced_sha"],
            "numeric": numeric,
            "update_ok": observed["update_ok"],
            "params_sha": observed["params_sha"],
        }
        locals_all: list[Any] = [None] * world
        dist.all_gather_object(locals_all, local, group=gloo)
        result: dict[str, Any] = {"config": cfg}
        if dist.get_rank() == 0:
            topology = ExpectedRankTopology(tp=cfg["tp"], dp=world // cfg["tp"])
            report = validate_gradient_rank_set(readbacks, topology)
            result["report"] = report.to_dict()
            result["report_fingerprint"] = report.fingerprint
            result["negatives"] = _negatives(readbacks, topology)
            result["ranks"] = locals_all
            replicas: dict[int, set[str]] = {}
            for item in locals_all:
                replicas.setdefault(item["tp_rank"], set()).add(item["params_sha"])
            result["params_replicated_after_step"] = all(len(v) == 1 for v in replicas.values())
            result["sync_calls_per_rank"] = [
                len(item.evidence["sync"]["calls"]) for item in readbacks
            ]
            result["unmatched_calls_per_rank"] = [
                item.evidence["sync"]["unmatched_calls"] for item in readbacks
            ]
            result["ok"] = bool(
                report.complete
                and result["negatives"]["ok"]
                and result["params_replicated_after_step"]
                and all(item["numeric"]["ok"] and item["update_ok"] for item in locals_all)
                and all(item["read_only"] for item in locals_all if item["reproducible"])
            )
        return result
    finally:
        mpu.destroy_model_parallel()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="grad_ranks.json")
    parser.add_argument("--only", nargs="*", help="config names to run (default: all)")
    args = parser.parse_args()

    os.environ.update(
        RL_KERNEL_ATTENTION_CASE="P/P",
        RL_KERNEL_FFN_CASE="P/P",
        RL_KERNEL_LOGP_CASE="R/R",
        VIME_RL_KERNEL_STRICT="1",
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    gloo = dist.new_group(list(range(dist.get_world_size())), backend="gloo")

    from rl_engine.integrations.ablation import integration_plan_from_environment
    from rl_engine.integrations.megatron_runtime import install_megatron_integration

    install_megatron_integration(integration_plan_from_environment())

    results, status = [], 0
    for cfg in CONFIGS:
        if args.only and cfg["name"] not in args.only:
            continue
        try:
            result = run_config(cfg, gloo)
        except Exception as error:  # record and continue with the next configuration
            result = {"config": cfg, "ok": False, "error": f"{type(error).__name__}: {error}"}
            if dist.get_rank() == 0:
                import traceback

                traceback.print_exc()
        if dist.get_rank() == 0:
            results.append(result)
            print(
                f"{cfg['name']}: ok={result.get('ok')} "
                f"collective={result.get('report', {}).get('collective')} "
                f"negatives={result.get('negatives')} error={result.get('error')}",
                flush=True,
            )
            status |= 0 if result.get("ok") else 1
        dist.barrier(group=gloo)
    if dist.get_rank() == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1, sort_keys=True, default=str))
    flag = torch.tensor([status], dtype=torch.int64)
    dist.broadcast(flag, src=0, group=gloo)
    dist.destroy_process_group()
    return int(flag.item())


if __name__ == "__main__":
    raise SystemExit(main())
