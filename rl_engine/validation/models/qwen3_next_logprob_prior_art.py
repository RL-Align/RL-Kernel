# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Prior art for the Qwen3-Next vocabulary-parallel selected-token logprob.

Every candidate receives the same logits: the full-vocabulary product of the
official LM head with seeded hidden states, computed identically on every rank
and sliced into 37984-column shards for TP4. Only the cross-rank reduction
therefore differs between candidates and between TP degrees.

Candidates:

* ``rl_kernel_ws2`` -- RL-Kernel's provider (per-tile max/sum-exp, all-gather
  transport, merge in global tile order); what both engines run here.
* ``vllm_gather`` -- vLLM's sampler path: all-gather the logits, then
  ``log_softmax(dtype=float32)`` over the full vocabulary.
* ``megatron_ce`` -- Megatron-core ``vocab_parallel_cross_entropy`` (NCCL
  all-reduce of max, then of target logit and sum-exp); logprob = -loss.
* ``vime_default`` -- VIME's own ``calculate_log_probs_and_entropy``.

For each candidate a launch records a digest of the selected logprobs, whether
the first rows computed alone equal the same rows inside the full batch, the
maximum error against an FP64 log-softmax and a median latency.
"""

from __future__ import annotations

import hashlib
import statistics
from typing import Any, Callable

import torch
import torch.distributed as dist

VOCAB, SHARD = 151936, 37984
ALONE_ROWS = 16


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def _rl_kernel(tp: int, group):
    from rl_engine.contracts.operators.logprob import (
        LogprobContract,
        LogprobDType,
        LogprobRole,
        MaskSpec,
        ReductionSpec,
        ShardingSpec,
    )
    from rl_engine.models.qwen3_next.qwen3_next_forward import shared_logprob

    rank = dist.get_rank(group) if tp > 1 else 0

    def run(local, targets):
        n = local.shape[0]
        contract = LogprobContract(
            role=LogprobRole.TRAIN,
            dtype=LogprobDType.BF16,
            mask=MaskSpec(num_tokens=n, active_mask=(True,) * n),
            sharding=ShardingSpec(
                tp_rank=rank,
                tp_world_size=tp,
                vocab_shard_bounds=tuple(
                    (i * (VOCAB // tp), (i + 1) * (VOCAB // tp)) for i in range(tp)
                ),
                real_vocab_size=VOCAB,
                padded_vocab_size=VOCAB,
                cp_rank=0,
                cp_world_size=1,
            ),
            reduction=ReductionSpec(),
        )
        return shared_logprob(
            local,
            targets,
            contract=contract,
            tp_group=group if tp > 1 else None,
            with_entropy=False,
        )[0]

    return run


def _vllm_gather(tp: int, group):
    from vllm.v1.sample.sampler import Sampler

    def run(local, targets):
        full = local
        if tp > 1:
            peers = [torch.empty_like(local) for _ in range(tp)]
            dist.all_gather(peers, local.contiguous(), group=group)
            full = torch.cat(peers, dim=1)
        logprobs = Sampler.compute_logprobs(full)
        return logprobs.gather(1, targets[:, None]).squeeze(1)

    return run


def _megatron(tp: int, group):
    from megatron.core.tensor_parallel import vocab_parallel_cross_entropy

    def run(local, targets):
        loss = vocab_parallel_cross_entropy(local.float()[:, None, :], targets[:, None])
        return -loss.squeeze(1)

    return run


def _vime(tp: int, group):
    from vime.utils.ppo_utils import calculate_log_probs_and_entropy

    def run(local, targets):
        logprob, _ = calculate_log_probs_and_entropy(local.float(), targets, group)
        return logprob.reshape(-1)

    return run


CANDIDATES: dict[str, Callable[[int, Any], Callable]] = {
    "rl_kernel_ws2": _rl_kernel,
    "vllm_gather": _vllm_gather,
    "megatron_ce": _megatron,
    "vime_default": _vime,
}


def _median_us(fn, iters=20):
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def candidate_report(full_logits, targets, *, tp: int, group) -> dict[str, Any]:
    """Run every candidate on this rank's shard of ``full_logits``."""
    rank = dist.get_rank(group) if tp > 1 else 0
    width = VOCAB // tp
    local = full_logits[:, rank * width : (rank + 1) * width].contiguous()
    reference = torch.log_softmax(full_logits.double(), dim=-1).gather(1, targets[:, None])[:, 0]
    out, kept = {}, {}
    for key, factory in CANDIDATES.items():
        try:
            run = factory(tp, group)
            with torch.no_grad():
                values = run(local, targets)
                alone = run(local[:ALONE_ROWS].contiguous(), targets[:ALONE_ROWS])
            values = values.float()
            kept[key] = values
            out[key] = {
                "selected_sha256": digest(values),
                "batch_bitwise": bool(torch.equal(alone.float(), values[:ALONE_ROWS])),
                "max_abs_vs_fp64": float((values.double() - reference).abs().max()),
                "latency_us": _median_us(lambda r=run: r(local, targets)),
            }
        except Exception as exc:  # noqa: BLE001 - recorded per candidate
            out[key] = {"unavailable": f"{type(exc).__name__}: {str(exc)[:300]}"}
    for key, values in kept.items():
        out[key]["max_abs_vs"] = {
            other: float((values - peer).abs().max())
            for other, peer in kept.items()
            if other != key
        }
    return out
