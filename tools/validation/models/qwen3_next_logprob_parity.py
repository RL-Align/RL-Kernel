#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Digest every cross-rank point of the Qwen3-Next TP4 path on real weights.

    torchrun --nproc-per-node 4 tools/validation/models/qwen3_next_logprob_parity.py \\
        --checkpoint <Qwen3-Next-80B-A3B-Instruct> --output <dir>

Run it once per NCCL setting (``NCCL_ALGO``, ``NCCL_PROTO``, ``NCCL_MAX_NCHANNELS``)
and compare the reports with ``qwen3_next_logprob_parity_compare.py``. Each rank
writes ``rank-<r>.json`` with SHA-256 digests of the raw bytes at every point
where ranks combine values:

* the vocabulary-parallel embedding and its weight gradient;
* the row-parallel outputs and input gradients of GDN layer 0, full-attention
  layer 3 and the MoE block of layer 0;
* the TP4 selected-token logprob, LSE and entropy (and the logits gradient),
  the gathered sampling logits, and the same logprob computed on one rank over
  the gathered full-vocabulary logits (TP1), which must be bitwise equal;
* a negative control: the attention ``o_proj`` row-parallel partials reduced by
  RL-Kernel's deterministic collective and, separately, by ``torch.distributed``
  ``all_reduce`` on NCCL.
"""

import hashlib
import os
import sys
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from rl_engine.contracts.operators.logprob import (  # noqa: E402
    LogprobContract,
    LogprobDType,
    LogprobRole,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.integrations.engines.train.vllm.qwen3_next_provider import (  # noqa: E402
    GDNProviderConfig,
)
from rl_engine.models.qwen3_next.qwen3_next_embedding import shared_embedding  # noqa: E402
from rl_engine.models.qwen3_next.qwen3_next_forward import (  # noqa: E402
    shared_linear,
    shared_logprob,
)
from rl_engine.models.qwen3_next.qwen3_next_logprob import qwen_logprob_contract  # noqa: E402
from rl_engine.models.qwen3_next.qwen3_next_tp import _reduce  # noqa: E402
from rl_engine.models.qwen3_next.qwen3_next_tp_blocks import (  # noqa: E402
    ATTENTION_HF_SHAPES,
    TP4FullAttention,
    TP4MoE,
    moe_hf_shapes,
)
from rl_engine.models.qwen3_next.qwen3_next_tp_gdn import TP4GDN  # noqa: E402
from tools.validation.models.qwen3_next_gate_common import CheckpointWeights, run_gate  # noqa: E402
from tools.validation.models.qwen3_next_tp_gdn_check import load_first_gdn  # noqa: E402

TOKENS = 1024
VOCAB, SHARD = 151936, 37984
NCCL_SETTINGS = ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MAX_NCHANNELS", "NCCL_MIN_NCHANNELS")


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def _hidden(generator):
    return torch.randn(TOKENS, 2048, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.1


def _full_vocab_contract(n, dtype):
    return LogprobContract(
        role=LogprobRole.TRAIN,
        dtype=dtype,
        mask=MaskSpec(num_tokens=n, active_mask=(True,) * n),
        sharding=ShardingSpec(
            tp_rank=0,
            tp_world_size=1,
            vocab_shard_bounds=((0, VOCAB),),
            real_vocab_size=VOCAB,
            padded_vocab_size=VOCAB,
            cp_rank=0,
            cp_world_size=1,
        ),
        reduction=ReductionSpec(),
    )


def _block(name, module, run, generator):
    """Forward output and input-gradient digests of one TP4 block."""
    hidden = _hidden(generator).requires_grad_(True)
    out = run(module, hidden)
    dy = torch.randn(out.shape, device="cuda", dtype=out.dtype, generator=generator)
    (grad,) = torch.autograd.grad(out, hidden, dy)
    return {f"{name}.forward": digest(out), f"{name}.input_grad": digest(grad)}


def parity(checkpoint, generator):
    rank = dist.get_rank()
    group = dist.group.WORLD
    points = {}
    with ExitStack() as stack:
        top = CheckpointWeights(
            checkpoint, "", ("model.embed_tokens.weight", "lm_head.weight"), stack
        )
        embedding = top["model.embed_tokens.weight"][rank * SHARD : (rank + 1) * SHARD]
        embedding = embedding.cuda().requires_grad_(True)
        lm_head_full = top["lm_head.weight"].cuda()

        tokens = torch.randint(0, VOCAB, (TOKENS,), device="cuda", generator=generator)
        embedded = shared_embedding(tokens, embedding, group)
        dy = torch.randn(embedded.shape, device="cuda", dtype=embedded.dtype, generator=generator)
        (weight_grad,) = torch.autograd.grad(embedded, embedding, dy)
        points["embedding.forward"] = digest(embedded)
        points["embedding.weight_grad"] = digest(weight_grad)
        del embedding, weight_grad

        gdn = TP4GDN(group=group, device="cuda", config=GDNProviderConfig())
        gdn.load_hf_weights(load_first_gdn(checkpoint))
        indices = torch.tensor([1], device="cuda", dtype=torch.int32)
        bounds = torch.tensor([0, TOKENS])
        points.update(
            _block(
                "gdn_layer0",
                gdn,
                lambda m, h: m(h, m.initial_state(2), indices, bounds)[0],
                generator,
            )
        )
        del gdn

        attention = TP4FullAttention(group=group, device="cuda")
        attention.load_hf_weights(
            CheckpointWeights(checkpoint, "model.layers.3.self_attn.", ATTENTION_HF_SHAPES, stack)
        )
        points.update(
            _block(
                "attention_layer3",
                attention,
                lambda m, h: m(h, m.initial_state(2), indices, bounds)[0],
                generator,
            )
        )
        # Negative control on a real row-parallel projection: the same local
        # partials reduced by the deterministic collective and by NCCL all_reduce.
        with torch.no_grad():
            local_in = _hidden(generator)[:, :1024].contiguous()
            partial = shared_linear(local_in, attention.o_proj.weight)
            points["o_proj_partials.deterministic"] = digest(_reduce(partial, group))
            nccl = partial.clone()
            dist.all_reduce(nccl, group=group)
            points["o_proj_partials.nccl_all_reduce"] = digest(nccl)
        del attention

        moe = TP4MoE(group=group, device="cuda")
        moe.load_hf_weights(
            CheckpointWeights(checkpoint, "model.layers.0.mlp.", moe_hf_shapes(), stack)
        )
        points.update(_block("moe_layer0", moe, lambda m, h: m(h), generator))
        del moe

        hidden = _hidden(generator)
        targets = torch.randint(0, VOCAB, (TOKENS,), device="cuda", generator=generator)
        lm_head = lm_head_full[rank * SHARD : (rank + 1) * SHARD].contiguous()
        local = shared_linear(hidden, lm_head).requires_grad_(True)
        contract = qwen_logprob_contract(local, rank=rank, world_size=4)
        selected, lse, entropy = shared_logprob(local, targets, contract=contract, tp_group=group)
        dsel = torch.randn(selected.shape, device="cuda", dtype=selected.dtype, generator=generator)
        (logit_grad,) = torch.autograd.grad(selected, local, dsel)
        points["logprob.tp4.selected"] = digest(selected)
        points["logprob.tp4.lse"] = digest(lse)
        points["logprob.tp4.entropy"] = digest(entropy)
        points["logprob.tp4.logit_grad"] = digest(logit_grad)

        with torch.no_grad():
            peers = [torch.empty_like(local) for _ in range(4)]
            dist.all_gather(peers, local.detach().contiguous())
            gathered = torch.cat(peers, dim=1)
            points["sampling_logits.gathered"] = digest(gathered)
            # TP1 sees the same logits as the four shards: only the cross-rank
            # merge differs, which is what the vocabulary-parallel contract fixes.
            dtype = {torch.bfloat16: LogprobDType.BF16, torch.float32: LogprobDType.FP32}[
                gathered.dtype
            ]
            tp1 = shared_logprob(gathered, targets, contract=_full_vocab_contract(TOKENS, dtype))
            full = shared_linear(hidden, lm_head_full)
        points["logprob.tp1.selected"] = digest(tp1[0])
        points["logprob.tp1.lse"] = digest(tp1[1])
        points["logprob.tp1.entropy"] = digest(tp1[2])
        points["full_logits.matches_gathered"] = bool(torch.equal(full, gathered))

    for peer_points in _gather_objects(points):
        for key in points:
            if key.startswith(("embedding.forward", "logprob.tp4.selected", "sampling_logits")):
                if peer_points[key] != points[key]:
                    raise AssertionError(f"{key} differs between ranks")
    for part in ("selected", "lse", "entropy"):
        if points[f"logprob.tp4.{part}"] != points[f"logprob.tp1.{part}"]:
            raise AssertionError(f"TP4 {part} differs from TP1")
    return {
        "tokens": TOKENS,
        "nccl_settings": {name: os.environ.get(name) for name in NCCL_SETTINGS},
        "points": points,
    }


def _gather_objects(value):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


if __name__ == "__main__":
    run_gate(__doc__, "tp4_distributed_logprob_parity", parity)
