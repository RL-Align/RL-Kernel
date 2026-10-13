#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Real-weight TP4 embedding, final norm, LM-head and logprob component gate."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from qwen3_next_contract import atomic_json
from safetensors import safe_open

from rl_engine.backends.cuda.norm.rmsnorm import Qwen3NextRMSNormCudaOp
from rl_engine.models.qwen3_next.qwen3_next_embedding import shared_embedding
from rl_engine.models.qwen3_next.qwen3_next_forward import shared_linear, shared_logprob
from rl_engine.models.qwen3_next.qwen3_next_logprob import qwen_logprob_contract
from rl_engine.models.qwen3_next.qwen3_next_tp import _CopyToTP
from rl_engine.validation.common.tensor_identity import assert_tensor_bitwise_equal as exact

REQUIRED_CASES = [
    "hf-embedding-roundtrip",
    "hf-lm-head-roundtrip",
    "replicated-norm-gradient",
    "nonzero-finite-embedding-head-gradient",
    "embedding-repeat-100",
    *[f"batch-invariant-prompt{length}" for length in (8, 64, 256, 1024)],
    "logprob-original-and-promoted-logits",
    "entropy-interface",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.device_count() != 4:
        raise RuntimeError("Head acceptance requires four ranks and four visible GPUs")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(minutes=10), device_id=torch.device("cuda", rank)
    )
    group = dist.group.WORLD
    try:
        index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

        def load(name):
            with safe_open(args.checkpoint / index[name], framework="pt", device="cpu") as reader:
                value = reader.get_tensor(name)
            if value.dtype != torch.bfloat16:
                raise ValueError("Official component weight must be BF16")
            return value

        weights = {}
        for name in ("model.embed_tokens.weight", "lm_head.weight"):
            original = load(name)
            local = torch.nn.Parameter(original.chunk(4)[rank].cuda().contiguous())
            peers = [torch.empty_like(local) for _ in range(4)]
            dist.all_gather(peers, local.detach())
            exact(torch.cat(peers).cpu(), original, name=f"loaded {name} HF roundtrip")
            weights[name] = local
            del original, peers
        embedding, head = weights["model.embed_tokens.weight"], weights["lm_head.weight"]
        norm = torch.nn.Parameter(load("model.norm.weight").cuda())
        norm_op = Qwen3NextRMSNormCudaOp()

        def forward(tokens):
            hidden = shared_embedding(tokens, embedding, group)
            normalized = norm_op(hidden, norm, eps=1e-6)
            return shared_linear(_CopyToTP.apply(normalized, group), head)

        with torch.no_grad():
            for length in (8, 64, 256, 1024):
                tokens = torch.arange(4 * (length + 16), device="cuda", dtype=torch.int64)
                tokens = ((tokens * 2399 + 1) % 151936).reshape(4, length + 16)
                batch = forward(tokens)
                singleton = forward(tokens[:1])
                exact(batch[:1], singleton, name=f"embedding/norm/LM batch {length}")
            repeated = torch.tensor([1, 37985, 75969, 113953, 1, 75969], device="cuda")
            expected = shared_embedding(repeated, embedding, group)
            for _ in range(100):
                exact(shared_embedding(repeated, embedding, group), expected)

        tokens = torch.tensor([1, 37985, 75969, 113953] * 4, device="cuda")
        logits = forward(tokens)
        contract = qwen_logprob_contract(logits, rank=rank, world_size=4)
        logp, lse, entropy = shared_logprob(logits, tokens, contract=contract, tp_group=group)
        promoted = logits.float()
        promoted_logp, _, _ = shared_logprob(
            promoted,
            tokens,
            contract=qwen_logprob_contract(promoted, rank=rank, world_size=4),
            tp_group=group,
        )
        exact(logp, promoted_logp, name="native VIME loss input promotion")
        without_entropy, _, no_entropy = shared_logprob(
            logits, tokens, contract=contract, tp_group=group, with_entropy=False
        )
        exact(logp, without_entropy, name="entropy request logprob identity")
        if no_entropy is not None or not all(torch.isfinite(t).all() for t in (entropy, lse)):
            raise AssertionError("Entropy/LSE interface is invalid")
        (-logp.mean()).backward()
        for name, parameter in (("embedding", embedding), ("lm_head", head), ("norm", norm)):
            gradient = parameter.grad
            valid = gradient is not None
            valid = valid and bool(torch.isfinite(gradient).all() and gradient.count_nonzero())
            if not valid:
                raise AssertionError(f"Missing finite nonzero {name} gradient")
        peers = [torch.empty_like(norm.grad) for _ in range(4)]
        dist.all_gather(peers, norm.grad)
        for peer in peers:
            exact(peer, norm.grad, name="replicated final norm gradient")
        from rl_engine.integrations.common.qwen3_next_binary import imported_provider_identity

        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output / f"rank-{rank}.json",
            {
                "status": "passed",
                "scope": "tp4_embedding_lm_head_only",
                "rank": rank,
                "provider_binary": imported_provider_identity(),
                "full_model_l2": "not_evaluated",
                "cases": sorted(REQUIRED_CASES),
            },
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
