# SPDX-License-Identifier: Apache-2.0
"""Isolated-process replay of items 11/15; prints byte digests only."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rl_engine.contracts.operators.indexer_q_preprocess import IndexerQPreprocessContract
from rl_engine.contracts.operators.partial_rope import PartialRoPEContract
from rl_engine.models.deepseek_v4.indexer_q_preprocess import IndexerQPreprocess
from rl_engine.models.deepseek_v4.partial_rope import PartialRoPE
from tests.models.deepseek_v4.rope_mxfp4_helpers import angles


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def replay(output):
    g = torch.Generator().manual_seed(151100)
    cos, sin = angles(4096, device="cuda")
    positions = torch.randint(0, 4096, (37,), generator=g).cuda()
    positions[5] = -1
    result = {}
    for dim, heads in ((128, 64), (512, 16)):
        op = PartialRoPE(PartialRoPEContract(head_dim=dim, rope_dim=64))
        x = torch.randn(37, heads, dim, generator=g).bfloat16().cuda().requires_grad_()
        y = op(x, positions, cos, sin)
        y.backward(torch.randn(y.shape, generator=g).bfloat16().cuda())
        result[f"rope{dim}"] = {"y": digest(y), "dx": digest(x.grad)}
        result[f"rope{dim}"]["implementation"] = op.implementation_fingerprint
    op = IndexerQPreprocess(IndexerQPreprocessContract(num_heads=64))
    q = torch.randn(37, 64, 128, generator=g).bfloat16().cuda().requires_grad_()
    out = op(q, positions, cos, sin)
    out.values.backward(torch.randn(out.values.shape, generator=g).bfloat16().cuda())
    result["indexer_q"] = {
        "values": digest(out.values),
        "packed": digest(out.packed),
        "scales": digest(out.scales),
        "dq": digest(q.grad),
        "implementation": op.implementation_fingerprint,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    replay(parser.parse_args().output)


if __name__ == "__main__":
    main()
