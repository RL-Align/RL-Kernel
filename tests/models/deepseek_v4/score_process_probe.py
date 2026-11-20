# SPDX-License-Identifier: Apache-2.0
"""Isolated-process replay of the Indexer score; prints byte digests only."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rl_engine.contracts.operators.indexer_score import IndexerScoreContract
from rl_engine.models.deepseek_v4.indexer_score import IndexerScore


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def replay(output):
    op = IndexerScore(IndexerScoreContract(num_heads=64, head_dim=128))
    g = torch.Generator().manual_seed(161010)
    icv = (torch.randn(19, 64, 333, generator=g) * 4).cuda()
    weights = torch.randn(19, 64, generator=g).cuda()
    grad = torch.randn(19, 333, generator=g).cuda()
    record = op.forward(icv, weights)
    d_icv, d_weights = op.backward(record, grad)
    result = {
        "scores": digest(record.scores),
        "d_icv": digest(d_icv),
        "d_weights": digest(d_weights),
        "implementation": op.implementation_fingerprint,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    replay(parser.parse_args().output)


if __name__ == "__main__":
    main()
