# SPDX-License-Identifier: Apache-2.0
"""Explicit original-checkpoint loading for the frozen-quantized-weight profile."""

import torch
from safetensors.torch import load_file

from rl_engine.backends.rocm.engram.embedding import ResidentEngramRows
from rl_engine.backends.rocm.gemm.v41_fp8 import V41FP8Linear
from .engram import Engram
from .expert import V41FP4Linear


class Dense(torch.nn.Module):
    """Actual ROCm library GEMM; not advertised as an RLK quantized kernel."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


def projection(checkpoint, name, device, *, tp=None, shard_dim=0):
    weight, scale = checkpoint.projection(name, device=device)
    if tp is not None and tp.size > 1:
        weight = tp.shard(weight, shard_dim)
        if scale is not None:
            scale = tp.shard(scale, shard_dim)
    if weight.dtype == torch.float8_e4m3fn:
        return V41FP8Linear(weight, scale)
    if weight.dtype == torch.uint8:
        return V41FP4Linear(weight, scale)
    return Dense(weight)


def grouped_weight(checkpoint, name, device):
    """Released convert.py's block32 dequantization of wo_a to BF16."""
    w, s = checkpoint.projection(name, device=device)
    if s is None:
        return w.bfloat16()
    n, k = w.shape
    return (
        (w.float().reshape(n // 32, 32, k // 32, 32) * s.float()[:, None, :, None])
        .reshape(n, k)
        .bfloat16()
    )


def engram(checkpoint, config, layer_id, device):
    prefix = f"layers.{layer_id}.engram"
    rows = load_file(
        str(checkpoint.directory / f"engram-{layer_id}-rows.safetensors"), device=str(device)
    )
    table = ResidentEngramRows(
        rows["original_row_ids"], rows[prefix + ".embed.weight"], rows[prefix + ".embed.scale"]
    )
    return Engram(
        config,
        layer_id,
        table,
        projection(checkpoint, prefix + ".wkv", device),
        checkpoint.tensor(prefix + ".q_weight", device=device),
        checkpoint.tensor(prefix + ".k_weight", device=device),
    )
