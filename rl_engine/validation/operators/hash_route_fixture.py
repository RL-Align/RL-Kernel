# SPDX-License-Identifier: Apache-2.0
"""Independently verify the fixed 0731 table identity before GPU selection."""

import torch

from rl_engine.contracts.operators.hash_route import (
    CHECKPOINT_TABLE_SHA256,
    HASH_LAYERS,
    MODEL,
    REVISION,
    SLOTS,
    TABLE_BYTES,
    VOCABULARY,
)
from rl_engine.ops.moe.hash_route import tensor_sha256


def load_checkpoint_table(path, absolute_layer):
    if type(absolute_layer) is not int or absolute_layer not in HASH_LAYERS:
        raise ValueError("checkpoint Hash layer must be an absolute layer 0, 1 or 2")
    fixture = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(fixture, dict) or not isinstance(fixture.get("metadata"), dict):
        raise ValueError("checkpoint Hash table requires frozen metadata")
    metadata = fixture.get("metadata", {})
    for key, expected in (
        ("model", MODEL),
        ("revision", REVISION),
        ("absolute_layer", absolute_layer),
        ("tensor", f"layers.{absolute_layer}.ffn.gate.tid2eid"),
        ("shard", f"model-{absolute_layer + 2:05d}-of-00048.safetensors"),
        ("sha256", CHECKPOINT_TABLE_SHA256[absolute_layer]),
    ):
        if metadata.get(key) != expected:
            raise ValueError(f"checkpoint Hash table {key} identity mismatch")
    table = fixture.get("table")
    if (
        not isinstance(table, torch.Tensor)
        or table.dtype != torch.int64
        or (tuple(table.shape) != (VOCABULARY, SLOTS))
    ):
        raise ValueError("checkpoint Hash table must be I64 [129280,6]")
    tensor = metadata.get("tensor_metadata", {})
    if not isinstance(tensor, dict):
        raise ValueError("checkpoint Hash table tensor metadata mismatch")
    if tensor.get("dtype") != "I64" or tensor.get("shape") != [VOCABULARY, SLOTS]:
        raise ValueError("checkpoint Hash table tensor metadata mismatch")
    offsets = tensor.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(value) is not int or value < 0 for value in offsets)
        or offsets[1] - offsets[0] != TABLE_BYTES
    ):
        raise ValueError("checkpoint Hash table byte span mismatch")
    if tensor_sha256(table) != CHECKPOINT_TABLE_SHA256[absolute_layer]:
        raise ValueError("checkpoint Hash table actual bytes differ from frozen SHA256")
    return table, metadata
