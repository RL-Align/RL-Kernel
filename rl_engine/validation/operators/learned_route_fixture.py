# SPDX-License-Identifier: Apache-2.0
"""Independent frozen checkpoint-bias identity verification before GPU use."""

import torch

from rl_engine.contracts.operators.learned_route import (
    CHECKPOINT_BIAS_SHA256,
    CHECKPOINT_MODEL,
    CHECKPOINT_REVISION,
)
from rl_engine.ops.moe.learned_route import tensor_sha256


def load_checkpoint_bias(path):
    fixture = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(fixture, dict) or not isinstance(fixture.get("metadata"), dict):
        raise ValueError("checkpoint bias requires frozen metadata")
    metadata = fixture["metadata"]
    for key, expected in [
        ("model", CHECKPOINT_MODEL),
        ("revision", CHECKPOINT_REVISION),
        ("absolute_layer", 3),
        ("tensor", "layers.3.ffn.gate.bias"),
        ("shard", "model-00005-of-00048.safetensors"),
        ("sha256", CHECKPOINT_BIAS_SHA256),
    ]:
        if metadata.get(key) != expected:
            raise ValueError(f"checkpoint bias {key} identity mismatch")
    bias = fixture.get("bias")
    if (
        not isinstance(bias, torch.Tensor)
        or bias.dtype != torch.float32
        or tuple(bias.shape) != (256,)
    ):
        raise ValueError("checkpoint bias must be FP32 [256]")
    tensor = metadata.get("tensor_metadata", {})
    if not isinstance(tensor, dict) or tensor.get("dtype") != "F32" or tensor.get("shape") != [256]:
        raise ValueError("checkpoint bias layout metadata mismatch")
    offsets = tensor.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(v) is not int or v < 0 for v in offsets)
        or offsets[1] - offsets[0] != 1024
    ):
        raise ValueError("checkpoint bias byte span mismatch")
    if tensor_sha256(bias) != CHECKPOINT_BIAS_SHA256:
        raise ValueError("checkpoint bias actual bytes differ from frozen SHA256")
    return bias, metadata
