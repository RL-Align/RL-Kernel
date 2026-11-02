# SPDX-License-Identifier: Apache-2.0
"""Frozen identity and validation for the task-28 0731 gate replay fixture."""

import hashlib

import torch

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
TENSOR = "layers.0.ffn.gate.weight"
SHARD = "model-00002-of-00048.safetensors"
SHAPE = (256, 4096)
WEIGHT_BYTES = 256 * 4096 * 2
WEIGHT_SHA256 = "e110e3c4e310e7c9981f287038c5a0c8ad73da1b36b36feb3b909e04477a6a12"


def validate_tensor_metadata(tensor: dict) -> tuple[int, int]:
    """Validate the byte span before it can control an HTTP request."""
    if not isinstance(tensor, dict) or (
        tensor.get("dtype") != "BF16" or tensor.get("shape") != list(SHAPE)
    ):
        raise ValueError("0731 gate requires BF16 tensor metadata with shape [256, 4096]")
    offsets = tensor.get("data_offsets")
    if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
        raise ValueError("gate data_offsets must contain two nonnegative integers")
    if any(type(value) is not int or value < 0 for value in offsets):
        raise ValueError("gate data_offsets must contain two nonnegative integers")
    start, end = offsets
    if end - start != WEIGHT_BYTES:
        raise ValueError(f"gate byte span must be exactly {WEIGHT_BYTES} bytes")
    return start, end


def validate_weight_bytes(data: bytes) -> str:
    if len(data) != WEIGHT_BYTES:
        raise ValueError(f"gate payload must be exactly {WEIGHT_BYTES} bytes")
    digest = hashlib.sha256(data).hexdigest()
    if digest != WEIGHT_SHA256:
        raise ValueError("gate payload SHA256 does not match the frozen 0731 tensor")
    return digest


def checksum(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(data).hexdigest()


def validate_gate_fixture(fixture: dict) -> torch.Tensor:
    """Reject a different checkpoint or self-consistent replacement before GPU use."""
    if not isinstance(fixture, dict) or not isinstance(fixture.get("metadata"), dict):
        raise ValueError("gate fixture requires weight and frozen checkpoint metadata")
    metadata = fixture["metadata"]
    for key, expected in (
        ("model", MODEL),
        ("revision", REVISION),
        ("tensor", TENSOR),
        ("shard", SHARD),
        ("weight_sha256", WEIGHT_SHA256),
    ):
        if metadata.get(key) != expected:
            raise ValueError(f"gate fixture {key} does not match the frozen 0731 tensor")
    validate_tensor_metadata(metadata.get("tensor_metadata"))
    weight = fixture.get("weight")
    if (
        not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
        or (tuple(weight.shape) != SHAPE)
    ):
        raise ValueError("gate fixture weight must be BF16 [256, 4096]")
    if checksum(weight) != WEIGHT_SHA256:
        raise ValueError("gate fixture weight SHA256 does not match the frozen 0731 tensor")
    return weight
