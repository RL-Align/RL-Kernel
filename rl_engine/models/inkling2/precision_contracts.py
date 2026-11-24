# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini checkpoint and low-bit precision contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import torch


class PrecisionContractError(ValueError):
    """Raised when a checkpoint or precision contract is invalid."""


@dataclass(frozen=True)
class TensorFingerprint:
    name: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class CheckpointFingerprint:
    source: str
    revision: str
    tensors: tuple[TensorFingerprint, ...]

    @property
    def identity_sha256(self) -> str:
        payload = {
            "source": self.source,
            "revision": self.revision,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "revision": self.revision,
            "identity_sha256": self.identity_sha256,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("checkpoint tensors must be torch tensors")
    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)
    # Exact storage bytes for every dtype (floating, integer, bool); reshape(-1) keeps
    # 0-dim tensors viewable as raw bytes.
    return tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


def fingerprint_tensor(name: str, tensor: torch.Tensor) -> TensorFingerprint:
    if not isinstance(name, str) or not name:
        raise PrecisionContractError("tensor name must be nonempty")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("checkpoint entries must be torch tensors")
    digest = hashlib.sha256(_tensor_bytes(tensor)).hexdigest()
    return TensorFingerprint(
        name, tuple(int(dim) for dim in tensor.shape), str(tensor.dtype), digest
    )


def fingerprint_checkpoint(
    state_dict: Mapping[str, torch.Tensor], *, source: str, revision: str
) -> CheckpointFingerprint:
    if not source or not revision:
        raise PrecisionContractError("checkpoint source and revision must be nonempty")
    tensors = tuple(fingerprint_tensor(name, state_dict[name]) for name in sorted(state_dict))
    if not tensors:
        raise PrecisionContractError("checkpoint must contain at least one tensor")
    return CheckpointFingerprint(source, revision, tensors)


def compare_checkpoint_fingerprints(
    expected: CheckpointFingerprint, actual: CheckpointFingerprint
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if expected.source != actual.source:
        mismatches.append(f"source differs: {expected.source!r} != {actual.source!r}")
    if expected.revision != actual.revision:
        mismatches.append(f"revision differs: {expected.revision!r} != {actual.revision!r}")
    expected_tensors = {tensor.name: tensor for tensor in expected.tensors}
    actual_tensors = {tensor.name: tensor for tensor in actual.tensors}
    for name in sorted(expected_tensors.keys() | actual_tensors.keys()):
        if name not in expected_tensors:
            mismatches.append(f"unexpected tensor: {name}")
            continue
        if name not in actual_tensors:
            mismatches.append(f"missing tensor: {name}")
            continue
        if expected_tensors[name] != actual_tensors[name]:
            mismatches.append(f"tensor fingerprint differs: {name}")
    return tuple(mismatches)


def require_checkpoint_match(
    expected: CheckpointFingerprint, actual: CheckpointFingerprint
) -> None:
    mismatches = compare_checkpoint_fingerprints(expected, actual)
    if mismatches:
        raise PrecisionContractError("checkpoint identity mismatch: " + "; ".join(mismatches))


@dataclass(frozen=True)
class QuantBackendCapability:
    platform: str
    backend: str
    format: str
    supports_training: bool
    supports_inference: bool
    available: bool
    reason: str = ""


@dataclass(frozen=True)
class QuantBackendMatrix:
    entries: tuple[QuantBackendCapability, ...]

    def for_platform(self, platform: str) -> tuple[QuantBackendCapability, ...]:
        return tuple(entry for entry in self.entries if entry.platform == platform)

    def require(self, *, platform: str, format: str, purpose: str) -> QuantBackendCapability:
        if purpose not in {"training", "inference"}:
            raise PrecisionContractError("purpose must be training or inference")
        candidates = [entry for entry in self.for_platform(platform) if entry.format == format]
        if not candidates:
            raise PrecisionContractError(f"no {format} capability declared for {platform}")
        entry = candidates[0]
        supported = entry.supports_training if purpose == "training" else entry.supports_inference
        if not entry.available or not supported:
            detail = entry.reason or "capability is unavailable"
            raise RuntimeError(f"{platform} {format} {purpose} is gated: {detail}")
        return entry

    def to_dict(self) -> list[dict[str, Any]]:
        return [entry.__dict__.copy() for entry in self.entries]


@dataclass(frozen=True)
class QuantizedTensor:
    values: torch.Tensor
    scale: torch.Tensor
    format: str


def _per_tensor_scale(source: torch.Tensor, fmax: float) -> torch.Tensor:
    """FP32 scale = amax / fmax using a tensor divisor.

    Dividing by a Python scalar can be lowered to a reciprocal multiply on GPU, which
    differs from the CPU result by 1 ulp; a same-dtype/device tensor divisor is a true
    IEEE division on every device.
    """
    amax = source.abs().amax().clamp_min(torch.finfo(torch.float32).eps)
    return amax / torch.full_like(amax, fmax)


def quantize_per_tensor(tensor: torch.Tensor, *, format: str = "int8_symmetric") -> QuantizedTensor:
    if format not in {"int8_symmetric", "float8_e4m3fn"}:
        raise PrecisionContractError(f"unsupported quantization format {format!r}")
    if not tensor.is_floating_point() or tensor.numel() == 0:
        raise ValueError("quantization requires a nonempty floating point tensor")
    source = tensor.float()
    if format == "int8_symmetric":
        scale = _per_tensor_scale(source, 127.0)
        values = torch.round(source / scale).clamp(-127, 127).to(torch.int8)
    else:
        dtype = getattr(torch, "float8_e4m3fn", None)
        if dtype is None:
            raise RuntimeError("PyTorch float8_e4m3fn is unavailable")
        scale = _per_tensor_scale(source, 448.0)
        values = (source / scale).to(dtype)
    return QuantizedTensor(values, scale.reshape(()), format)


def dequantize_per_tensor(quantized: QuantizedTensor, *, dtype: torch.dtype) -> torch.Tensor:
    if quantized.values.numel() == 0 or quantized.scale.numel() != 1:
        raise PrecisionContractError("quantized tensor has invalid values or scale")
    if dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("dequantization dtype must be FP16, BF16, or FP32")
    return (quantized.values.float() * quantized.scale.float()).to(dtype)


def quantized_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    input_format: str = "int8_symmetric",
    weight_format: str = "int8_symmetric",
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[-1]:
        raise ValueError("input and weight have incompatible linear shapes")
    input_q = quantize_per_tensor(input, format=input_format)
    weight_q = quantize_per_tensor(weight, format=weight_format)
    input_dq = dequantize_per_tensor(input_q, dtype=torch.float32)
    weight_dq = dequantize_per_tensor(weight_q, dtype=torch.float32)
    return torch.nn.functional.linear(input_dq, weight_dq).to(output_dtype)
