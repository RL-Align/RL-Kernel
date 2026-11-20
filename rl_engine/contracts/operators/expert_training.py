# SPDX-License-Identifier: Apache-2.0
"""Task 48: local, already-dispatched expert training boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import torch
from torch import Tensor

SCHEMA = "dsv4-expert-training-v1"
MX_LAYOUT = "ocp-mx-block32-row-major-low-nibble-first-v1"


@dataclass(frozen=True)
class ExpertTrainingConfig:
    hidden: int
    ffn: int
    experts: int
    mode: str = "combined"
    alpha: float = 1.0

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in (self.hidden, self.ffn, self.experts)):
            raise ValueError("hidden, ffn and experts must be positive integers")
        if self.mode not in ("base_only", "lora_only", "combined"):
            raise ValueError("mode must be base_only, lora_only or combined")
        if self.use_base and (self.hidden % 32 or self.ffn % 32):
            raise ValueError("MX base requires hidden and ffn divisible by 32")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise TypeError("alpha must be a non-trainable effective multiplier")
        if not math.isfinite(self.alpha) or abs(self.alpha) > torch.finfo(torch.float32).max:
            raise ValueError("alpha must be finite and representable in FP32")

    @property
    def use_base(self) -> bool:
        return self.mode != "lora_only"

    @property
    def use_lora(self) -> bool:
        return self.mode != "base_only"


def require_tensor(t: Tensor, shape: tuple[int, ...], dtype: torch.dtype, device, name: str):
    if t.layout != torch.strided or tuple(t.shape) != shape or t.device != device:
        raise ValueError(f"{name} must be strided, shape {shape}, on {device}")
    if t.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {t.dtype}")


@dataclass(frozen=True)
class PackedExpertWeight:
    codes: Tensor
    scales: Tensor
    shape: tuple[int, int, int]
    layout: str = MX_LAYOUT

    def validate(self, shape: tuple[int, int, int], device) -> None:
        if self.shape != shape or self.layout != MX_LAYOUT or shape[-1] % 32:
            raise ValueError("packed weight shape/layout does not match the block-32 contract")
        e, n, k = shape
        require_tensor(self.codes, (e, n, k // 2), torch.uint8, device, "weight codes")
        require_tensor(self.scales, (e, n, k // 32), torch.uint8, device, "weight scales")
        if self.codes.requires_grad or self.scales.requires_grad:
            raise ValueError("base weights must be frozen")
        if bool((self.scales == 255).any()):
            raise ValueError("E8M0 NaN scale is not supported")


@dataclass(frozen=True)
class QuantizedExpertActivation:
    codes: Tensor
    scales: Tensor
    layout: str = MX_LAYOUT

    def validate(self, x: Tensor):
        m, k = x.shape
        if self.layout != MX_LAYOUT or k % 32:
            raise ValueError("activation must use the declared block-32 layout")
        require_tensor(self.codes, (m, k), torch.uint8, x.device, "activation codes")
        require_tensor(self.scales, (m, k // 32), torch.uint8, x.device, "activation scales")
        if bool((self.scales == 255).any()):
            raise ValueError("E8M0 NaN scale is not supported")


class ExpertActivationState(NamedTuple):
    sigmoid: Tensor
    silu: Tensor
    unweighted: Tensor


class ExpertSavedState(NamedTuple):
    gate: Tensor
    up: Tensor
    h: Tensor
    sigmoid: Tensor
    silu: Tensor
    unweighted: Tensor
    u1: Tensor
    u2: Tensor


class ExpertGradients(NamedTuple):
    dx: Tensor
    dp_s: Tensor | None
    da1: Tensor | None
    db1: Tensor | None
    da2: Tensor | None
    db2: Tensor | None


class ExpertTrainingBackend(Protocol):
    """Explicit provider: no reference implementation is selected implicitly."""

    @property
    def profile(self) -> str: ...

    def validate(self, config: ExpertTrainingConfig, device, need_route_grad: bool) -> None: ...

    def quantize(self, x: Tensor) -> QuantizedExpertActivation: ...

    def base_forward(
        self, q: QuantizedExpertActivation, w: PackedExpertWeight, offsets: Tensor
    ) -> Tensor: ...

    def base_backward(self, dy: Tensor, w: PackedExpertWeight, offsets: Tensor) -> Tensor: ...

    def lora_forward(
        self, x: Tensor, a: Tensor, b: Tensor, alpha: float
    ) -> tuple[Tensor, Tensor]: ...

    def lora_backward(
        self, dy: Tensor, x: Tensor, a: Tensor, b: Tensor, alpha: float, u: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    def activation_forward(
        self, gate: Tensor, up: Tensor, p: Tensor
    ) -> tuple[Tensor, ExpertActivationState]: ...

    def activation_backward(
        self,
        dh: Tensor,
        gate: Tensor,
        up: Tensor,
        p: Tensor,
        state: ExpertActivationState,
        need_route_grad: bool,
    ) -> tuple[Tensor, Tensor, Tensor | None]: ...


def validate_request(config, x, p, offsets, w1, w2, adapters):
    if x.ndim != 2:
        raise ValueError("expert X must be [M, hidden]")
    m = x.shape[0]
    require_tensor(x, (m, config.hidden), torch.bfloat16, x.device, "X")
    require_tensor(p, (m,), torch.float32, x.device, "p_s")
    if offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert offsets must be int32 or int64")
    require_tensor(offsets, (config.experts + 1,), offsets.dtype, x.device, "offsets")
    # This eager boundary intentionally synchronizes. Graph-safe prepared plans
    # and transport-owned validation are separate integration work.
    if offsets[0].item() != 0 or offsets[-1].item() != m:
        raise ValueError("expert offsets must start at 0 and end at M")
    if bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError("expert offsets must be non-decreasing")
    if config.use_base:
        if w1 is None or w2 is None:
            raise ValueError("base mode requires both frozen packed weights")
        w1.validate((config.experts, 2 * config.ffn, config.hidden), x.device)
        w2.validate((config.experts, config.hidden, config.ffn), x.device)
    elif w1 is not None or w2 is not None:
        raise ValueError("lora_only must not carry unused base weights")
    if config.use_lora:
        if len(adapters) != 4 or any(t is None for t in adapters):
            raise ValueError("LoRA requires A1, B1, A2 and B2")
        a1, b1, a2, b2 = adapters
        if a1.ndim != 2 or a1.shape[0] <= 0:
            raise ValueError("LoRA rank must be positive")
        rank = a1.shape[0]
        shapes = (
            (rank, config.hidden),
            (2 * config.ffn, rank),
            (rank, config.ffn),
            (config.hidden, rank),
        )
        for name, t, shape in zip(("a1", "b1", "a2", "b2"), adapters, shapes):
            require_tensor(t, shape, torch.bfloat16, x.device, name)
    elif any(t is not None for t in adapters):
        raise ValueError("base_only must not carry unused LoRA parameters")
