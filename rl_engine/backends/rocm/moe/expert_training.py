# SPDX-License-Identifier: Apache-2.0
"""Bind task 45/46 kernels to explicitly supplied task 43/44/49 providers."""

from __future__ import annotations

import importlib
from typing import Protocol

import torch
from torch import Tensor

from rl_engine.contracts.operators.expert_training import (
    ExpertActivationState,
    ExpertTrainingConfig,
    PackedExpertWeight,
    QuantizedExpertActivation,
)


class QuantizedExpertBackend(Protocol):
    """Task 43/44 owner supplies this adapter; it must never fall back to a reference.

    Backward accepts BF16 dY and returns FP32 dX only. Activation quantization
    uses identity STE. The declared profile fixes weight decoding and rounding.
    """

    profile: str
    quant_backward: str

    def validate(self, config: ExpertTrainingConfig, device) -> None: ...
    def quantize(self, x: Tensor) -> QuantizedExpertActivation: ...

    def forward(
        self, q: QuantizedExpertActivation, w: PackedExpertWeight, offsets: Tensor
    ) -> Tensor: ...
    def backward(self, dy: Tensor, w: PackedExpertWeight, offsets: Tensor) -> Tensor: ...


class RouteGradientBackend(Protocol):
    """Task 49 supplies the ordered dp_s reduction; task 48 only connects it."""

    profile: str

    def validate(self, config: ExpertTrainingConfig, device) -> None: ...

    def backward(
        self, dh: Tensor, gate: Tensor, up: Tensor, p: Tensor, state: ExpertActivationState
    ) -> Tensor: ...


def _load(module: str, symbol: str, task: str):
    try:
        return getattr(importlib.import_module(module), symbol)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"task {task} dependency unavailable: {module}:{symbol}") from exc


class RocmExpertTrainingBackend:
    """Explicit eager composition. No automatic registration or kernel substitution."""

    def __init__(
        self,
        base: QuantizedExpertBackend | None = None,
        route_gradient: RouteGradientBackend | None = None,
    ):
        self.base = base
        self.route_gradient = route_gradient
        self._lora = None

    @property
    def profile(self) -> str:
        base = "absent" if self.base is None else self.base.profile
        route = "absent" if self.route_gradient is None else self.route_gradient.profile
        return (
            f"dsv4-expert-training-v1;base={base};route={route};"
            "lora=dsv4-lora-rocm-fp32-v1;activation=dsv4-clamp-swiglu-rocm-fp32-v1"
        )

    def validate(self, config, device, need_route_grad):
        if not torch.version.hip or device.type != "cuda":
            raise RuntimeError("expert training requires ROCm GPU tensors; no CPU fallback")
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        if arch not in ("gfx942", "gfx950"):
            raise RuntimeError(f"expert training candidate requires gfx942/gfx950, got {arch}")
        if config.use_base:
            if self.base is None:
                raise RuntimeError("tasks 43/44 require an explicit quantization/GEMM backend")
            if not self.base.profile or self.base.quant_backward != "identity-ste":
                raise ValueError("base backend must declare its profile and identity STE")
            self.base.validate(config, device)
        if need_route_grad:
            if self.route_gradient is None:
                raise RuntimeError("task 49 route-gradient backend is required for trainable p_s")
            if not self.route_gradient.profile:
                raise ValueError("route-gradient backend must declare its numerical profile")
            self.route_gradient.validate(config, device)
        _load("rl_engine.backends.rocm.moe.clamp_swiglu", "RocmClampSwiGLUWeighted", "46 / #607")
        activation_profile = _load(
            "rl_engine.contracts.operators.clamp_swiglu", "PROFILE", "46 / #607"
        )
        if activation_profile != "dsv4-clamp-swiglu-rocm-fp32-v1":
            raise RuntimeError("task 46 numerical profile is incompatible with this composition")
        if config.use_lora and self._lora is None:
            cls = _load("rl_engine.backends.rocm.moe.lora_delta", "RocmLoRADelta", "45 / #601")
            if cls.numeric_profile != "dsv4-lora-rocm-fp32-v1":
                raise RuntimeError(
                    "task 45 numerical profile is incompatible with this composition"
                )
            self._lora = cls()  # stateless GEMM wrapper, not a weight/transposed-weight cache

    def quantize(self, x):
        if self.base is None:
            raise RuntimeError("tasks 43/44 backend is missing")
        return self.base.quantize(x)

    def base_forward(self, q, w, offsets):
        if self.base is None:
            raise RuntimeError("tasks 43/44 backend is missing")
        return self.base.forward(q, w, offsets)

    def base_backward(self, dy, w, offsets):
        if self.base is None:
            raise RuntimeError("tasks 43/44 backend is missing")
        return self.base.backward(dy, w, offsets)

    def lora_forward(self, x, a, b, alpha):
        if self._lora is None:
            raise RuntimeError("task 45 must be validated before execution")
        return self._lora.forward(x, a, b, alpha)

    def lora_backward(self, dy, x, a, b, alpha, u):
        if self._lora is None:
            raise RuntimeError("task 45 must be validated before execution")
        return self._lora.backward(dy, x, a, b, alpha, u)

    def activation_forward(self, gate, up, p):
        cls = _load(
            "rl_engine.backends.rocm.moe.clamp_swiglu", "RocmClampSwiGLUWeighted", "46 / #607"
        )
        h, state = cls.forward(gate, up, p)
        return h, ExpertActivationState(state.sigmoid, state.silu, state.unweighted)

    def activation_backward(self, dh, gate, up, p, state, need_route_grad):
        cls = _load(
            "rl_engine.backends.rocm.moe.clamp_swiglu", "RocmClampSwiGLUWeighted", "46 / #607"
        )
        saved_cls = _load(
            "rl_engine.contracts.operators.clamp_swiglu", "ClampSwiGLUState", "46 / #607"
        )
        dg, du = cls.backward(dh, gate, up, p, saved_cls(*state))
        dp = None
        if need_route_grad:
            if self.route_gradient is None:
                raise RuntimeError("task 49 route-gradient backend is missing")
            dp = self.route_gradient.backward(dh, gate, up, p, state)
        return dg, du, dp
