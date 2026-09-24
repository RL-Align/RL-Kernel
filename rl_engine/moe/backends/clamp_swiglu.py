# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA provider for P5-2 clamp_swiglu_weighted."""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.moe.contract import ORACLE_PROFILE
from rl_engine.moe.provider import ReferenceProvider

_FORWARD_SYMBOL = "clamp_swiglu_weighted_forward"
_BACKWARD_SYMBOL = "clamp_swiglu_weighted_backward"
_PACKED_FORWARD_SYMBOL = "clamp_swiglu_weighted_packed_forward"
_PACKED_BACKWARD_SYMBOL = "clamp_swiglu_weighted_packed_backward"

_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)

_SAVED_KEYS = (
    "gate32",
    "up32",
)


def _require_cuda_extension() -> None:
    if not _EXT_AVAILABLE or _C is None:
        raise RuntimeError(
            "ClampSwiGLUWeightedCudaProvider requires the compiled " "rl_engine._C extension."
        )

    required_symbols = (
        _FORWARD_SYMBOL,
        _BACKWARD_SYMBOL,
        _PACKED_FORWARD_SYMBOL,
        _PACKED_BACKWARD_SYMBOL,
    )

    missing = [name for name in required_symbols if not hasattr(_C, name)]

    if missing:
        raise RuntimeError(
            "P5-2 CUDA symbols are unavailable: "
            f"{', '.join(missing)}. "
            "Rebuild rl_engine._C from this branch."
        )


def _validate_matrix(
    x: torch.Tensor,
    name: str,
) -> None:
    if x.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor, got {x.device}.")

    if x.dim() != 2:
        raise ValueError(f"{name} must be 2D [rows, width], " f"got shape {tuple(x.shape)}.")

    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, " f"got {x.dtype}.")


def _validate_packed_matrix(
    gate_up: torch.Tensor,
) -> None:
    _validate_matrix(
        gate_up,
        "gate_up",
    )

    if gate_up.shape[1] % 2 != 0:
        raise ValueError(
            "gate_up must have shape [rows, 2 * width], " f"got shape {tuple(gate_up.shape)}."
        )


class ClampSwiGLUWeightedCudaProvider(ReferenceProvider):
    """P5-2 CUDA provider with deterministic FP32 computation."""

    name = "cuda-clamp-swiglu-weighted"
    numeric_profile = ORACLE_PROFILE

    def __init__(self) -> None:
        _require_cuda_extension()

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "cuda",
            "delivered_ops": ["clamp_swiglu_weighted"],
            "geometry": ["one-row", "packed"],
            "layouts": ["separate", "gate-up-packed"],
            "devices": ["cuda"],
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": self.name,
            "numeric_profile": self.numeric_profile,
            "torch_version": torch.__version__,
        }

    def clamp_swiglu_weighted_fwd(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        p_s: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _validate_matrix(gate, "gate")
        _validate_matrix(up, "up")

        if gate.device != up.device:
            raise RuntimeError(
                "gate and up must share a CUDA device, " f"got {gate.device} and {up.device}."
            )

        if gate.shape != up.shape:
            raise ValueError("gate and up must share shape, " f"got {gate.shape} and {up.shape}.")

        if p_s is not None:
            if p_s.device != gate.device:
                raise RuntimeError(
                    "p_s and gate must share a CUDA device, " f"got {p_s.device} and {gate.device}."
                )

            if p_s.dtype != torch.float32:
                raise TypeError(f"p_s must have dtype fp32, got {p_s.dtype}.")

            if p_s.shape != (gate.shape[0],):
                raise ValueError(
                    f"p_s must have shape ({gate.shape[0]},), " f"got {tuple(p_s.shape)}."
                )

        gate32 = gate.float().contiguous()
        up32 = up.float().contiguous()
        p_s32 = None if p_s is None else p_s.contiguous()

        (h,) = getattr(
            _C,
            _FORWARD_SYMBOL,
        )(
            gate32,
            up32,
            p_s32,
        )

        saved = {
            "gate32": gate32,
            "up32": up32,
        }

        if p_s32 is not None:
            saved["p_s"] = p_s32

        return h, saved

    def clamp_swiglu_weighted_bwd(
        self,
        dh: torch.Tensor,
        saved: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        missing = [key for key in _SAVED_KEYS if key not in saved]

        if missing:
            raise KeyError(f"saved state is missing: {', '.join(missing)}")

        _validate_matrix(dh, "dh")

        gate32 = saved["gate32"]

        if dh.device != gate32.device:
            raise RuntimeError(
                "dh and saved tensors must share a CUDA device, "
                f"got {dh.device} and {gate32.device}."
            )

        if dh.shape != gate32.shape:
            raise ValueError(
                "dh and saved tensors must share shape, " f"got {dh.shape} and {gate32.shape}."
            )

        p_s = saved.get("p_s")

        dgate, dup, dp_s = getattr(
            _C,
            _BACKWARD_SYMBOL,
        )(
            dh.contiguous(),
            gate32,
            saved["up32"],
            p_s,
        )

        return (
            dgate,
            dup,
            dp_s if p_s is not None else None,
        )

    def clamp_swiglu_weighted_packed_fwd(
        self,
        gate_up: torch.Tensor,
        p_s: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _validate_packed_matrix(gate_up)

        if p_s is not None:
            if p_s.device != gate_up.device:
                raise RuntimeError(
                    "p_s and gate_up must share a CUDA device, "
                    f"got {p_s.device} and {gate_up.device}."
                )

            if p_s.dtype != torch.float32:
                raise TypeError(f"p_s must have dtype fp32, got {p_s.dtype}.")

            if p_s.shape != (gate_up.shape[0],):
                raise ValueError(
                    f"p_s must have shape ({gate_up.shape[0]},), " f"got {tuple(p_s.shape)}."
                )

        gate_up32 = gate_up.float().contiguous()
        p_s32 = None if p_s is None else p_s.contiguous()

        (h,) = getattr(
            _C,
            _PACKED_FORWARD_SYMBOL,
        )(
            gate_up32,
            p_s32,
        )

        saved = {
            "gate_up32": gate_up32,
        }

        if p_s32 is not None:
            saved["p_s"] = p_s32

        return h, saved

    def clamp_swiglu_weighted_packed_bwd(
        self,
        dh: torch.Tensor,
        saved: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        if "gate_up32" not in saved:
            raise KeyError("saved state is missing: gate_up32")

        _validate_matrix(dh, "dh")

        gate_up32 = saved["gate_up32"]
        rows = gate_up32.shape[0]
        width = gate_up32.shape[1] // 2

        if dh.device != gate_up32.device:
            raise RuntimeError(
                "dh and gate_up32 must share a CUDA device, "
                f"got {dh.device} and {gate_up32.device}."
            )

        if dh.shape != (rows, width):
            raise ValueError(f"dh must have shape ({rows}, {width}), " f"got {tuple(dh.shape)}.")

        p_s = saved.get("p_s")

        dgate, dup, dp_s = getattr(
            _C,
            _PACKED_BACKWARD_SYMBOL,
        )(
            dh.contiguous(),
            gate_up32,
            p_s,
        )

        return (
            dgate,
            dup,
            dp_s if p_s is not None else None,
        )
