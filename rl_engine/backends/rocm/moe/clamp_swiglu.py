# SPDX-License-Identifier: Apache-2.0
"""Explicit ROCm task 46 candidate; hardware acceptance is separate from eligibility."""

import torch
from torch import Tensor

from rl_engine.contracts.operators.clamp_swiglu import (
    PROFILE,
    ClampSwiGLUState,
    validate_backward,
    validate_inputs,
)

_BLOCK = 256


def _require_rocm(gate: Tensor) -> None:
    if not torch.version.hip or gate.device.type != "cuda":
        raise RuntimeError(f"{PROFILE} requires ROCm; no CPU fallback is registered")
    arch = torch.cuda.get_device_properties(gate.device).gcnArchName.split(":")[0]
    if arch not in ("gfx942", "gfx950"):
        raise RuntimeError(f"{PROFILE} has no kernel candidate for {arch}")


class RocmClampSwiGLUWeighted:
    """MI300X/gfx942 and MI355X/gfx950 candidate, selected explicitly by the caller."""

    @staticmethod
    @torch.no_grad()
    def forward(gate: Tensor, up: Tensor, p_s: Tensor | None) -> tuple[Tensor, ClampSwiGLUState]:
        validate_inputs(gate, up, p_s)
        _require_rocm(gate)
        from .clamp_swiglu_kernels import _forward

        h = torch.empty(gate.shape, device=gate.device, dtype=torch.bfloat16)
        saved = [torch.empty(gate.shape, device=gate.device, dtype=torch.float32) for _ in range(3)]
        state = ClampSwiGLUState(*saved)
        if gate.shape[0]:
            with torch.cuda.device(gate.device):
                _forward[(gate.shape[0], (gate.shape[1] + _BLOCK - 1) // _BLOCK)](
                    gate,
                    up,
                    gate if p_s is None else p_s,
                    h,
                    *saved,
                    gate.shape[1],
                    *gate.stride(),
                    *up.stride(),
                    0 if p_s is None else p_s.stride(0),
                    p_s is not None,
                    _BLOCK,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return h, state

    @staticmethod
    @torch.no_grad()
    def backward(
        dh: Tensor, gate: Tensor, up: Tensor, p_s: Tensor | None, state: ClampSwiGLUState
    ) -> tuple[Tensor, Tensor]:
        validate_backward(dh, gate, up, p_s, state)
        _require_rocm(gate)
        from .clamp_swiglu_kernels import _backward

        dg = torch.empty(gate.shape, device=gate.device, dtype=torch.float32)
        du = torch.empty_like(dg)
        if gate.shape[0]:
            with torch.cuda.device(gate.device):
                _backward[(gate.shape[0], (gate.shape[1] + _BLOCK - 1) // _BLOCK)](
                    dh,
                    gate,
                    up,
                    gate if p_s is None else p_s,
                    state.sigmoid,
                    state.silu,
                    dg,
                    du,
                    gate.shape[1],
                    *dh.stride(),
                    *gate.stride(),
                    *up.stride(),
                    0 if p_s is None else p_s.stride(0),
                    p_s is not None,
                    _BLOCK,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return dg, du
