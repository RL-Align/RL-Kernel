# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Grouped output projection with inverse GPT-J RoPE."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor

from rl_engine.kernels.dsv4.attention.contract import (
    KERNEL_ID_O_PROJ_DET_GEMM,
    KERNEL_ID_O_PROJ_ORACLE,
    KERNEL_ID_O_PROJ_TORCH_FP32,
)
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status
from rl_engine.kernels.dsv4.attention.o_proj.oracle import (
    OProjForwardTensors,
    o_proj_grouped_bwd,
    o_proj_grouped_fwd,
    sequential_linear,
)
from rl_engine.kernels.dsv4.attention.provenance import ActualProvenance


def _check_devices(reference: Tensor, *tensors: Tensor) -> None:
    if any(tensor.device != reference.device for tensor in tensors):
        raise DSv4FailClosedError(
            DSv4Status.SCHEMA_MISMATCH,
            "grouped o-proj tensors must be on the same device",
        )


def _det_gemm_linear() -> Callable[[Tensor, Tensor], Tensor] | None:
    try:
        from rl_engine.kernels.dsv4.attention.cuda_runtime import ensure_native_kernels

        ensure_native_kernels()
    except Exception:
        pass
    from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or _C is None:
        return None

    from rl_engine.kernels.ops.cuda.matmul import det_gemm

    # JIT may refresh _C after this module was imported.
    det_gemm._C = _C
    det_gemm._EXT_AVAILABLE = _EXT_AVAILABLE
    try:
        linear = det_gemm.DetGemmOp().linear
    except RuntimeError as exc:
        raise DSv4FailClosedError(DSv4Status.UNSUPPORTED_CAPABILITY, str(exc)) from exc

    def _linear(x: Tensor, weight: Tensor) -> Tensor:
        _check_devices(x, weight)
        if not x.is_cuda or not weight.is_cuda:
            raise DSv4FailClosedError(
                DSv4Status.UNSUPPORTED_CAPABILITY,
                "DetGemm grouped o-proj requires CUDA tensors",
            )
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise DSv4FailClosedError(
                DSv4Status.ROUND_POINT_MISMATCH,
                "det_gemm o-proj requires BF16 inputs; refusing silent downcast",
            )
        return linear(x, weight)

    return _linear


def torch_fp32_linear(x: Tensor, weight: Tensor) -> Tensor:
    """FP32 matmul with TF32 disabled."""

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return x.float() @ weight.float().transpose(-2, -1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


@dataclass
class OProjResult:
    y: Tensor
    provenance: ActualProvenance
    saved: OProjForwardTensors | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"provenance": self.provenance.to_dict()}


class OProjGroupedOp:
    def __init__(self, *, backend: str = "oracle") -> None:
        if backend not in {"auto", "oracle", "det_gemm", "torch_fp32"}:
            raise DSv4FailClosedError(
                DSv4Status.UNSUPPORTED_CAPABILITY,
                f"backend must be auto|oracle|det_gemm|torch_fp32, got {backend!r}",
            )
        self.backend = backend

    def _linear(self, x: Tensor) -> tuple[Callable[[Tensor, Tensor], Tensor], str, str]:
        if self.backend == "oracle" or (self.backend == "auto" and not x.is_cuda):
            return sequential_linear, "oracle", KERNEL_ID_O_PROJ_ORACLE
        if self.backend == "torch_fp32":
            return torch_fp32_linear, "torch_fp32", KERNEL_ID_O_PROJ_TORCH_FP32
        if x.is_cuda and x.dtype != torch.bfloat16:
            raise DSv4FailClosedError(
                DSv4Status.ROUND_POINT_MISMATCH,
                "det_gemm o-proj requires BF16 inputs; refusing silent downcast",
            )
        with torch.cuda.device(x.device) if x.is_cuda else nullcontext():
            fn = _det_gemm_linear()
        if fn is None:
            raise DSv4FailClosedError(
                DSv4Status.UNSUPPORTED_CAPABILITY,
                "grouped o-proj requires all DetGemm forward and backward exports",
            )
        return fn, "det_gemm", KERNEL_ID_O_PROJ_DET_GEMM

    def forward(
        self,
        o: Tensor,
        w_a: Tensor,
        w_b: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> OProjResult:
        _check_devices(o, w_a, w_b, cos, sin)
        linear, backend, kernel_id = self._linear(o)
        saved = o_proj_grouped_fwd(o, w_a, w_b, cos, sin, linear=linear)
        if output_dtype is None or saved.y.dtype == output_dtype:
            y = saved.y
        else:
            y = saved.y.to(output_dtype)
        return OProjResult(
            y=y,
            provenance=ActualProvenance(
                backend=backend,
                kernel_id=kernel_id,
                device=str(o.device),
                dtype=str(y.dtype),
                extra={"o_storage_unchanged": o.data_ptr() == saved.o.data_ptr()},
            ),
            saved=saved,
        )

    def forward_fp32(
        self, o: Tensor, w_a: Tensor, w_b: Tensor, cos: Tensor, sin: Tensor
    ) -> OProjResult:
        return self.forward(o, w_a, w_b, cos, sin, output_dtype=torch.float32)

    def backward(
        self,
        d_y: Tensor,
        saved: OProjForwardTensors,
        cos: Tensor,
        sin: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        _check_devices(
            saved.o, d_y, saved.o_tilde, *saved.z_groups, saved.z,
            saved.w_a, saved.w_b, cos, sin,
        )
        linear, backend, _ = self._linear(saved.w_a)
        gemm_dtype = torch.bfloat16 if backend == "det_gemm" else None
        return o_proj_grouped_bwd(
            d_y, saved, cos, sin, linear=linear, gemm_dtype=gemm_dtype
        )

    def __call__(self, o: Tensor, w_a: Tensor, w_b: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        return self.forward(o, w_a, w_b, cos, sin).y
