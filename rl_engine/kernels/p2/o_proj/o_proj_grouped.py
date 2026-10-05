# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Public T06-B grouped output projection. GEMM reuses DetGemmOp; RoPE consumes T02."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor

from rl_engine.kernels.p2.contract import (
    KERNEL_ID_O_PROJ_DET_GEMM,
    KERNEL_ID_O_PROJ_ORACLE,
    KERNEL_ID_O_PROJ_TORCH_FP32,
)
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status
from rl_engine.kernels.p2.o_proj.oracle import (
    OProjForwardTensors,
    o_proj_grouped_bwd,
    o_proj_grouped_fwd,
    sequential_linear,
)
from rl_engine.kernels.p2.provenance import ActualProvenance


def _det_gemm_linear() -> Callable[[Tensor, Tensor], Tensor] | None:
    try:
        from rl_engine.kernels.p2.cuda_runtime import ensure_native_kernels

        ensure_native_kernels()
    except Exception:
        pass
    from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or _C is None or not hasattr(_C, "det_gemm_fwd_rhs_transposed"):
        return None

    class _DetLinearFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight):
            ctx.save_for_backward(x, weight)
            return _C.det_gemm_fwd_rhs_transposed(x, weight)

        @staticmethod
        def backward(ctx, grad):
            x, weight = ctx.saved_tensors
            g = grad.contiguous()
            if g.dtype != torch.bfloat16:
                g = g.to(torch.bfloat16)
            dx = _C.det_gemm_fwd(g, weight) if ctx.needs_input_grad[0] else None
            dw = _C.det_gemm_db_transposed(x, g) if ctx.needs_input_grad[1] else None
            return dx, dw

    def _linear(x: Tensor, weight: Tensor) -> Tensor:
        if not x.is_cuda or not weight.is_cuda:
            raise P2FailClosedError(
                P2Status.UNSUPPORTED_CAPABILITY,
                "DetGemm grouped o-proj requires CUDA tensors",
            )
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise P2FailClosedError(
                P2Status.ROUND_POINT_MISMATCH,
                "det_gemm o-proj requires BF16 inputs; refusing silent downcast",
            )
        return _DetLinearFn.apply(x.contiguous(), weight.contiguous())

    return _linear


def torch_fp32_linear(x: Tensor, weight: Tensor) -> Tensor:
    """Declared GPU/CPU FP32 matmul. Not DetGemm; not a silent fallback."""

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
            raise P2FailClosedError(
                P2Status.UNSUPPORTED_CAPABILITY,
                f"backend must be auto|oracle|det_gemm|torch_fp32, got {backend!r}",
            )
        self.backend = backend

    def _linear(self, x: Tensor) -> tuple[Callable[[Tensor, Tensor], Tensor], str, str]:
        if self.backend == "oracle":
            return sequential_linear, "oracle", KERNEL_ID_O_PROJ_ORACLE
        if self.backend == "torch_fp32":
            return torch_fp32_linear, "torch_fp32", KERNEL_ID_O_PROJ_TORCH_FP32
        if self.backend == "det_gemm":
            fn = _det_gemm_linear()
            if fn is None:
                raise P2FailClosedError(
                    P2Status.UNSUPPORTED_CAPABILITY,
                    "det_gemm backend requested but DetGemmOp is unavailable",
                )
            return fn, "det_gemm", KERNEL_ID_O_PROJ_DET_GEMM
        # auto: CUDA requires DetGemm; CPU uses sequential oracle. No silent mix.
        if x.is_cuda:
            fn = _det_gemm_linear()
            if fn is None:
                raise P2FailClosedError(
                    P2Status.UNSUPPORTED_CAPABILITY,
                    "auto o-proj on CUDA requires DetGemmOp; refusing sequential/torch fallback",
                )
            return fn, "det_gemm", KERNEL_ID_O_PROJ_DET_GEMM
        return sequential_linear, "oracle", KERNEL_ID_O_PROJ_ORACLE

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
        linear, backend, _ = self._linear(saved.w_a)
        gemm_dtype = torch.bfloat16 if backend == "det_gemm" else None
        return o_proj_grouped_bwd(
            d_y, saved, cos, sin, linear=linear, gemm_dtype=gemm_dtype
        )

    def __call__(self, o: Tensor, w_a: Tensor, w_b: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        return self.forward(o, w_a, w_b, cos, sin).y
