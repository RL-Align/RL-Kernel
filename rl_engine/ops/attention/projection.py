# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""QKV and output-projection boundaries for the WS2 Attention experiment.

The framework still owns the native TE/vLLM implementation.  This wrapper only
freezes the semantics that must be shared by training and inference: BF16 I/O,
FP32 accumulation, ascending K reduction, and no Split-K.  A native callable is
accepted only when its result is bitwise equal to the deterministic fallback.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch
from torch import Tensor

from rl_engine.backends.cuda.gemm.det_gemm import DetGemmOp

ProjectionCallable = Callable[[Tensor, Tensor], Tensor]

QKV_PROJECTION = "qkv"
O_PROJ_PROJECTION = "o_proj"
PROJECTION_POLICY_ID = "ws2.attention.projection.v1"
CUDA_DETERMINISTIC_PROJECTION_BACKEND_ID = "rlkernel.cuda.det_gemm"
ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID = "rlkernel.rocm.triton_det_gemm"


@dataclass(frozen=True)
class ProjectionCollectiveContract:
    """TP/SP directions fixed by the Attention table."""

    projection: str
    tp_forward: str
    tp_backward: str
    sp_forward: str
    sp_backward: str
    reduction_forward: str
    reduction_backward: str

    def __post_init__(self) -> None:
        if self.projection not in {QKV_PROJECTION, O_PROJ_PROJECTION}:
            raise ValueError(f"unsupported projection {self.projection!r}")

    def to_dict(self) -> dict[str, str]:
        return {
            "projection": self.projection,
            "tp_forward": self.tp_forward,
            "tp_backward": self.tp_backward,
            "sp_forward": self.sp_forward,
            "sp_backward": self.sp_backward,
            "reduction_forward": self.reduction_forward,
            "reduction_backward": self.reduction_backward,
        }


QKV_COLLECTIVE_CONTRACT = ProjectionCollectiveContract(
    projection=QKV_PROJECTION,
    tp_forward="column_parallel",
    tp_backward="all_reduce",
    sp_forward="all_gather",
    sp_backward="reduce_scatter",
    reduction_forward="none",
    reduction_backward="none",
)
O_PROJ_COLLECTIVE_CONTRACT = ProjectionCollectiveContract(
    projection=O_PROJ_PROJECTION,
    tp_forward="row_parallel",
    tp_backward="none",
    sp_forward="reduce_scatter",
    sp_backward="all_gather",
    reduction_forward="all_reduce",
    reduction_backward="none",
)


@dataclass(frozen=True)
class ProjectionPlan:
    projection: str
    backend_id: str
    fallback: bool
    fallback_reason: str | None
    probe_id: str
    input_dtype: str = "torch.bfloat16"
    weight_dtype: str = "torch.bfloat16"
    output_dtype: str = "torch.bfloat16"
    accumulation_dtype: str = "torch.float32"
    reduction_order: str = "k_ascending"
    split_k: bool = False
    policy_id: str = PROJECTION_POLICY_ID
    collective: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.projection not in {QKV_PROJECTION, O_PROJ_PROJECTION}:
            raise ValueError(f"unsupported projection {self.projection!r}")
        if self.input_dtype != "torch.bfloat16" or self.weight_dtype != "torch.bfloat16":
            raise ValueError("Attention projections require BF16 input and weight")
        if self.output_dtype != "torch.bfloat16" or self.accumulation_dtype != "torch.float32":
            raise ValueError("Attention projections require FP32 accumulation and BF16 output")
        if self.reduction_order != "k_ascending" or self.split_k:
            raise ValueError(
                "Attention projections require ascending K reduction with Split-K disabled"
            )
        object.__setattr__(self, "collective", MappingProxyType(dict(self.collective)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection": self.projection,
            "backend_id": self.backend_id,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "probe_id": self.probe_id,
            "input_dtype": self.input_dtype,
            "weight_dtype": self.weight_dtype,
            "output_dtype": self.output_dtype,
            "accumulation_dtype": self.accumulation_dtype,
            "reduction_order": self.reduction_order,
            "split_k": self.split_k,
            "policy_id": self.policy_id,
            "collective": dict(self.collective),
        }


@dataclass(frozen=True)
class ProjectionResult:
    output: Tensor
    plan: ProjectionPlan

    def to_readback(self) -> dict[str, Any]:
        return self.plan.to_dict()


class AttentionProjectionOp:
    """Native-first projection wrapper with a deterministic common fallback."""

    def __init__(
        self,
        projection: str,
        *,
        native: ProjectionCallable | None = None,
        native_backend_id: str | None = None,
        deterministic: ProjectionCallable | None = None,
        deterministic_backend_id: str | None = None,
        policy_id: str = PROJECTION_POLICY_ID,
    ) -> None:
        if projection not in {QKV_PROJECTION, O_PROJ_PROJECTION}:
            raise ValueError(f"unsupported projection {projection!r}")
        self.projection = projection
        self.native = native
        self.native_backend_id = native_backend_id or f"native.{projection}"
        self.deterministic = deterministic or DetGemmOp()
        self.deterministic_backend_id = deterministic_backend_id or (
            ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID
            if torch.version.hip is not None
            else CUDA_DETERMINISTIC_PROJECTION_BACKEND_ID
        )
        self.policy_id = policy_id
        self.collective = (
            QKV_COLLECTIVE_CONTRACT if projection == QKV_PROJECTION else O_PROJ_COLLECTIVE_CONTRACT
        )

    def __call__(self, x: Tensor, weight: Tensor) -> ProjectionResult:
        _validate_projection_inputs(x, weight)
        deterministic_out = self.deterministic(x, weight)
        if deterministic_out.dtype is not torch.bfloat16:
            deterministic_out = deterministic_out.to(torch.bfloat16)
        probe_id = _probe_id(x, weight)

        if self.native is not None:
            try:
                native_out = self.native(x, weight)
                if native_out.dtype is not torch.bfloat16:
                    native_out = native_out.to(torch.bfloat16)
                if torch.equal(native_out, deterministic_out):
                    return ProjectionResult(
                        native_out,
                        ProjectionPlan(
                            projection=self.projection,
                            backend_id=self.native_backend_id,
                            fallback=False,
                            fallback_reason=None,
                            probe_id=probe_id,
                            policy_id=self.policy_id,
                            collective=self.collective.to_dict(),
                        ),
                    )
                reason = "native_projection_bitwise_probe_failed"
            except Exception as exc:  # framework backend failure: use common fallback
                reason = f"native_projection_unavailable:{type(exc).__name__}"
        else:
            reason = "native_projection_not_supplied"

        return ProjectionResult(
            deterministic_out,
            ProjectionPlan(
                projection=self.projection,
                backend_id=self.deterministic_backend_id,
                fallback=True,
                fallback_reason=reason,
                probe_id=probe_id,
                policy_id=self.policy_id,
                collective=self.collective.to_dict(),
            ),
        )


def split_qkv(
    projected_qkv: Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Split a [Q, K, V] projection in the fixed contiguous Q/K/V order."""

    if projected_qkv.dim() != 2:
        raise ValueError("projected QKV must be [tokens, features]")
    q_width = q_heads * head_dim
    kv_width = kv_heads * head_dim
    expected = q_width + kv_width + kv_width
    if projected_qkv.shape[-1] != expected:
        raise ValueError(f"projected QKV width must be {expected}, got {projected_qkv.shape[-1]}")
    q, k, v = projected_qkv.split((q_width, kv_width, kv_width), dim=-1)
    return q, k, v


def _validate_projection_inputs(x: Tensor, weight: Tensor) -> None:
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError("projection inputs must be [tokens, K] and [K, N]")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError("projection K dimensions must match")
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        raise TypeError("Attention projections require BF16 inputs and weights")
    if x.device != weight.device:
        raise ValueError("projection inputs must be on the same device")


def _probe_id(x: Tensor, weight: Tensor) -> str:
    payload = {
        "x_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "device": str(x.device),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


__all__ = [
    "AttentionProjectionOp",
    "CUDA_DETERMINISTIC_PROJECTION_BACKEND_ID",
    "O_PROJ_COLLECTIVE_CONTRACT",
    "O_PROJ_PROJECTION",
    "PROJECTION_POLICY_ID",
    "ProjectionCollectiveContract",
    "ProjectionPlan",
    "ProjectionResult",
    "QKV_COLLECTIVE_CONTRACT",
    "QKV_PROJECTION",
    "ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID",
    "split_qkv",
]
