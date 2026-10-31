# SPDX-License-Identifier: Apache-2.0
"""Explicit sparse Indexer auxiliary/VJP profile; no inferred model geometry."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from rl_engine.contracts.operators.attention import AttentionContractError


class IndexerTrainingError(AttentionContractError):
    pass


def require(condition, message):
    if not condition:
        raise IndexerTrainingError(message)


@dataclass(frozen=True)
class IndexerAuxContract:
    global_query_count: int
    loss_coeff: float
    num_heads: int
    teacher_heads: int
    teacher_dim: int
    selected_capacity: int = 512
    head_dim: int = 128
    ratio: int = 4
    precision: str = "mxfp4"
    max_queries: int = 512
    max_keys: int = 131072
    max_workspace_bytes: int = 1 << 30
    cp_size: int = 1
    teacher_scale: float | None = None
    graph_capture: bool = False
    quant_vjp: str = "dequantized_operand_identity_ste.v1"
    loss_profile: str = "mlite.sparse_kl.eps1e-10.global_query_mean.v1"

    def __post_init__(self):
        for name in (
            "global_query_count",
            "num_heads",
            "teacher_heads",
            "teacher_dim",
            "selected_capacity",
            "head_dim",
            "ratio",
            "max_queries",
            "max_keys",
            "max_workspace_bytes",
            "cp_size",
        ):
            require(
                type(getattr(self, name)) is int and getattr(self, name) > 0,
                f"{name} must be a positive integer",
            )
        require(
            self.head_dim == 128 and self.teacher_dim in (128, 512),
            "Indexer D128 and teacher D128/D512 are the admitted geometry",
        )
        require(self.num_heads <= 128 and self.teacher_heads <= 128, "head count exceeds profile")
        require(
            self.ratio == 4, "the separate Indexer compressor is C4; main C128 is a different path"
        )
        require(self.selected_capacity <= 512, "larger sparse active sets require separate signoff")
        require(
            self.max_queries < 2**31
            and self.max_keys < 2**31
            and self.max_queries * self.selected_capacity < 2**31,
            "index capacity exceeds int32",
        )
        require(self.precision in ("mxfp4", "fp8"), "explicit quantization profile required")
        require(
            type(self.loss_coeff) in (int, float)
            and math.isfinite(self.loss_coeff)
            and self.loss_coeff >= 0,
            "loss coefficient must be finite and nonnegative",
        )
        require(
            self.teacher_scale is None
            or (
                type(self.teacher_scale) in (int, float)
                and math.isfinite(self.teacher_scale)
                and self.teacher_scale > 0
            ),
            "teacher softmax scale must be finite and positive",
        )
        require(self.graph_capture is False, "whole auxiliary/autograd Graph is not validated")
        require(
            self.quant_vjp == "dequantized_operand_identity_ste.v1",
            "unknown quantization surrogate",
        )
        require(
            self.loss_profile == "mlite.sparse_kl.eps1e-10.global_query_mean.v1",
            "unknown auxiliary objective",
        )

    @property
    def softmax_scale(self):
        return self.teacher_dim**-0.5 if self.teacher_scale is None else float(self.teacher_scale)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
