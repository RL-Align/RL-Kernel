# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5 start kit: MXFP4 Routed Expert + LoRA + Shared Expert contracts (issue #8)."""

from rl_engine.moe.contract import (
    GATE_CLAMP_MAX,
    ORACLE_PROFILE,
    SCHEMA_VERSION,
    UP_CLAMP_MAX,
    UP_CLAMP_MIN,
    ExpertBatch,
    LoRAParams,
    SharedBatch,
    tensor_sha256,
)
from rl_engine.moe.mx_format import MX_BLOCK, MXTensor, mx_dequantize, mx_quantize
from rl_engine.moe.provider import ExpertProvider, ReferenceProvider, StubProvider, resolve_provider
from rl_engine.moe.trace import ExpertTrace, first_divergence

__all__ = [
    "GATE_CLAMP_MAX",
    "ORACLE_PROFILE",
    "SCHEMA_VERSION",
    "UP_CLAMP_MAX",
    "UP_CLAMP_MIN",
    "ExpertBatch",
    "ExpertProvider",
    "ExpertTrace",
    "LoRAParams",
    "MXTensor",
    "MX_BLOCK",
    "ReferenceProvider",
    "SharedBatch",
    "StubProvider",
    "first_divergence",
    "mx_dequantize",
    "mx_quantize",
    "resolve_provider",
    "tensor_sha256",
]
