# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P1 start kit: mHC + RMSNorm deterministic forward/backward contracts (issue #2)."""

from rl_engine.mhc.contract import (
    COMB_SLICE,
    FUSION_MODES,
    MHC_EPS,
    ORACLE_PROFILE,
    POST_SLICE,
    PRE_SLICE,
    PROD_CONTROLLER_N,
    PROD_HIDDEN,
    RMSNORM_EPS,
    SCHEMA_VERSION,
    SINKHORN_ITERS,
    TRAINABILITY_MODES,
    ControllerParams,
    GradBoundary,
    LayerContract,
    NormParams,
    ResidualBatch,
    tensor_sha256,
)
from rl_engine.mhc.provider import (
    MHCProvider,
    ReferenceProvider,
    StubProvider,
    check_capability,
    resolve_provider,
)
from rl_engine.mhc.reduction import (
    HC_MULT,
    STREAM4_TREE,
    fixed_dot,
    fixed_sum,
    fixed_sumsq,
    stream4_sum,
)
from rl_engine.mhc.trace import MHCTrace, first_divergence

__all__ = [
    "COMB_SLICE",
    "FUSION_MODES",
    "HC_MULT",
    "MHC_EPS",
    "ORACLE_PROFILE",
    "POST_SLICE",
    "PRE_SLICE",
    "PROD_CONTROLLER_N",
    "PROD_HIDDEN",
    "RMSNORM_EPS",
    "SCHEMA_VERSION",
    "SINKHORN_ITERS",
    "STREAM4_TREE",
    "TRAINABILITY_MODES",
    "ControllerParams",
    "GradBoundary",
    "LayerContract",
    "MHCProvider",
    "MHCTrace",
    "NormParams",
    "ReferenceProvider",
    "ResidualBatch",
    "StubProvider",
    "check_capability",
    "first_divergence",
    "fixed_dot",
    "fixed_sum",
    "fixed_sumsq",
    "resolve_provider",
    "stream4_sum",
    "tensor_sha256",
]
