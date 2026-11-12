# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Test-only integration helpers for the alignment framework."""

from rl_engine.validation.fixtures.smoke_ops import (
    SMOKE_ONLY_LOGP_OFFSET_BACKEND_ID,
    SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID,
    register_smoke_operators,
    smoke_operator_descriptors,
)

__all__ = [
    "SMOKE_ONLY_LOGP_OFFSET_BACKEND_ID",
    "SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID",
    "register_smoke_operators",
    "smoke_operator_descriptors",
]
