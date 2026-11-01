# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""M10 boundary semantics for the T02 / P7 consistency contract.

M10 is the selected-token raw log-probability boundary: the training side and the
rollout side must both score with the existing ``linear_logp`` on raw model log
probabilities. This package holds the versioned M10 input schema, the fail-closed
pre-checks that run before any scoring, and the stub fixture used while the real
providers (T05/T06) are not yet available.
"""

from rl_engine.validation.m10.precheck import M10PrecheckResult, evaluate_m10_precheck
from rl_engine.validation.m10.schema import M10_INPUT_SCHEMA_VERSION, M10Input, M10Status
from rl_engine.validation.m10.stub import StubM10Provider, build_m10_stub

__all__ = [
    "M10_INPUT_SCHEMA_VERSION",
    "M10Input",
    "M10PrecheckResult",
    "M10Status",
    "StubM10Provider",
    "build_m10_stub",
    "evaluate_m10_precheck",
]
