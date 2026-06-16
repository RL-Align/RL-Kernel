# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Opt-in Vime integration helpers."""

from rl_engine.integrations.vime.rollout_logp_probe import (
    ENV_ENABLED,
    ENV_STRICT,
    METADATA_KEY,
    RLKernelProbeResult,
    custom_generate,
    run_logp_probe,
)

__all__ = [
    "ENV_ENABLED",
    "ENV_STRICT",
    "METADATA_KEY",
    "RLKernelProbeResult",
    "custom_generate",
    "run_logp_probe",
]
