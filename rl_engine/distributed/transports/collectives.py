# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compatibility imports for the unified deterministic collectives.

The implementation now lives in :mod:`rl_engine.distributed.algorithms.collectives`.
This module remains as a stable import path for older benchmark and integration
callers; it contains no separate collective implementation.
"""

from rl_engine.distributed.algorithms.collectives import (
    RCCLDeterministicCollective,
    TorchDistributedDeterministicCollective,
    create_deterministic_collective,
)

__all__ = [
    "RCCLDeterministicCollective",
    "TorchDistributedDeterministicCollective",
    "create_deterministic_collective",
]
