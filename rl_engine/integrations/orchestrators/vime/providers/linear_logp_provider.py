# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Compatibility entry; the canonical provider lives in the common layer.

The implementation moved to ``rl_engine.integrations.common.linear_logp_provider``
so that Megatron training no longer imports an orchestrator package.  This module
keeps the historical import path and every configuration string that names it
working by replacing itself with the canonical module object, which also keeps
``monkeypatch`` on this path acting on the real implementation.
"""

from __future__ import annotations

import importlib
import sys

from rl_engine.integrations.common.linear_logp_provider import (
    LinearLogpProviderUnavailable,
    LinearLogpResult,
    provider,
)

sys.modules[__name__] = importlib.import_module(
    "rl_engine.integrations.common.linear_logp_provider"
)

__all__ = ["LinearLogpResult", "LinearLogpProviderUnavailable", "provider"]
