# SPDX-License-Identifier: Apache-2.0
"""Compatibility import; the canonical module is loaded below."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("rl_engine.models.gemma.gemma4_workload")
