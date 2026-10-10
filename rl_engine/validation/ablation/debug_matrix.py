# SPDX-License-Identifier: Apache-2.0
"""Compatibility import."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("rl_engine.contracts.diagnostics.modules")
