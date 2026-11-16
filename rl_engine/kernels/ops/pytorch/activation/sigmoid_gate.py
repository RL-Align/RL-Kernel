# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the canonical reference module."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("rl_engine.reference.activation.sigmoid_gate")
