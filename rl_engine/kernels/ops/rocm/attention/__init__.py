# SPDX-License-Identifier: Apache-2.0
"""Compatibility import; the canonical module is loaded below."""

from importlib import import_module


def __getattr__(name):
    return getattr(import_module("rl_engine.backends.rocm.attention"), name)
