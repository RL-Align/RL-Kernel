# SPDX-License-Identifier: Apache-2.0
"""Compatibility import; the canonical module is loaded below."""

import importlib.util
from importlib import import_module

_TARGET = "rl_engine.backends.shared.triton.norm"


def __getattr__(name):
    """Resolve against the shared Triton norm package, then its operator module.

    Two steps, where the sibling packages under ``kernels/ops/triton/`` need one,
    because ``rl_engine.backends.shared.triton.norm`` re-exports nothing from its
    ``__init__`` while ``...triton.activation`` re-exports its ops. A single
    ``getattr`` against this package therefore raises for every name. ``find_spec``
    rather than catching ``ModuleNotFoundError`` so that an import error raised
    *inside* a submodule is not mistaken for the submodule being absent.
    """
    if importlib.util.find_spec(f"{_TARGET}.{name}") is not None:
        return import_module(f"{_TARGET}.{name}")
    return getattr(import_module(f"{_TARGET}.fused_add_norm"), name)
