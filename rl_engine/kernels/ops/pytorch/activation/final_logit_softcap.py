# SPDX-License-Identifier: Apache-2.0
"""Compatibility import; the canonical module is loaded below."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("rl_engine.reference.activation.final_logit_softcap")
