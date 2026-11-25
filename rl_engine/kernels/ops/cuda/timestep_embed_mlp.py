# SPDX-License-Identifier: Apache-2.0
"""Compatibility import; the canonical module is loaded below."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "rl_engine.backends.cuda.embedding.timestep_embed_mlp"
)
