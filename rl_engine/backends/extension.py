# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from typing import Any

import torch  # noqa: F401  # Load torch shared libraries before importing the binary extension.

from rl_engine.utils.logger import logger

_C: Any = None

try:
    from rl_engine import _C

    _EXT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Core binary extension (_C) unavailable: {e}. Falling back to native code.")
    _EXT_AVAILABLE = False
    _C = None
