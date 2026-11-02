# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for :mod:`build_tools.envs`."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

if __name__ == "__main__":
    import runpy

    runpy.run_module("build_tools.envs", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("build_tools.envs")
