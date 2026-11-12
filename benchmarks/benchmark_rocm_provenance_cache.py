# SPDX-License-Identifier: Apache-2.0
"""Compatibility import; the canonical module is loaded below."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    import runpy

    runpy.run_module(
        "benchmarks.backends.rocm.benchmark_rocm_provenance_cache", run_name="__main__"
    )
else:
    sys.modules[__name__] = importlib.import_module(
        "benchmarks.backends.rocm.benchmark_rocm_provenance_cache"
    )
