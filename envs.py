# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Environment variable parsing helpers for build scripts.

This module is intentionally import-safe for setup.py: keep it free of torch or
other heavy runtime imports.
"""

import os


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


KERNEL_ALIGN_USE_FAST_MATH = "KERNEL_ALIGN_USE_FAST_MATH"
KERNEL_ALIGN_NCU_LINEINFO = "KERNEL_ALIGN_NCU_LINEINFO"
KERNEL_ALIGN_ALLOW_UNSUPPORTED_MSVC = "KERNEL_ALIGN_ALLOW_UNSUPPORTED_MSVC"
KERNEL_ALIGN_FORCE_SM90 = "KERNEL_ALIGN_FORCE_SM90"
