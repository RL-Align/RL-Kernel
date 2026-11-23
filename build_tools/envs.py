# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Environment variable parsing helpers for build scripts.

This module is intentionally import-safe for setup.py: keep it free of torch or
other heavy runtime imports.
"""

import os
import shlex


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if not value or value.strip() == "":
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
RL_KERNEL_REQUIRE_EXT = "RL_KERNEL_REQUIRE_EXT"
RL_KERNEL_H3_STRICT = "RL_KERNEL_H3_STRICT"


def h3_strict_cuda_flags() -> list[str]:
    """Opt-in, cache-visible arithmetic profile for the H3 CUDA entry points.

    External nvcc flags are rejected in this profile: append flags and response
    files can silently override these options after build_tools/extensions.py constructs them.
    Other operators keep the existing build defaults when this opt-in is absent.
    """
    if not env_flag(RL_KERNEL_H3_STRICT):
        return []
    if env_flag(KERNEL_ALIGN_USE_FAST_MATH):
        raise ValueError("RL_KERNEL_H3_STRICT is incompatible with KERNEL_ALIGN_USE_FAST_MATH")
    for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        if shlex.split(os.environ.get(name, "")):
            raise ValueError(f"Unset {name} for the audited RL_KERNEL_H3_STRICT build")
    return [
        "-DRL_KERNEL_H3_STRICT=1",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    ]


KERNEL_ALIGN_FORCE_ASCEND = "KERNEL_ALIGN_FORCE_ASCEND"
KERNEL_ALIGN_ASCEND_ARCH = "KERNEL_ALIGN_ASCEND_ARCH"
