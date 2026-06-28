# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import importlib.util
import os


def _fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        _fail(f"PyTorch is not installed: {exc}")

    if torch.version.hip is None:
        _fail(f"PyTorch is not a ROCm build: torch={torch.__version__}")

    if not torch.cuda.is_available():
        _fail("ROCm GPU is not available to PyTorch")

    device_name = torch.cuda.get_device_name(0)
    triton_available = importlib.util.find_spec("triton") is not None
    flash_attn_func_available = False
    # flash-attn selects the ROCm CK/Triton backend at import time.
    os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
    try:
        from flash_attn import flash_attn_func
    except (ImportError, OSError, RuntimeError) as exc:
        flash_attn_status = f"not available ({exc})"
    else:
        flash_attn_func_available = flash_attn_func is not None
        flash_attn_status = "available" if flash_attn_func_available else "not available"

    print("backend availability:")
    print(
        "  ROCm PyTorch runtime: "
        f"available (torch={torch.__version__}, hip={torch.version.hip}, GPU={device_name})"
    )
    print("  PyTorch SDPA fallback: available")
    print(f"  Triton package: {'available' if triton_available else 'not available'}")
    print(f"  flash-attn AMD Triton: {flash_attn_status}")
    print("  ROCm CK: not selected by this checker")

    if not flash_attn_func_available:
        _fail("flash_attn AMD Triton backend is required but could not be imported")


if __name__ == "__main__":
    main()
