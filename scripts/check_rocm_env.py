# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def _rocm_major_minor(version: str | None) -> tuple[str, str] | None:
    if not version:
        return None
    match = re.search(r"([0-9]+)\.([0-9]+)", version)
    if match is None:
        return None
    return match.group(1), match.group(2)


def _find_hipcc() -> str | None:
    for env_name in ("ROCM_HOME", "HIP_PATH"):
        env_path = os.environ.get(env_name)
        if env_path:
            hipcc = Path(env_path) / "bin" / "hipcc"
            if hipcc.exists():
                return str(hipcc)

    hipcc = shutil.which("hipcc")
    if hipcc:
        return hipcc

    fallback = Path("/opt/rocm/bin/hipcc")
    if fallback.exists():
        return str(fallback)

    return None


def _flash_attn_backend() -> str | None:
    if importlib.util.find_spec("flash_attn") is None:
        return None
    if os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE", "").upper() == "TRUE":
        return "triton"
    if importlib.util.find_spec("flash_attn_2_cuda") is not None:
        return "ck"
    return "triton-available-if-enabled"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local ROCm PyTorch environment.")
    parser.add_argument(
        "--require-flash-attn",
        action="store_true",
        help="fail if flash_attn_func cannot be imported",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import torch
    except ImportError as exc:
        _fail(f"PyTorch is not installed in {sys.executable}: {exc}")

    if torch.version.hip is None:
        _fail(f"PyTorch is not a ROCm build: torch={torch.__version__}")

    if not torch.cuda.is_available():
        _fail("ROCm GPU is not available to PyTorch")

    hipcc = _find_hipcc()
    if hipcc is None:
        _fail("Could not find hipcc from ROCM_HOME, HIP_PATH, PATH, or /opt/rocm/bin/hipcc")

    try:
        hipcc_output = subprocess.check_output([hipcc, "--version"], text=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        _fail(f"Could not run {hipcc} --version: {exc}")

    torch_rocm = _rocm_major_minor(torch.version.hip)
    hipcc_rocm = _rocm_major_minor(hipcc_output)
    if torch_rocm is None:
        _fail(f"Could not parse torch.version.hip={torch.version.hip!r}")
    if hipcc_rocm is None:
        _fail(f"Could not parse {hipcc} --version output")
    if torch_rocm != hipcc_rocm:
        _fail(
            "ROCm version mismatch: "
            f"torch.version.hip={torch.version.hip}, hipcc major/minor={'.'.join(hipcc_rocm)}"
        )

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    triton_available = importlib.util.find_spec("triton") is not None
    flash_attn_backend = _flash_attn_backend()
    flash_attn_func_available = False
    flash_attn_error = None
    if flash_attn_backend is not None:
        if flash_attn_backend == "triton-available-if-enabled":
            os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
        try:
            from flash_attn import flash_attn_func
        except (ImportError, OSError, RuntimeError) as exc:
            flash_attn_error = str(exc)
        else:
            flash_attn_func_available = flash_attn_func is not None

    print(f"Python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"torch.version.hip: {torch.version.hip}")
    print(f"hipcc: {hipcc}")
    print(f"hipcc ROCm: {'.'.join(hipcc_rocm)}")
    print(f"GPU: {device_name}")
    print(f"compute capability: {capability}")
    print(f"triton: {'available' if triton_available else 'missing'}")
    print(f"flash_attn backend: {flash_attn_backend or 'not installed'}")
    print(f"flash_attn_func: {'available' if flash_attn_func_available else 'not available'}")

    if flash_attn_error:
        print(f"flash_attn import error: {flash_attn_error}")

    if args.require_flash_attn and not flash_attn_func_available:
        _fail("flash_attn_func is required but could not be imported")


if __name__ == "__main__":
    main()
