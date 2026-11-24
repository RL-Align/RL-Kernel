# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Pinned DeepGEMM dependency and its fail-closed capability check (DSv4, SM90).

DSv4's dense FP8 GEMMs (attention projections, shared expert) run on
unmodified DeepGEMM. Its batch invariance and determinism were established by
``scripts/probe_dsv4_deepgemm_bi.py`` for one commit, one JIT toolkit and SM90
(design D3), so every caller goes through :func:`require_deepgemm`, which
refuses to hand out the module unless:

* the GPU is SM90;
* ``deep_gemm`` is installed at the pinned commit (its package version carries
  the short commit: ``2.6.1+e21c821``);
* the ``nvcc`` DeepGEMM JIT-compiles with is CUDA 12.8 or newer (DeepGEMM picks
  ``DG_JIT_NVCC_COMPILER``, else ``$CUDA_HOME``/``$CUDA_PATH``, else ``nvcc`` on
  ``PATH``; older toolkits fail to compile its kernels);
* the entry points DSv4 uses are exported.

Install the pinned build with the ``dsv4`` extra (``CUDA_HOME`` must point at
CUDA 12.8+ during the build and at runtime)::

    CUDA_HOME=/path/to/cuda-12.8 pip install --no-build-isolation -e ".[dsv4]"

``RL_KERNEL_DEEPGEMM_ALLOW_UNPINNED=1`` accepts another DeepGEMM build for
development only; re-run the probe before trusting its results.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType

import torch

DEEPGEMM_REPO = "https://github.com/vllm-project/DeepGEMM.git"
DEEPGEMM_COMMIT = "e21c821f39a2056d68067a466c64ddc942200106"
DEEPGEMM_VERSION = f"2.6.1+{DEEPGEMM_COMMIT[:7]}"
MIN_NVCC = (12, 8)
ALLOW_UNPINNED_ENV = "RL_KERNEL_DEEPGEMM_ALLOW_UNPINNED"
REQUIRED_SYMBOLS = (
    "fp8_gemm_nt",
    "get_mn_major_tma_aligned_tensor",
    "get_num_sms",
)


class DeepGemmUnavailable(RuntimeError):
    """DeepGEMM cannot be used for the DSv4 batch-invariant contract."""


@dataclass(frozen=True)
class DeepGemmCapability:
    version: str
    pinned: bool
    nvcc: str
    nvcc_version: tuple[int, int]
    device_capability: tuple[int, int]
    num_sms: int


def jit_nvcc_path() -> str | None:
    """The ``nvcc`` DeepGEMM's JIT will use, resolved the way DeepGEMM does."""
    explicit = os.environ.get("DG_JIT_NVCC_COMPILER")
    if explicit:
        return explicit
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        return os.path.join(cuda_home, "bin", "nvcc")
    return shutil.which("nvcc")


def nvcc_version(path: str) -> tuple[int, int]:
    output = subprocess.run(
        [path, "--version"], capture_output=True, text=True, check=True, timeout=60
    ).stdout
    match = re.search(r"release (\d+)\.(\d+)", output)
    if match is None:
        raise DeepGemmUnavailable(f"cannot parse the nvcc version from {path}")
    return int(match.group(1)), int(match.group(2))


def _allow_unpinned() -> bool:
    return os.environ.get(ALLOW_UNPINNED_ENV, "0").strip().lower() in {"1", "true", "yes"}


def _check_device(device: torch.device) -> tuple[int, int]:
    if torch.version.hip is not None or not torch.cuda.is_available():
        raise DeepGemmUnavailable("DeepGEMM requires NVIDIA CUDA")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 9:
        raise DeepGemmUnavailable(
            f"the DSv4 DeepGEMM contract is validated on SM90 only, got SM{capability[0]}"
            f"{capability[1]}"
        )
    return capability


def _check_version() -> tuple[str, bool]:
    try:
        version = importlib.metadata.version("deep_gemm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise DeepGemmUnavailable(
            "deep_gemm is not installed; install the pinned build with "
            '`pip install --no-build-isolation -e ".[dsv4]"`'
        ) from exc
    pinned = version == DEEPGEMM_VERSION
    if not pinned and not _allow_unpinned():
        raise DeepGemmUnavailable(
            f"deep_gemm {version} is installed, but DSv4 is validated against {DEEPGEMM_VERSION} "
            f"({DEEPGEMM_REPO}@{DEEPGEMM_COMMIT}); set {ALLOW_UNPINNED_ENV}=1 only for development"
        )
    return version, pinned


def _check_nvcc() -> tuple[str, tuple[int, int]]:
    path = jit_nvcc_path()
    if path is None or not os.path.exists(path):
        raise DeepGemmUnavailable(
            "no nvcc for DeepGEMM's JIT; set CUDA_HOME to a CUDA 12.8+ toolkit"
        )
    try:
        version = nvcc_version(path)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DeepGemmUnavailable(f"cannot run {path} --version") from exc
    if version < MIN_NVCC:
        raise DeepGemmUnavailable(
            f"DeepGEMM's JIT would use nvcc {version[0]}.{version[1]} at {path}; "
            f"CUDA {MIN_NVCC[0]}.{MIN_NVCC[1]}+ is required (set CUDA_HOME or DG_JIT_NVCC_COMPILER)"
        )
    return path, version


@lru_cache(maxsize=None)
def _require(device_index: int) -> tuple[ModuleType, DeepGemmCapability]:
    capability = _check_device(torch.device("cuda", device_index))
    version, pinned = _check_version()
    nvcc, nvcc_ver = _check_nvcc()
    try:
        module = importlib.import_module("deep_gemm")
    except ImportError as exc:
        raise DeepGemmUnavailable(f"deep_gemm failed to import: {exc}") from exc
    missing = [name for name in REQUIRED_SYMBOLS if not hasattr(module, name)]
    if missing:
        raise DeepGemmUnavailable(f"deep_gemm lacks {missing}")
    return module, DeepGemmCapability(
        version=version,
        pinned=pinned,
        nvcc=nvcc,
        nvcc_version=nvcc_ver,
        device_capability=capability,
        num_sms=int(module.get_num_sms()),
    )


def require_deepgemm(
    device: torch.device | int | None = None,
) -> tuple[ModuleType, DeepGemmCapability]:
    """Return ``(deep_gemm, capability)`` or raise :class:`DeepGemmUnavailable`."""
    if device is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    elif isinstance(device, int):
        index = device
    else:
        index = device.index if device.index is not None else torch.cuda.current_device()
    return _require(index)


def deepgemm_available(device: torch.device | int | None = None) -> bool:
    try:
        require_deepgemm(device)
    except DeepGemmUnavailable:
        return False
    return True
