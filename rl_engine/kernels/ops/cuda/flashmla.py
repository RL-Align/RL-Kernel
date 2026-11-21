# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Pinned FlashMLA dependency and its fail-closed capability check (DSv4, SM90).

DSv4's sparse attention on Hopper runs FlashMLA's ``flash_mla_sparse_fwd``
(the SM90 sparse prefill kernel) for both prefill and decode, as vLLM does
with ``VLLM_DS4_DECODE_KERNEL=sparse`` under ``VLLM_BATCH_INVARIANT=1``.

vLLM (``ISEEKYAN/vllm@1a6542c``) pins ``vllm-project/FlashMLA@a8f794d1``, which
builds only against torch's stable ABI (torch >= 2.10; the vLLM fork's own
environment is torch 2.13 + CUDA 13.0, where we build exactly that commit). On
older torch we build ``b70aff3`` instead: the five commits in between change
only the Python / torch binding layer and SM100 code, and ``csrc/sm90`` and
``csrc/smxx`` are byte-identical, so the SM90 kernels are the pinned ones.
Install (SM90 only; ``CUDA_HOME`` pointing at CUDA 12.8+)::

    git -C FlashMLA checkout a8f794d1   # or b70aff3 on torch < 2.10
    git -C FlashMLA submodule update --init --recursive
    FLASH_MLA_DISABLE_SM100=1 pip install --no-build-isolation --no-deps ./FlashMLA

:func:`require_flashmla` refuses to hand out the module unless the GPU is SM90,
the installed build is one of those commits (package version ``1.0.0+a8f794d``
or ``1.0.0+b70aff3``) and the sparse entry point exists.
``RL_KERNEL_FLASHMLA_ALLOW_UNPINNED=1`` accepts another build for development
only; re-run ``scripts/probe_dsv4_flashmla_bi.py`` first.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType

import torch

FLASHMLA_REPO = "https://github.com/vllm-project/FlashMLA.git"
FLASHMLA_COMMIT = "a8f794d"  # vLLM's pin (torch >= 2.10)
FLASHMLA_SM90_EQUIVALENT = "b70aff3"  # same csrc/sm90 and csrc/smxx, builds on older torch
FLASHMLA_VERSIONS = (f"1.0.0+{FLASHMLA_COMMIT}", f"1.0.0+{FLASHMLA_SM90_EQUIVALENT}")
ALLOW_UNPINNED_ENV = "RL_KERNEL_FLASHMLA_ALLOW_UNPINNED"
REQUIRED_SYMBOLS = ("flash_mla_sparse_fwd",)


def _import_interface() -> ModuleType:
    """``flash_mla.flash_mla_interface``, skipping a broken package ``__init__``.

    At ``b70aff3`` the package ``__init__`` re-exports ``flash_mla_with_kvcache_fp8``,
    which the interface does not define, so ``import flash_mla`` fails; the
    interface and the compiled ``flash_mla.cuda`` extension are fine (vLLM
    vendors only those two). Load them under a bare package entry instead.
    """
    try:
        return importlib.import_module("flash_mla.flash_mla_interface")
    except ImportError:
        spec = importlib.util.find_spec("flash_mla")
        if spec is None or not spec.submodule_search_locations:
            raise
        for name in [m for m in sys.modules if m == "flash_mla" or m.startswith("flash_mla.")]:
            del sys.modules[name]
        package = ModuleType("flash_mla")
        package.__path__ = list(spec.submodule_search_locations)
        sys.modules["flash_mla"] = package
        return importlib.import_module("flash_mla.flash_mla_interface")


class FlashMlaUnavailable(RuntimeError):
    """FlashMLA cannot be used for the DSv4 batch-invariant contract."""


@dataclass(frozen=True)
class FlashMlaCapability:
    version: str
    pinned: bool
    device_capability: tuple[int, int]


def _allow_unpinned() -> bool:
    return os.environ.get(ALLOW_UNPINNED_ENV, "0").strip().lower() in {"1", "true", "yes"}


@lru_cache(maxsize=None)
def _require(device_index: int) -> tuple[ModuleType, FlashMlaCapability]:
    if torch.version.hip is not None or not torch.cuda.is_available():
        raise FlashMlaUnavailable("FlashMLA requires NVIDIA CUDA")
    capability = torch.cuda.get_device_capability(device_index)
    if capability[0] != 9:
        raise FlashMlaUnavailable(
            f"the DSv4 FlashMLA contract is validated on SM90 only, got SM{capability[0]}"
            f"{capability[1]}"
        )
    try:
        version = importlib.metadata.version("flash_mla")
    except importlib.metadata.PackageNotFoundError as exc:
        raise FlashMlaUnavailable(
            f"flash_mla is not installed; build {FLASHMLA_REPO}@{FLASHMLA_COMMIT} "
            f"(or {FLASHMLA_SM90_EQUIVALENT} on torch < 2.10) "
            "with FLASH_MLA_DISABLE_SM100=1"
        ) from exc
    pinned = version in FLASHMLA_VERSIONS
    if not pinned and not _allow_unpinned():
        raise FlashMlaUnavailable(
            f"flash_mla {version} is installed, but DSv4 is validated against {FLASHMLA_VERSIONS}; "
            f"set {ALLOW_UNPINNED_ENV}=1 only for development"
        )
    try:
        module = _import_interface()
    except ImportError as exc:
        raise FlashMlaUnavailable(f"flash_mla failed to import: {exc}") from exc
    missing = [name for name in REQUIRED_SYMBOLS if not hasattr(module, name)]
    if missing:
        raise FlashMlaUnavailable(f"flash_mla lacks {missing}")
    return module, FlashMlaCapability(version, pinned, capability)


def require_flashmla(
    device: torch.device | int | None = None,
) -> tuple[ModuleType, FlashMlaCapability]:
    """Return ``(flash_mla interface, capability)`` or raise :class:`FlashMlaUnavailable`."""
    if device is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    elif isinstance(device, int):
        index = device
    else:
        index = device.index if device.index is not None else torch.cuda.current_device()
    return _require(index)


def flashmla_available(device: torch.device | int | None = None) -> bool:
    try:
        require_flashmla(device)
    except FlashMlaUnavailable:
        return False
    return True
