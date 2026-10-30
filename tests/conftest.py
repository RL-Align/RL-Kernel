# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import os
import pathlib
import sys

import pytest


def _add_windows_dll_dirs():
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    try:
        import torch
    except ImportError:
        return

    candidate_dirs = [pathlib.Path(torch.__file__).parent / "lib"]

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidate_dirs.append(pathlib.Path(cuda_path) / "bin")

    for path in candidate_dirs:
        if path.exists():
            os.add_dll_directory(str(path))


_add_windows_dll_dirs()

# P7 helpers (p7_doubles, megatron_training_*.py) are imported as top-level
# modules and also run as standalone scripts; appending keeps them from
# shadowing anything already importable.
_TESTS_DIR = str(pathlib.Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.append(_TESTS_DIR)


def _is_rocm() -> bool:
    """Whether this interpreter is running a ROCm PyTorch build.

    ``torch.cuda`` is the device API on ROCm too, so ``cuda.is_available()`` and
    ``device_count()`` cannot distinguish the platforms; ``torch.version.hip``
    can.
    """

    try:
        import torch
    except ImportError:
        return False
    return torch.version.hip is not None


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cuda_only: test depends on CUDA-exclusive functionality and is skipped on ROCm",
    )


def pytest_collection_modifyitems(config, items):
    """Skip CUDA-exclusive tests on ROCm instead of failing them.

    Some kernels are compiled out of ROCm builds on purpose (the CUDA-IPC
    deterministic collectives, for one), so their tests cannot pass there.
    Because ROCm reports GPUs through the CUDA device API, a
    ``cuda.device_count()`` guard does not exclude them and they fail instead of
    skipping - which makes a ROCm run look broken rather than out of scope.
    """

    if not _is_rocm():
        return
    skip_rocm = pytest.mark.skip(
        reason="CUDA-exclusive functionality; not built for ROCm (torch.version.hip is set)"
    )
    for item in items:
        if "cuda_only" in item.keywords:
            item.add_marker(skip_rocm)
