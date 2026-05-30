# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

import os
import pathlib
import sys


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
