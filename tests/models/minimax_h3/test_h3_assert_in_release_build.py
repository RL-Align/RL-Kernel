# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The H3 kernels' CUDA_KERNEL_ASSERT bounds checks survive a release (-DNDEBUG) build.

The native entry points check caller-built metadata and index values with ``CUDA_KERNEL_ASSERT``
on the device (a host-side check would add a synchronisation to every call). Unlike ``assert()``,
c10 keeps that macro under ``NDEBUG`` on CUDA, so a release build still stops on a bad value
instead of reading out of bounds. This compiles a one-kernel program with the release flags and
the torch headers this repository builds against, and checks that the assertion fires. The tests
that run the built extension on bad metadata and bad indices live next to each op
(``test_native_bounds_assertions``, ``test_the_backward_stops_on_metadata...``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
import torch
from torch.utils import cpp_extension

SOURCE = textwrap.dedent(
    """
    #include <c10/macros/Macros.h>
    #include <cstdio>
    #include <cstdlib>

    __global__ void check(const long* v, int n) {
      const int i = threadIdx.x;
      if (i < n) CUDA_KERNEL_ASSERT(v[i] >= 0 && v[i] < 100);
    }

    int main(int argc, char** argv) {
      long host[4] = {0, 1, 2, 3};
      if (argc > 1) host[2] = std::atol(argv[1]);
      long* device = nullptr;
      cudaMalloc(&device, sizeof(host));
      cudaMemcpy(device, host, sizeof(host), cudaMemcpyHostToDevice);
      check<<<1, 4>>>(device, 4);
      const cudaError_t status = cudaDeviceSynchronize();
      std::printf("%s\\n", cudaGetErrorString(status));
      return status == cudaSuccess ? 0 : 2;
    }
    """
)


def _nvcc() -> str | None:
    for root in (os.environ.get("CUDA_HOME"), "/usr/local/cuda"):
        if root and (Path(root) / "bin" / "nvcc").exists():
            return str(Path(root) / "bin" / "nvcc")
    return shutil.which("nvcc")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.skipif(_nvcc() is None, reason="needs nvcc")
def test_cuda_kernel_assert_stays_active_with_ndebug(tmp_path):
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    (tmp_path / "check.cu").write_text(SOURCE)
    build = subprocess.run(
        [
            _nvcc(),
            "-O3",
            "-DNDEBUG",
            "-std=c++17",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            *(f"-I{path}" for path in cpp_extension.include_paths()),
            str(tmp_path / "check.cu"),
            "-o",
            str(tmp_path / "check"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert build.returncode == 0, build.stderr[-1500:]

    ok = subprocess.run([str(tmp_path / "check"), "2"], capture_output=True, text=True, timeout=60)
    assert ok.returncode == 0, ok.stdout + ok.stderr  # the control: a valid value passes
    for bad in ("-1", "100", "123456"):
        done = subprocess.run(
            [str(tmp_path / "check"), bad], capture_output=True, text=True, timeout=60
        )
        assert done.returncode != 0, bad
        assert "device-side assert triggered" in done.stdout, (bad, done.stdout, done.stderr)
