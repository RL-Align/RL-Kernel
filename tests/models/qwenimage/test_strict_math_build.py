# SPDX-License-Identifier: Apache-2.0
"""Build the real kernel source with and without --use_fast_math and ask it which one it is."""

import shutil
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or shutil.which("nvcc") is None,
    reason="CUDA and nvcc required",
)

ROOT = Path(__file__).resolve().parents[3]
BINDING = """
#include <torch/extension.h>
bool qwenimage_strict_math_compiled();
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("strict", &qwenimage_strict_math_compiled); }
"""


def build(name, flags, tmp_path):
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    (tmp_path / name).mkdir()
    binding = tmp_path / f"{name}.cpp"
    binding.write_text(BINDING)
    return load(
        name=name,
        sources=[str(binding), str(ROOT / "csrc/cuda/model_specific/qwenimage/det_linear.cu")],
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]
        + flags,
        build_directory=str(tmp_path / name),
    )


@pytest.mark.parametrize(
    ("flags", "strict"), [([], True), (["--use_fast_math"], False)], ids=["default", "fast_math"]
)
def test_probe_reports_the_actual_build_flags(flags, strict, tmp_path):
    name = "qwenimage_math_" + ("default" if strict else "fast_math")
    assert build(name, flags, tmp_path).strict() is strict
