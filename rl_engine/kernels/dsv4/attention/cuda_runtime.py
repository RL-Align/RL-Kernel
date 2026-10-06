# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Reuse validated native T06 kernels or refresh them through JIT."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[4]
_MERGED = None
_ATTENTION_EXPORTS = (
    "mqa_joint_attention_sink_forward",
    "mqa_joint_attention_sink_forward_into",
    "mqa_joint_attention_sink_backward",
    "mqa_joint_attention_sink_workspace_validation_version",
)


def resolve_cuda_home() -> str:
    from torch.utils import cpp_extension

    candidates = [
        os.environ.get("T06_CUDA_HOME"),
        os.environ.get("CUDA_HOME"),
        cpp_extension.CUDA_HOME,
    ]
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(str(Path(nvcc).resolve().parent.parent))
    candidates.append("/usr/local/cuda")
    candidates.extend(str(home) for home in sorted(Path("/usr/local").glob("cuda-*"), reverse=True))
    for home in candidates:
        if home and (Path(home) / "bin" / "nvcc").is_file():
            return home
    raise RuntimeError("no CUDA toolkit with nvcc found")


def _prepare_env() -> None:
    import torch

    cuda_home = resolve_cuda_home()
    os.environ["CUDA_HOME"] = cuda_home
    from torch.utils import cpp_extension

    cpp_extension.CUDA_HOME = cuda_home
    os.environ["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        str(Path(torch.__file__).parent / "lib")
        + os.pathsep
        + str(Path(cuda_home) / "lib64")
        + os.pathsep
        + os.environ.get("LD_LIBRARY_PATH", "")
    )
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


def _load(name: str, sources: list[str], extra_include_paths: list[str] | None = None):
    from torch.utils.cpp_extension import load

    cuda_flags = ["-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda"]
    cxx = os.environ.get("CXX")
    # cpp_extension already adds -ccbin for an explicit CC.
    if cxx and not os.environ.get("CC"):
        cuda_flags.append(f"-ccbin={cxx}")
    return load(
        name=name,
        sources=sources,
        extra_include_paths=extra_include_paths or [],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def _merge(target, *modules):
    merged = target if target is not None else SimpleNamespace()
    for mod in modules:
        for key in dir(mod):
            if not key.startswith("_"):
                setattr(merged, key, getattr(mod, key))
    return merged


def ensure_native_kernels() -> str:
    """Load validated T06 kernels while retaining existing native exports."""

    global _MERGED
    import torch

    import rl_engine.kernels.ops.base as base
    from rl_engine.kernels.dsv4.attention import mqa_joint_attention_sink as mqa

    native_has_gemm = (
        base._EXT_AVAILABLE
        and base._C is not None
        and all(
            hasattr(base._C, name)
            for name in (
                "det_gemm_fwd_rhs_transposed",
                "det_gemm_fwd",
                "det_gemm_db_transposed",
            )
        )
    )
    native_ok = (
        native_has_gemm
        and all(hasattr(base._C, name) for name in _ATTENTION_EXPORTS)
        and base._C.mqa_joint_attention_sink_workspace_validation_version == 3
    )
    if native_ok:
        mqa._C = base._C
        mqa._EXT_AVAILABLE = True
        return "jit" if base._C is _MERGED else "native"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    if _MERGED is not None:
        base._C = _MERGED
        base._EXT_AVAILABLE = True
        mqa._C = _MERGED
        mqa._EXT_AVAILABLE = True
        return "jit"

    _prepare_env()
    attn = _load(
        "mqa_t06_verify",
        [
            str(_ROOT / "csrc/cuda/attention/mqa_joint_attention_sink.cu"),
            str(_ROOT / "csrc/cuda/attention/mqa_joint_attention_sink_jitbind.cpp"),
        ],
    )
    if native_has_gemm:
        merged = _merge(base._C, attn)
    else:
        gemm = _load(
            "det_gemm_t06_verify",
            [
                str(_ROOT / "csrc/cuda/gemm/det_gemm_kernel.cu"),
                str(_ROOT / "csrc/cuda/gemm/det_gemm_jitbind.cpp"),
            ],
            extra_include_paths=[str(_ROOT / "csrc/cuda/gemm")],
        )
        merged = _merge(base._C, attn, gemm)
    _MERGED = merged
    base._C = merged
    base._EXT_AVAILABLE = True
    mqa._C = merged
    mqa._EXT_AVAILABLE = True
    return "jit"


def ensure_t06_cuda_kernel() -> str:
    return ensure_native_kernels()
