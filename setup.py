# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import os
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    from torch.utils.cpp_extension import ROCMExtension
except ImportError:
    ROCMExtension = None


def _cuda_define_from_env(name: str, macro: str) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return []
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return [f"-D{macro}={parsed}"]


def get_extensions():
    extensions = []
    is_rocm = torch.version.hip is not None

    if is_rocm:
        extensions.append(
            ROCMExtension(
                name="rl_engine._C",
                sources=[
                    "csrc/ops.cpp",
                    "csrc/fused_logp_kernel.cpp",
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "hipcc": ["-O3", "--use_fast_math", "-Xhipcc", "-compress-all"],
                },
            )
        )
    elif torch.cuda.is_available():
        cuda_sources = [
            "csrc/ops.cpp",
            "csrc/fused_logp_kernel.cu",
        ]

        cc_major, _ = torch.cuda.get_device_capability()
        nvcc_flags = ["-O3", "--use_fast_math", "-Xfatbin", "-compress-all"]
        nvcc_flags.extend(
            _cuda_define_from_env(
                "FUSED_LOGP_TWOPASS_BLOCK_SIZE",
                "FUSED_LOGP_TWOPASS_BLOCK_SIZE",
            )
        )
        nvcc_flags.extend(
            _cuda_define_from_env(
                "FUSED_LOGP_ONLINE_BLOCK_SIZE",
                "FUSED_LOGP_ONLINE_BLOCK_SIZE",
            )
        )
        nvcc_flags.extend(
            _cuda_define_from_env(
                "FUSED_LOGP_ONLINE_SPARSE_LARGE_VOCAB_BLOCK_SIZE",
                "FUSED_LOGP_ONLINE_SPARSE_LARGE_VOCAB_BLOCK_SIZE",
            )
        )
        nvcc_flags.extend(
            _cuda_define_from_env(
                "FUSED_LOGP_ONLINE_LARGE_ROW_BYTES_THRESHOLD",
                "FUSED_LOGP_ONLINE_LARGE_ROW_BYTES_THRESHOLD",
            )
        )
        nvcc_flags.extend(
            _cuda_define_from_env(
                "FUSED_LOGP_ONLINE_SPARSE_DENSITY_NUMERATOR",
                "FUSED_LOGP_ONLINE_SPARSE_DENSITY_NUMERATOR",
            )
        )
        nvcc_flags.extend(
            _cuda_define_from_env(
                "FUSED_LOGP_ONLINE_SPARSE_DENSITY_DENOMINATOR",
                "FUSED_LOGP_ONLINE_SPARSE_DENSITY_DENOMINATOR",
            )
        )
        nvcc_flags.extend(
            _cuda_define_from_env(
                "FUSED_LOGP_ONLINE_MIN_BLOCKS_PER_SM",
                "FUSED_LOGP_ONLINE_MIN_BLOCKS_PER_SM",
            )
        )
        if os.environ.get("KERNEL_ALIGN_NCU_LINEINFO") == "1":
            nvcc_flags.append("-lineinfo")

        cxx_flags = ["-O3", "-std=c++17", "-DKERNEL_ALIGN_WITH_CUDA"]
        extra_link_args = []

        tma_src = "csrc/cuda/fused_logp_sm90.cu"
        enable_sm90 = cc_major >= 9 or os.environ.get("KERNEL_ALIGN_FORCE_SM90") == "1"
        if enable_sm90 and os.path.exists(tma_src):
            cuda_sources.append(tma_src)
            nvcc_flags.append("-gencode=arch=compute_90a,code=sm_90a")
            cxx_flags.append("-DKERNEL_ALIGN_WITH_SM90")
            extra_link_args.append("-lcuda")

        extensions.append(
            CUDAExtension(
                name="rl_engine._C",
                sources=cuda_sources,
                extra_compile_args={
                    "cxx": cxx_flags,
                    "nvcc": nvcc_flags,
                },
                extra_link_args=extra_link_args,
            )
        )
    return extensions


setup(
    name="rl-engine",
    version="0.1.0",
    packages=find_packages(include=["rl_engine", "rl_engine.*"]),
    install_requires=[
        "torch>=2.4.0",
        "tabulate",
        "numpy",
        "accelerate",
        "transformers",
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    extras_require={
        "cuda": ["flashinfer"],
        "rocm": ["aiter"],
    },
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)
