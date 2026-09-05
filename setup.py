# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import importlib.util
import os
import warnings
from pathlib import Path

from setuptools import find_packages, setup


def _load_envs_module():
    envs_path = Path(__file__).with_name("envs.py")
    spec = importlib.util.spec_from_file_location("_rl_kernel_envs", envs_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load environment helpers from {envs_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


envs = _load_envs_module()


def _load_torch_extension_tools():
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return None, None, None

    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # CUDAExtension is also the supported extension entry point for ROCm
    # PyTorch builds. BuildExtension dispatches .cu/.hip sources to hipcc when
    # torch.version.hip is set.
    return torch, BuildExtension, CUDAExtension


def _native_extension_required() -> bool:
    """Whether the caller explicitly requested a native extension build."""
    return (
        envs.env_flag(envs.RL_KERNEL_REQUIRE_EXT)
        or bool(os.environ.get("PYTORCH_ROCM_ARCH", "").strip())
        or bool(os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip())
        or envs.env_flag("FORCE_CUDA")
    )


def _cuda_define_from_env(name: str, macro: str) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return []
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return [f"-D{macro}={parsed}"]


_ROCM_UNSUPPORTED_NVCC_FLAG_PREFIXES = (
    "-Xfatbin",
    "-compress-all",
    "-gencode",
    "--generate-code",
    "--expt-",
    "-lineinfo",
    "-allow-unsupported-compiler",
    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
)
_ROCM_NVCC_FLAGS_WITH_SEPARATE_VALUE = {
    "-Xfatbin",
    "-gencode",
    "--generate-code",
}


def _filter_rocm_incompatible_nvcc_flags(flags: list[str]) -> list[str]:
    """Remove CUDA-only device compiler flags before BuildExtension calls hipcc."""
    filtered_flags = []
    skip_next = False
    for flag in flags:
        if skip_next:
            skip_next = False
            continue
        if flag in _ROCM_NVCC_FLAGS_WITH_SEPARATE_VALUE:
            skip_next = True
            continue
        if flag.startswith(_ROCM_UNSUPPORTED_NVCC_FLAG_PREFIXES):
            continue
        filtered_flags.append(flag)
    return filtered_flags


def get_extensions():
    torch, _, CUDAExtension = _load_torch_extension_tools()
    if torch is None:
        message = (
            "PyTorch is unavailable, so rl_engine._C cannot be built. Install a matching "
            "CUDA/ROCm PyTorch build first, then run "
            "`RL_KERNEL_REQUIRE_EXT=1 python -m pip install --no-build-isolation -e .`."
        )
        if _native_extension_required():
            raise RuntimeError(message)
        warnings.warn(
            f"{message} Continuing with the pure-Python fallback because no native extension "
            "was explicitly requested.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    extensions = []
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    torch_rpath = ["-Wl,-rpath,$ORIGIN/../torch/lib"]
    if os.environ.get("KERNEL_ALIGN_DEV_RPATH") == "1":
        torch_rpath.append(f"-Wl,-rpath,{torch_lib_dir}")
    is_rocm = getattr(torch.version, "hip", None) is not None

    # CUDAExtension is intentionally used for both CUDA and ROCm. On ROCm,
    # PyTorch's BuildExtension hipifies CUDA sources and invokes hipcc; it also
    # consumes PYTORCH_ROCM_ARCH (one or more ';'-separated gfx targets) to add
    # --offload-arch. Do not require a visible GPU when a ROCm target was
    # explicitly selected.
    no_rocm_arch = not os.environ.get("PYTORCH_ROCM_ARCH", "").strip()
    if is_rocm and no_rocm_arch and torch.cuda.device_count() == 0:
        raise RuntimeError(
            "ROCm builds without a visible GPU require PYTORCH_ROCM_ARCH. "
            "Set one or more ';'-separated targets, for example "
            "PYTORCH_ROCM_ARCH='gfx942;gfx950'."
        )

    if is_rocm or torch.cuda.is_available():
        cuda_sources = [
            "csrc/ops.cpp",
            "csrc/fused_logp_kernel.cu",
            "csrc/deterministic_logp_kernel.cu",
            "csrc/cuda/gemm/det_gemm_kernel.cu",
            "csrc/cuda/rmsnorm.cu",
            "csrc/cuda/activation.cu",
            "csrc/cuda/attention/deterministic_attention.cu",
            "csrc/cuda/distributed/deterministic_collective.cu",
        ]
        if not is_rocm:
            # This source contains NVIDIA PTX (cp.async, ldmatrix, and mma.sync).
            # The ROCm dispatcher falls back to PyTorch SDPA for this operator.
            cuda_sources.append("csrc/cuda/attention/prefix_shared_attention.cu")

        nvcc_flags = ["-O3", "-Xfatbin", "-compress-all"]
        if envs.env_flag(envs.KERNEL_ALIGN_USE_FAST_MATH):
            nvcc_flags.append("--use_fast_math")
        if not is_rocm:
            cc_major, cc_minor = torch.cuda.get_device_capability()
            enable_sm90 = os.environ.get("KERNEL_ALIGN_FORCE_SM90") == "1"
            if not enable_sm90:
                # SM90 build emits 90a below; mixing plain compute_90 breaks TMA ptxas.
                nvcc_flags.append(
                    f"-gencode=arch=compute_{cc_major}{cc_minor},code=sm_{cc_major}{cc_minor}"
                )
            nvcc_flags.append("--expt-relaxed-constexpr")
            nvcc_flags.append("--expt-extended-lambda")
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
        if not is_rocm and envs.env_flag(envs.KERNEL_ALIGN_NCU_LINEINFO):
            nvcc_flags.append("-lineinfo")
        if (
            not is_rocm
            and os.name == "nt"
            and envs.env_flag(envs.KERNEL_ALIGN_ALLOW_UNSUPPORTED_MSVC)
        ):
            nvcc_flags.append("-allow-unsupported-compiler")
            nvcc_flags.append("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")

        # Let PyTorch's BuildExtension choose the C++ standard required by the
        # installed PyTorch release (for example, C++20 in PyTorch 2.14).
        cxx_flags = ["-O3", "-DKERNEL_ALIGN_WITH_CUDA"]
        extra_link_args = list(torch_rpath)
        if os.name != "nt":
            # CUDA IPC metadata queries use the driver API (cuPointerGetAttribute).
            extra_link_args.append("-lcuda")

        if not is_rocm:
            sm90_srcs = [
                "csrc/cuda/fused_logp_sm90.cu",
                "csrc/cuda/fused_linear_logp_sm90.cu",  # TMA + WGMMA fused linear log-prob
                "csrc/cuda/batch_invariant_logp_kernel_sm90.cu",  # TMA batch-invariant logp
                "csrc/cuda/rope_sm90.cu",  # RoPE rotate-half apply, gated to SM90 build
                # Single-card batch-invariant embedding/lm-head.
                "csrc/cuda/embedding_lm_head_sm90.cu",
            ]
            enable_sm90 = envs.env_flag(envs.KERNEL_ALIGN_FORCE_SM90)
            present_sm90 = [s for s in sm90_srcs if os.path.exists(s)]
            if enable_sm90 and present_sm90:
                tma_arch = f"{cc_major}{cc_minor}a"  # WGMMA/TMA require the arch-native 'a' variant
                cuda_sources.extend(present_sm90)
                nvcc_flags.append(f"-gencode=arch=compute_{tma_arch},code=sm_{tma_arch}")
                cxx_flags.append("-DKERNEL_ALIGN_WITH_SM90")
                if "-lcuda" not in extra_link_args:
                    extra_link_args.append("-lcuda")

            # det_gemm SM90 (mma.sync + TMA) path: independent of the fused_logp
            # SM90 sources, which currently fail ptxas on CUDA 12.4 (shared::cta in
            # the shared tma_utils.cuh). det_gemm uses its own gemm/det_gemm_tma.cuh.
            enable_det_gemm_sm90 = os.environ.get("KERNEL_ALIGN_DET_GEMM_SM90") == "1"
            if enable_det_gemm_sm90:
                tma_arch = f"{cc_major}{cc_minor}a"
                arch_flag = f"-gencode=arch=compute_{tma_arch},code=sm_{tma_arch}"
                if arch_flag not in nvcc_flags:
                    nvcc_flags.append(arch_flag)
                if "-lcuda" not in extra_link_args:
                    extra_link_args.append("-lcuda")
                nvcc_flags.append("-DRL_KERNEL_ENABLE_SM90")
                cxx_flags.append("-DRL_KERNEL_ENABLE_SM90")

        if is_rocm:
            nvcc_flags = _filter_rocm_incompatible_nvcc_flags(nvcc_flags)

        extensions.append(
            CUDAExtension(
                name="rl_engine._C",
                sources=cuda_sources,
                include_dirs=[],
                extra_compile_args={
                    "cxx": cxx_flags,
                    "nvcc": nvcc_flags,
                },
                extra_link_args=extra_link_args,
            )
        )

    if _native_extension_required() and not extensions:
        raise RuntimeError(
            "rl_engine._C was requested but no CUDA/ROCm build environment is available. "
            "Use a matching GPU-enabled PyTorch build; for a GPU-less ROCm build, set "
            "PYTORCH_ROCM_ARCH to the target architecture."
        )

    return extensions


def get_cmdclass():
    _, BuildExtension, _ = _load_torch_extension_tools()
    if BuildExtension is None:
        return {}
    return {"build_ext": BuildExtension}


setup(
    name="rl-engine",
    version="0.1.0",
    packages=find_packages(include=["rl_engine", "rl_engine.*"]),
    install_requires=[
        "torch>=2.4.1",
        "tabulate",
        "numpy",
        "accelerate",
        "transformers==5.13.1",
    ],
    ext_modules=get_extensions(),
    cmdclass=get_cmdclass(),
    extras_require={
        "cuda": ["flashinfer"],
        "rocm": ["aiter"],
        "vllm": ["vllm>=0.6.0"],
        "drift-viewer": ["Pillow>=10", "PySide6>=6.6"],
    },
    entry_points={
        "console_scripts": [
            "rlk-drift-view=rl_engine.alignment.cross_config.drift_viewer:main",
        ],
    },
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)
