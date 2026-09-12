# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import importlib.util
import os
import sysconfig
import warnings
from distutils.errors import CompileError
from distutils.spawn import find_executable
from pathlib import Path

from setuptools import Extension, find_packages, setup


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
        ]
        if is_rocm:
            # ROCm-tuned WS2 vocab-parallel logprob kernels; the shared
            # deterministic_logp_kernel.cu keeps the SM90-tuned CUDA path.
            cuda_sources.extend(
                [
                    "csrc/hip/hip_deterministic_logp_kernel.hip",
                    "csrc/rocm/distributed/deterministic_collective.hip",
                ]
            )
        else:
            # CUDA IPC and the fixed-tree collective implementation are not
            # part of the ROCm extension.
            cuda_sources.append("csrc/cuda/distributed/deterministic_collective.cu")
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
        if is_rocm:
            for tile_knob in (
                "DETERMINISTIC_LOGP_TILE_BLOCK_SIZE",
                "DETERMINISTIC_LOGP_TILE_VECTOR_ELEMENTS",
                "DETERMINISTIC_LOGP_BACKWARD_BLOCK_SIZE",
            ):
                nvcc_flags.extend(_cuda_define_from_env(tile_knob, tile_knob))
        else:
            # Same idea for the CUDA tile-stats kernel: the defaults are tuned for
            # sm_90, and a different architecture or vocabulary split may prefer
            # another block size or vector width.
            for tile_knob in (
                "DETERMINISTIC_LOGP_TILE_BLOCK_SIZE_NARROW",
                "DETERMINISTIC_LOGP_TILE_BLOCK_SIZE_WIDE",
                "DETERMINISTIC_LOGP_TILE_VECTOR_BYTES",
            ):
                nvcc_flags.extend(_cuda_define_from_env(tile_knob, tile_knob))
        if not is_rocm and envs.env_flag(envs.KERNEL_ALIGN_NCU_LINEINFO):
            nvcc_flags.append("-lineinfo")
        if (
            not is_rocm
            and os.name == "nt"
            and envs.env_flag(envs.KERNEL_ALIGN_ALLOW_UNSUPPORTED_MSVC)
        ):
            nvcc_flags.append("-allow-unsupported-compiler")
            nvcc_flags.append("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")

        platform_define = "-DKERNEL_ALIGN_WITH_ROCM" if is_rocm else "-DKERNEL_ALIGN_WITH_CUDA"
        cxx_flags = ["-O3", "-std=c++17", platform_define]
        extra_link_args = list(torch_rpath)
        if os.name != "nt" and not is_rocm:
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

    extensions.extend(_ascend_extensions())

    if _native_extension_required() and not extensions:
        raise RuntimeError(
            "rl_engine._C was requested but no CUDA/ROCm build environment is available. "
            "Use a matching GPU-enabled PyTorch build; for a GPU-less ROCm build, set "
            "PYTORCH_ROCM_ARCH to the target architecture."
        )

    return extensions


def _ascend_extensions():
    """Ascend C (CANN) kernels, built with bisheng. Gated on KERNEL_ALIGN_FORCE_ASCEND=1.

    Follows the official torch_npu cpp_extension_asc pattern: .asc sources
    (kernel + host + pybind) are compiled by the CANN bisheng compiler into a
    single rl_engine._C_npu extension module. Requires CANN toolkit (bisheng on
    PATH or ASCEND_HOME_PATH set) and torch_npu.
    """
    if not envs.env_flag(envs.KERNEL_ALIGN_FORCE_ASCEND):
        return []
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "KERNEL_ALIGN_FORCE_ASCEND=1 requires torch and torch_npu to be installed"
        ) from e

    asc_srcs = sorted(str(p) for p in Path("csrc/ascend").glob("*.asc"))
    if not asc_srcs:
        raise RuntimeError("KERNEL_ALIGN_FORCE_ASCEND=1 but no .asc sources under csrc/ascend/")
    return [Extension(name="rl_engine._C_npu", sources=asc_srcs, language="asc")]


def _bisheng_compile_cmd(ext, ext_fullpath):
    """Single-command bisheng build for an Ascend C extension (see op-plugin example)."""
    import torch
    import torch.utils.cpp_extension as cpp_extension
    import torch_npu

    if find_executable("bisheng") is None:
        raise RuntimeError(
            "bisheng compiler not found on PATH; source the CANN toolkit environment first"
        )

    soc = os.environ.get(envs.KERNEL_ALIGN_ASCEND_ARCH, "dav-2201")  # A2/A3; A5: dav-3510
    abi_value = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
    module_name = ext.name.rsplit(".", 1)[-1]

    torch_npu_dir = os.path.dirname(os.path.realpath(torch_npu.__file__))
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")

    include_dirs = [
        *cpp_extension.include_paths(),
        sysconfig.get_config_var("INCLUDEPY"),
        os.path.join(torch_npu_dir, "include"),
        os.path.join(torch_npu_dir, "include", "third_party", "acl", "inc"),
        os.path.join(ascend_home, "include"),
    ]
    lib_dirs = [
        sysconfig.get_config_var("LIBDIR"),
        os.path.join(os.path.dirname(torch.__file__), "lib"),
        os.path.join(torch_npu_dir, "lib"),
        os.path.join(ascend_home, "lib64"),
    ]

    cmd = [
        "bisheng",
        "-x",
        "asc",
        f"--npu-arch={soc}",
        "-shared",
        "-fPIC",
        "-std=c++17",
        "-O2",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi_value}",
        f"-DTORCH_EXTENSION_NAME={module_name}",
        "-lascendcl",
        "-ltorch_npu",
        "-ltorch",
        "-ltorch_cpu",
        "-ltorch_python",
        "-lc10",
        *ext.sources,
        "-o",
        ext_fullpath,
    ]
    cmd += [f"-I{d}" for d in include_dirs if d]
    cmd += [f"-L{d}" for d in lib_dirs if d]
    return cmd


def get_cmdclass():
    _, BuildExtension, _ = _load_torch_extension_tools()
    if BuildExtension is None:
        return {}

    class AscendBuildExtension(BuildExtension):
        """torch BuildExtension + bisheng path for language="asc" extensions."""

        def build_extension(self, ext):
            if getattr(ext, "language", None) != "asc":
                super().build_extension(ext)
                return
            ext_fullpath = self.get_ext_fullpath(ext.name)
            os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)
            try:
                self.spawn(_bisheng_compile_cmd(ext, ext_fullpath))
            except Exception as e:
                raise CompileError(str(e)) from e

    return {"build_ext": AscendBuildExtension}


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
