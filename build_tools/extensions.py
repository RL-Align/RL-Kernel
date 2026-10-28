# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import importlib.util
import os
import shlex
import stat
import sysconfig
import warnings
from distutils.errors import CompileError
from distutils.spawn import find_executable
from pathlib import Path

from setuptools import Extension
from setuptools.command.build_py import build_py as _BuildPy

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# MiniMax-H3 (RFC #420) kernels promise bitwise parity with diffusers' eager
# FP32 math. --use_fast_math swaps expf/sinf/cosf/rsqrtf and division for
# approximations and flushes denormals, so these sources are left out of
# KERNEL_ALIGN_USE_FAST_MATH=1 builds; their bindings are gated on
# KERNEL_ALIGN_WITH_H3_KERNELS and dispatch falls back to the PyTorch reference.
H3_BITWISE_SOURCES = (
    "csrc/cuda/h3/timestep_sinusoid.cu",
    "csrc/cuda/h3/det_linear.cu",
    "csrc/cuda/h3/adaln_row_gather.cu",
    "csrc/cuda/h3/rmsnorm_modulate.cu",
)
H3_KERNELS_DEFINE = "-DKERNEL_ALIGN_WITH_H3_KERNELS"
# nvcc options that relax FP32 math. They can also arrive through the environment, where PyTorch's
# BuildExtension appends them after the flags built here, so the H3 kernels check for them too.
# nvcc takes ``--ftz``, ``--prec-div`` and ``--prec-sqrt`` with a value, as ``--opt=value`` or
# ``--opt value``, and accepts one or two leading dashes; ``--use_fast_math`` takes none. nvcc only
# takes true / false for them and stops on ``--ftz 1``; 1 / 0 are listed too, so that a spelling
# nvcc rejects is never read as the safe one.
_RELAXED_MATH_FLAG = "use_fast_math"
_RELAXED_MATH_VALUE_FLAGS = {
    "ftz": {"true", "1"},
    "prec-div": {"false", "0"},
    "prec-sqrt": {"false", "0"},
}
# The spelling that keeps the math exact. A value that is neither this nor in the set above is one
# nvcc does not know; it is reported as unreadable, not taken for the safe one.
_EXACT_MATH_VALUES = {"ftz": "false", "prec-div": "true", "prec-sqrt": "true"}
# An options file is read in full before anything checks it; a device node or a huge file named by
# a stray variable would hang or exhaust memory the configure step.
_OPTIONS_FILE_MAX_BYTES = 1 << 20
# nvcc reads more options from the files named by ``--options-file`` / ``-optf`` (a comma-separated
# list, as ``--options-file=a,b`` or ``--options-file a,b``), also when the option comes from
# NVCC_*_FLAGS. nvcc 13.0 does not expand ``@file`` (it takes it for an input file and stops); that
# spelling is read as well, so a driver that does expand it is covered.
_OPTIONS_FILE_FLAGS = {"optf", "options-file"}
_OPTIONS_FILE_DEPTH = 4


class _UncheckableFlags(Exception):
    """The flags cannot be read well enough to say whether they relax math."""


def _nvcc_flag_words(text: str) -> list[str]:
    """Split like nvcc does for the double-quoted values it accepts, and on ``;`` as well.

    ``;`` separates the flags of TORCH_NVCC_FLAGS; here it separates words outside quotes only, so
    ``--options-file "my opts;1.txt"`` keeps its file name in one piece, as nvcc reads it.
    """

    lexer = shlex.shlex(text, posix=True)
    lexer.whitespace_split = True
    lexer.whitespace += ";"
    lexer.commenters = ""
    return list(lexer)


def _read_options_file(name: str) -> str:
    path = Path(name)
    try:
        info = path.stat()  # a missing file is an OSError, reported as unreadable
        if not stat.S_ISREG(info.st_mode):
            raise _UncheckableFlags(f"options file {name!r} is not a regular file")
        if info.st_size > _OPTIONS_FILE_MAX_BYTES:
            raise _UncheckableFlags(
                f"options file {name!r} is larger than {_OPTIONS_FILE_MAX_BYTES} bytes"
            )
        return path.read_text()
    except OSError as exc:
        raise _UncheckableFlags(f"cannot read options file {name!r}: {exc.strerror}") from exc


def _nvcc_flag_tokens(text: str, *, depth: int = 0) -> list[str]:
    """Split an NVCC_*_FLAGS value into tokens, expanding options files.

    A word that holds whitespace or ``;`` came out of quotes; nvcc takes it as one argument (an
    option it then rejects, or a file name). It is kept whole for that reading, and its pieces
    follow it, so ``"-O2 --use_fast_math"`` in one quoted word still reads as relaxed. Raises
    _UncheckableFlags for an options file that cannot be read or an option nvcc rejects for lack
    of its file name, and ValueError for a string that cannot be split.
    """

    words = _nvcc_flag_words(text)
    tokens: list[str] = []
    i = 0
    while i < len(words):
        token = words[i]
        i += 1
        files: list[str]
        if token.startswith("@"):
            files = [token[1:]]
        elif token.startswith("-") and token.lstrip("-").partition("=")[0] in _OPTIONS_FILE_FLAGS:
            _, has_value, value = token.lstrip("-").partition("=")
            if not has_value:
                if i >= len(words) or words[i].startswith("-"):
                    raise _UncheckableFlags(f"{token} has no file name")
                value = words[i]
                i += 1
            files = value.split(",")
        else:
            tokens.append(token)
            pieces = token.replace(";", " ").split()
            if len(pieces) > 1:
                tokens.extend(pieces)
            continue
        if depth >= _OPTIONS_FILE_DEPTH:
            raise _UncheckableFlags(f"options files nest deeper than {_OPTIONS_FILE_DEPTH} levels")
        for name in files:
            tokens.extend(_nvcc_flag_tokens(_read_options_file(name), depth=depth + 1))
    return tokens


def _tokens_relax_math(tokens: list[str]) -> str | None:
    """Return the first option that relaxes FP32 math, spelled as given, or None."""

    for i, token in enumerate(tokens):
        if not token.startswith("-"):
            continue
        name, has_value, value = token.lstrip("-").partition("=")
        if name == _RELAXED_MATH_FLAG:
            return token
        relaxed_values = _RELAXED_MATH_VALUE_FLAGS.get(name)
        if relaxed_values is None:
            continue
        spelled = token
        if not has_value:
            # "--ftz true": the value is the next word. Without one, nvcc rejects the option; treat
            # it as relaxed rather than guess.
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if nxt.startswith("-") or not nxt:
                return f"{token} without a value"
            value = nxt
            spelled = f"{token} {nxt}"
        if value.lower() in relaxed_values:
            return spelled
        if value.lower() != _EXACT_MATH_VALUES[name]:
            return f"{spelled} (a value nvcc does not take)"
    return None


def _relaxed_math_reason() -> str | None:
    """Say why NVCC_*_FLAGS / TORCH_NVCC_FLAGS turn on relaxed FP32 math, or None if they do not.

    Fails closed: a value that cannot be parsed or an options file that cannot be read gives a
    reason too, because the H3 kernels would otherwise be built with flags nobody checked.
    """

    for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "TORCH_NVCC_FLAGS"):
        try:
            relaxed = _tokens_relax_math(_nvcc_flag_tokens(os.environ.get(name, "")))
        except ValueError as exc:
            return f"{name} cannot be parsed ({exc})"
        except _UncheckableFlags as exc:
            return f"{name}: {exc}"
        if relaxed is not None:
            return f"{name} contains {relaxed}"
    return None


def _resolve_package_data_source(
    filename: str | os.PathLike[str],
    *,
    project_root: Path | None = None,
) -> Path:
    """Dereference JSON package data, including symlink-disabled checkouts."""

    source = Path(filename)
    if source.suffix != ".json":
        return source

    root = (project_root or _PROJECT_ROOT).resolve()
    if source.is_symlink():
        target = source.resolve(strict=True)
    else:
        if not source.is_file() or source.stat().st_size > 4096:
            return source
        text = source.read_text(encoding="utf-8").strip()
        if not text or text.startswith(("{", "[")) or "\n" in text:
            return source
        if not text.endswith(".json"):
            return source
        target = (source.parent / text).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"package-data JSON link target does not exist: {target}")

    if not target.is_relative_to(root):
        raise ValueError(f"package-data JSON link escapes the project root: {source} -> {target}")
    if target.suffix != ".json":
        raise ValueError(f"package-data JSON link does not target JSON: {source} -> {target}")
    return target


class _DereferencePackageDataBuildPy(_BuildPy):
    """Copy target bytes instead of symlink metadata into wheels."""

    def copy_file(self, infile, outfile, *args, **kwargs):
        source = _resolve_package_data_source(infile)
        return super().copy_file(str(source), outfile, *args, **kwargs)


def _load_envs_module():
    envs_path = Path(__file__).with_name("envs.py")
    spec = importlib.util.spec_from_file_location("_rl_kernel_envs", envs_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load environment helpers from {envs_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


envs = _load_envs_module()


def _musa_build_available(torch) -> bool:
    try:
        import torch_musa  # noqa: F401
    except ImportError:
        return False
    return bool(
        hasattr(torch, "musa")
        and (torch.musa.is_available() or bool(os.environ.get("TORCH_MUSA_ARCH_LIST", "").strip()))
    )


def _load_torch_extension_tools():
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return None, None, None

    if _musa_build_available(torch):
        from torch_musa.utils.musa_extension import BuildExtension, MUSAExtension

        return torch, BuildExtension, MUSAExtension

    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # CUDAExtension is also the supported extension entry point for ROCm
    # PyTorch builds. BuildExtension dispatches .cu/.hip sources to hipcc when
    # torch.version.hip is set.
    return torch, BuildExtension, CUDAExtension


def _native_extension_required() -> bool:
    """Whether the caller explicitly requested a native extension build."""
    return (
        envs.env_flag(envs.RL_KERNEL_REQUIRE_EXT)
        or envs.env_flag(envs.RL_KERNEL_H3_STRICT)
        or bool(os.environ.get("PYTORCH_ROCM_ARCH", "").strip())
        or bool(os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip())
        or envs.env_flag("FORCE_CUDA")
        or bool(os.environ.get("TORCH_MUSA_ARCH_LIST", "").strip())
        or envs.env_flag("FORCE_MUSA")
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
    h3_flags = envs.h3_strict_cuda_flags()
    torch, _, GpuExtension = _load_torch_extension_tools()
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

    if h3_flags and (is_rocm or _musa_build_available(torch)):
        raise RuntimeError("RL_KERNEL_H3_STRICT currently qualifies NVIDIA CUDA only")

    if _musa_build_available(torch):
        extensions.append(
            GpuExtension(
                name="rl_engine._C",
                sources=[
                    "csrc/musa/ops.cpp",
                    "csrc/musa/fused_logp_kernel.mu",
                ],
                include_dirs=[],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17", "-DKERNEL_ALIGN_WITH_MUSA"],
                    "mcc": ["-O3", "-std=c++17", "-DKERNEL_ALIGN_WITH_MUSA"],
                },
                extra_link_args=list(torch_rpath),
            )
        )
        return extensions

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
        use_fast_math = envs.env_flag(envs.KERNEL_ALIGN_USE_FAST_MATH)
        # One predicate decides both the H3 sources and the define that enables their bindings, so
        # the bindings can never outlive the kernels.
        relaxed_reason = _relaxed_math_reason()
        relaxed_math_env = relaxed_reason is not None
        if relaxed_math_env and not is_rocm:
            warnings.warn(
                f"NVCC_*_FLAGS / TORCH_NVCC_FLAGS relax FP32 math or cannot be checked "
                f"({relaxed_reason}); the bitwise MiniMax-H3 kernels are left out of this build "
                "and dispatch falls back to the PyTorch reference.",
                RuntimeWarning,
                stacklevel=2,
            )
        h3_enabled = not is_rocm and not use_fast_math and not relaxed_math_env
        cuda_sources = [
            "csrc/bindings/ops.cpp",
            "csrc/cuda/logprob/fused_logp_kernel.cu",
            "csrc/cuda/logprob/deterministic_logp_kernel.cu",
            "csrc/cuda/gemm/det_gemm_kernel.cu",
            "csrc/cuda/norm/rmsnorm.cu",
            "csrc/cuda/activation/activation.cu",
            "csrc/cuda/attention/deterministic_attention.cu",
        ]
        if is_rocm:
            # ROCm-tuned WS2 vocab-parallel logprob kernels; the shared
            # deterministic_logp_kernel.cu keeps the SM90-tuned CUDA path.
            cuda_sources.extend(
                [
                    "csrc/rocm/logprob/hip_deterministic_logp_kernel.hip",
                    "csrc/rocm/collectives/deterministic_collective.hip",
                ]
            )
        else:
            # CUDA IPC and the fixed-tree collective implementation are not
            # part of the ROCm extension.
            cuda_sources.append("csrc/cuda/collectives/deterministic_collective.cu")
            if h3_enabled:
                cuda_sources.extend(H3_BITWISE_SOURCES)
            # This source contains NVIDIA PTX (cp.async, ldmatrix, and mma.sync).
            # The ROCm dispatcher falls back to PyTorch SDPA for this operator.
            cuda_sources.append("csrc/cuda/attention/prefix_shared_attention.cu")

        nvcc_flags = ["-O3", "-Xfatbin", "-compress-all"]
        if use_fast_math:
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
        if h3_enabled:
            cxx_flags.append(H3_KERNELS_DEFINE)
            nvcc_flags.append(H3_KERNELS_DEFINE)

        if not is_rocm:
            sm90_srcs = [
                "csrc/cuda/logprob/fused_logp_sm90.cu",
                "csrc/cuda/logprob/fused_linear_logp_sm90.cu",  # TMA + WGMMA fused linear log-prob
                "csrc/cuda/logprob/batch_invariant_logp_kernel_sm90.cu",  # TMA batch-invariant logp
                "csrc/cuda/rope/rope_sm90.cu",  # RoPE rotate-half apply, gated to SM90 build
                # Single-card batch-invariant embedding/lm-head.
                "csrc/cuda/embedding/embedding_lm_head_sm90.cu",
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
            GpuExtension(
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

        if h3_flags:
            # Separate DSO: strict arithmetic never changes other operators.
            extensions.append(
                GpuExtension(
                    name="rl_engine._H3_C",
                    sources=[
                        "csrc/bindings/h3_ops.cpp",
                        "csrc/cuda/attention/deterministic_attention.cu",
                    ],
                    extra_compile_args={
                        "cxx": ["-O3"],
                        "nvcc": [
                            "-O3",
                            *h3_flags,
                            "--expt-relaxed-constexpr",
                            "--expt-extended-lambda",
                        ],
                    },
                    extra_link_args=torch_rpath,
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

    asc_srcs = sorted(str(p) for p in Path("csrc/ascend").glob("**/*.asc"))
    if not asc_srcs:
        raise RuntimeError("KERNEL_ALIGN_FORCE_ASCEND=1 but no .asc sources under csrc/ascend/")
    sources: list[str] = asc_srcs
    # Some kernels ship a C++ pybind host alongside the .asc sources; include
    # it when present (compiled per-source by _bisheng_compile_cmd).
    host_cpp = Path("csrc/ascend/npu_module.cpp")
    if host_cpp.is_file():
        sources = [str(host_cpp), *asc_srcs]
    return [Extension(name="rl_engine._C_npu", sources=sources, language="asc")]


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

    if any(not str(src).endswith(".asc") for src in ext.sources):
        # Mixed extension (C++ pybind host + .asc kernels): the -x asc driver
        # cannot compile C++, so compile each source to an object and link them
        # in a second step.
        import subprocess
        import tempfile

        build_temp = tempfile.mkdtemp(prefix="rl_kernel_ascend_")
        objects = []
        for src in ext.sources:
            src = str(src)
            obj = os.path.join(build_temp, Path(src).name + ".o")
            src_cmd = [
                "bisheng",
                "-std=c++17",
                "-O2",
                "-fPIC",
                "-c",
                f"-D_GLIBCXX_USE_CXX11_ABI={abi_value}",
                f"-DTORCH_EXTENSION_NAME={module_name}",
            ]
            if src.endswith(".asc"):
                src_cmd += ["-x", "asc", f"--npu-arch={soc}"]
            src_cmd += [f"-I{d}" for d in include_dirs if d]
            src_cmd += [src, "-o", obj]
            subprocess.check_call(src_cmd)
            objects.append(obj)
        cmd = [
            "bisheng",
            "-shared",
            *objects,
            "-lascendcl",
            "-ltorch_npu",
            "-ltorch",
            "-ltorch_cpu",
            "-ltorch_python",
            "-lc10",
            "-o",
            ext_fullpath,
        ]
        cmd += [f"-L{d}" for d in lib_dirs if d]
        return cmd

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
    commands = {"build_py": _DereferencePackageDataBuildPy}
    if BuildExtension is None:
        return commands

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

    commands["build_ext"] = AscendBuildExtension
    return commands
