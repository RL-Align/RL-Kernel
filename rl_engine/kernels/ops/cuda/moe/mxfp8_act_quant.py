# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA MXFP8 activation quantization (P5-1).

Thin wrapper around ``csrc/cuda/moe/mxfp8_act_quant.cu``. The kernel reproduces
``rl_engine.moe.mx_format.mx_quantize(x, "e4m3")`` byte for byte (numeric
profile ``oracle-fp32-serial-v1``); see the ``.cu`` header for the frozen
recipe and how it stays immune to ``--use_fast_math``.

The compiled ``rl_engine._C`` extension is used when it exports the symbols. A
source tree without a built extension falls back to a JIT build of that single
``.cu`` file (``torch.utils.cpp_extension.load``) so the alignment tests and
the benchmark run without rebuilding the whole extension. The fallback is
CUDA-only (ROCm must register its own numeric profile), a failed build stays
failed for the process, and it is refused when ``RL_KERNEL_REQUIRE_EXT=1`` —
the repo's existing "the compiled extension must be present" switch, which
the GPU CI sets — so it can never paper over a missing AOT symbol there.
"""

from __future__ import annotations

import os
import pathlib
import threading
from typing import Any

import torch
from torch import Tensor

try:
    import envs
except ImportError:  # installed wheel: the repo-root envs.py is not shipped
    envs = None

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.moe_common import (
    finalize_act_quant,
    validate_act_quant_input,
    validate_ste_grad,
)
from rl_engine.moe.mx_format import MXTensor
from rl_engine.utils.logger import logger

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
_CU_SOURCE = _REPO_ROOT / "csrc" / "cuda" / "moe" / "mxfp8_act_quant.cu"
_AOT_SYMBOLS = ("mxfp8_act_quant_forward", "mxfp8_act_quant_ste_backward")

_lock = threading.Lock()
_impl: Any = None  # resolved once per process: rl_engine._C or the JIT module
_impl_error: BaseException | None = None


def _aot_available() -> bool:
    return _EXT_AVAILABLE and _C is not None and all(hasattr(_C, s) for s in _AOT_SYMBOLS)


def _jit_load() -> Any:
    require_ext = (
        envs.env_flag(envs.RL_KERNEL_REQUIRE_EXT)
        if envs is not None
        else os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1"
    )
    if require_ext:
        raise RuntimeError(
            "CUDA mxfp8_act_quant requires the compiled rl_engine._C extension "
            "(rebuild with csrc/cuda/moe/mxfp8_act_quant.cu); the JIT fallback is "
            "disabled because RL_KERNEL_REQUIRE_EXT=1."
        )
    if torch.version.hip is not None:
        raise RuntimeError(
            "CUDA mxfp8_act_quant has no ROCm build: the source needs cuda_fp8.h. "
            "ROCm must register its own P5 numeric profile."
        )
    if not _CU_SOURCE.exists():
        raise RuntimeError(f"P5-1 CUDA source not found at {_CU_SOURCE}")
    from torch.utils.cpp_extension import load

    logger.warning(
        "rl_engine._C has no mxfp8_act_quant symbols; JIT-building "
        f"{_CU_SOURCE.name} (first call only)."
    )
    return load(
        name="rl_engine_p5_mxfp8_act_quant",
        sources=[str(_CU_SOURCE)],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DRL_KERNEL_P5_STANDALONE"],
        verbose=False,
    )


def backend() -> Any:
    """The module exporting the two symbols: ``rl_engine._C`` (AOT) or the JIT build.

    Resolved once per process; a failed JIT build is remembered so later calls
    raise the real cause instead of torch's misleading "cannot open .so".
    """
    global _impl, _impl_error
    if _impl is not None:
        return _impl
    with _lock:
        if _impl is not None:
            return _impl
        if _impl_error is not None:
            raise RuntimeError(
                "P5-1 CUDA JIT build failed earlier in this process"
            ) from _impl_error
        try:
            _impl = _C if _aot_available() else _jit_load()
        except BaseException as exc:
            _impl_error = exc
            raise
        return _impl


def backend_name() -> str:
    """``"aot"`` when the symbols come from ``rl_engine._C``, else ``"jit"``."""
    return "aot" if _aot_available() else "jit"


def mxfp8_act_quant_fwd_cuda(x: Tensor, check_finite: bool = True) -> MXTensor:
    """BF16/FP16/FP32 ``[..., K]`` -> MXFP8 (E4M3 codes + block-32 E8M0 scales).

    ``check_finite`` reads back the kernel's fail-closed flag and therefore
    costs one device sync per call; it is only turned off for throughput
    measurement (the providers always keep it on), which also skips the flag
    memset.
    """
    x = validate_act_quant_input(x, "cuda")
    codes, scales, nonfinite = backend().mxfp8_act_quant_forward(x, check_finite)
    return finalize_act_quant(codes, scales, nonfinite, check_finite, tuple(x.shape))


def mxfp8_act_quant_bwd_cuda(dy: Tensor) -> Tensor:
    """Straight-through estimator: ``dX = dY`` for any floating dtype (contiguous, same shape)."""
    return backend().mxfp8_act_quant_ste_backward(validate_ste_grad(dy, "cuda"))
