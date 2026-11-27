# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Per-token-group FP8 (E4M3) activation quantization for DeepGEMM (DSv4, SM90).

Golden reference. It reproduces vLLM's ``per_token_group_fp8_quant`` with the
DSv4 rollout settings (group 128, ``eps=1e-10``, ``VLLM_USE_DEEP_GEMM_E8M0=1``)
byte for byte, so rollout and training feed DeepGEMM the same FP8 operands.
For every group of 128 consecutive values along the last dimension:

    absmax = fmax(eps, max |x|)            # fmax: NaN does not count
    s      = absmax / 448                  # FP32, IEEE division
    s      = fmax(s, 1e-10)
    s      = 2 ** ceil(log2(s))            # UE8M0, from the FP32 bits
    q      = e4m3(fmin(fmax(x / s, -448), 448))

The UE8M0 rounding uses the exponent and mantissa bits. vLLM's FP32-scale
kernel uses ``exp2f(ceilf(log2f(s)))``, which rounds down when ``s`` lies a
few ULPs above a power of two; with BF16/FP16 inputs ``s`` never does (checked
over every finite input value), so the two agree. FP32 inputs are rejected for
that reason. As in vLLM, a NaN input quantizes to -448 and an infinite group
quantizes its finite values to zero.

Quantization is row-local, so it is batch-invariant by construction. Scales
are returned in DeepGEMM's SM90 layout: FP32, logical shape ``[M, K / 128]``,
column-major with the column stride padded to a multiple of four rows (the
TMA alignment), so DeepGEMM consumes them without a conversion kernel.

``fp8_group_quant_tokens`` quantizes along the token dimension instead, for
the wgrad operands: ``x [T, C]`` becomes ``x.T`` in 128-token groups. Tokens
are zero-padded to a multiple of 128; zero tokens add nothing to a wgrad.
"""

from __future__ import annotations

import torch
from torch import Tensor

GROUP_SIZE = 128
FP8_MAX = 448.0
EPS = 1.0e-10
SCALE_FLOOR = 1.0e-10
TMA_ALIGN_ROWS = 4  # 16 bytes of FP32 scales
INPUT_DTYPES = (torch.bfloat16, torch.float16)


def tma_aligned_rows(rows: int) -> int:
    return -(-rows // TMA_ALIGN_ROWS) * TMA_ALIGN_ROWS


def ue8m0_ceil(scale: Tensor) -> Tensor:
    """Round positive FP32 values up to a power of two, exactly."""
    bits = scale.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return ((exponent + (mantissa != 0).to(torch.int32)) << 23).view(torch.float32)


def _check(x: Tensor, name: str) -> None:
    if type(x) is not Tensor:
        raise TypeError(f"{name} must be a torch.Tensor.")
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(x.shape)}.")
    if x.dtype not in INPUT_DTYPES:
        raise TypeError(f"{name} must be BF16 or FP16, got {x.dtype}.")
    if x.device.type not in {"cpu", "cuda"}:
        raise RuntimeError(f"{name} must be a CPU or CUDA tensor.")


def scale_layout(rows: int, groups: int, device: torch.device) -> Tensor:
    """An empty FP32 ``[rows, groups]`` tensor in DeepGEMM's TMA-aligned layout."""
    return torch.empty_strided(
        (rows, groups), (1, tma_aligned_rows(rows)), dtype=torch.float32, device=device
    )


@torch.no_grad()
def _quantize_rows(x: Tensor, scale_ue8m0: bool) -> tuple[Tensor, Tensor]:
    rows, cols = x.shape
    groups = cols // GROUP_SIZE
    xf = x.float().view(rows, groups, GROUP_SIZE)
    magnitude = xf.abs()
    magnitude = torch.where(torch.isnan(magnitude), torch.zeros_like(magnitude), magnitude)
    absmax = torch.clamp_min(magnitude.amax(dim=2), EPS)
    # Tensor / tensor: on CUDA, dividing by a Python scalar multiplies by its
    # reciprocal, which is not the correctly rounded quotient.
    scale = torch.clamp_min(absmax / torch.full_like(absmax, FP8_MAX), SCALE_FLOOR)
    if scale_ue8m0:
        scale = ue8m0_ceil(scale)
    value = xf / scale.unsqueeze(2)
    bound = value.new_tensor(FP8_MAX)
    value = torch.fmin(torch.fmax(value, -bound), bound)
    q = value.to(torch.float8_e4m3fn).view(rows, cols)
    out_scale = scale_layout(rows, groups, x.device)
    out_scale.copy_(scale)
    return q, out_scale


def fp8_group_quant(x: Tensor, *, scale_ue8m0: bool = True) -> tuple[Tensor, Tensor]:
    """Quantize ``x [M, K]`` in 1x128 groups along K.

    Returns ``q`` (FP8 E4M3, ``[M, K]``, contiguous) and ``scale`` (FP32,
    ``[M, K / 128]``, TMA-aligned column-major)."""
    _check(x, "x")
    if x.shape[1] == 0 or x.shape[1] % GROUP_SIZE:
        raise ValueError(f"x's last dimension must be a positive multiple of {GROUP_SIZE}.")
    return _quantize_rows(x, scale_ue8m0)


def fp8_group_quant_tokens(x: Tensor, *, scale_ue8m0: bool = True) -> tuple[Tensor, Tensor]:
    """Quantize ``x [T, C]`` along tokens, as the K-major wgrad operand ``x.T``.

    Returns ``q`` (FP8 E4M3, ``[C, T_pad]``) and ``scale`` (FP32,
    ``[C, T_pad / 128]``, TMA-aligned column-major), where ``T_pad`` is ``T``
    rounded up to a multiple of 128 and the padded tokens are zero."""
    _check(x, "x")
    tokens, channels = x.shape
    if tokens == 0 or channels == 0:
        raise ValueError("x must have at least one token and one channel.")
    padded = -(-tokens // GROUP_SIZE) * GROUP_SIZE
    xt = torch.zeros(channels, padded, dtype=x.dtype, device=x.device)
    xt[:, :tokens] = x.t()
    return _quantize_rows(xt, scale_ue8m0)


class Fp8GroupQuantOp:
    """PyTorch golden for ``fp8_group_quant``."""

    def __call__(self, x: Tensor, *, scale_ue8m0: bool = True) -> tuple[Tensor, Tensor]:
        return self.forward(x, scale_ue8m0=scale_ue8m0)

    def forward(self, x: Tensor, *, scale_ue8m0: bool = True) -> tuple[Tensor, Tensor]:
        return fp8_group_quant(x, scale_ue8m0=scale_ue8m0)

    def forward_tokens(self, x: Tensor, *, scale_ue8m0: bool = True) -> tuple[Tensor, Tensor]:
        return fp8_group_quant_tokens(x, scale_ue8m0=scale_ue8m0)
