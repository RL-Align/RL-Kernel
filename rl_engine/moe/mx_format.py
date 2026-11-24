# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Bit-exact CPU/GPU reference codecs for OCP Microscaling (MX) formats.

Implements the P5 quantization contract (#8; sub-issues P5-1 (#60), P5-4 (#61)):

- MX block size is fixed at 32 elements, blocked along the last dimension.
- Shared scales are E8M0 (8-bit power-of-two exponent, bias 127).
- MXFP8 elements are OCP E4M3 (torch ``float8_e4m3fn``); encode is
  clamp-to-[-448, 448] followed by round-to-nearest-even ("satfinite").
- MXFP4 elements are E2M1 with values {0, 0.5, 1, 1.5, 2, 3, 4, 6} per sign;
  encode is clamp-to-[-6, 6] followed by round-to-nearest-even.
- Scale derivation: ``shared_exp = floor(log2(amax)) - emax_elem`` where
  ``emax_elem`` is 8 for E4M3 and 2 for E2M1; an all-zero block gets code 127
  (scale 1.0). Non-finite inputs are rejected (fail-closed).
- FP4 codes are packed two per byte, low nibble first ("nibble-lo-first").

These functions define the golden bytes for the P5 fixtures; kernel backends
must reproduce them exactly or register an explicit numeric profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MX_BLOCK = 32
E8M0_BIAS = 127
E4M3_MAX = 448.0
E2M1_MAX = 6.0
EMAX_ELEM = {"e4m3": 8, "e2m1": 2}
NIBBLE_PACKING = "nibble-lo-first"

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_TIES_UP = (0.75, 1.75, 3.5)


@dataclass(frozen=True)
class MXTensor:
    """A block-scaled MX tensor (codes + E8M0 scales).

    ``shape`` is the logical element shape. For ``e4m3`` the codes tensor has
    exactly that shape (one byte per element); for ``e2m1`` the last dimension
    of ``codes`` is halved (two nibbles per byte, low nibble first).
    ``scales`` has the logical shape with the last dimension divided by 32.
    """

    codes: torch.Tensor
    scales: torch.Tensor
    elem_format: str
    shape: tuple[int, ...]
    packing: str = NIBBLE_PACKING

    def __post_init__(self) -> None:
        if self.elem_format not in EMAX_ELEM:
            raise ValueError(f"unsupported elem_format {self.elem_format!r}")
        if self.codes.dtype != torch.uint8 or self.scales.dtype != torch.uint8:
            raise TypeError("MXTensor codes/scales must be uint8")
        if self.shape[-1] % MX_BLOCK != 0:
            raise ValueError(f"last dim {self.shape[-1]} not divisible by MX block {MX_BLOCK}")

    def to(self, device: torch.device | str) -> "MXTensor":
        return MXTensor(
            self.codes.to(device),
            self.scales.to(device),
            self.elem_format,
            self.shape,
            self.packing,
        )


def _check_finite(x: torch.Tensor, what: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"non-finite values in {what}; P5 quantization is fail-closed")


def floor_log2(x: torch.Tensor) -> torch.Tensor:
    """Exact floor(log2(x)) for positive x via frexp (no libm log2 rounding)."""
    _, exp = torch.frexp(x)
    return exp.to(torch.int32) - 1


def e8m0_decode(code: torch.Tensor) -> torch.Tensor:
    """E8M0 code -> FP32 scale = 2**(code - 127). Code 255 (NaN) is rejected."""
    if bool((code == 255).any()):
        raise ValueError("E8M0 NaN code 255 is not allowed in the P5 contract")
    return torch.ldexp(
        torch.ones(code.shape, dtype=torch.float32, device=code.device),
        code.to(torch.int32) - E8M0_BIAS,
    )


def e8m0_scale_from_amax(amax: torch.Tensor, elem_format: str) -> torch.Tensor:
    """Derive the shared-scale code: floor(log2(amax)) - emax_elem, bias 127.

    All-zero blocks (amax == 0) get code 127 (scale 1.0).
    """
    _check_finite(amax, "amax")
    if bool((amax < 0).any()):
        raise ValueError("amax must be non-negative")
    emax = EMAX_ELEM[elem_format]
    exp = floor_log2(torch.clamp(amax, min=torch.finfo(torch.float32).tiny)) - emax
    exp = torch.clamp(exp, min=-E8M0_BIAS, max=E8M0_BIAS)
    code = (exp + E8M0_BIAS).to(torch.uint8)
    return torch.where(amax == 0, torch.full_like(code, E8M0_BIAS), code)


def e4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """FP32 -> OCP E4M3 byte codes: clamp to +/-448 then RNE cast (satfinite).

    The clamp-then-cast pair is the frozen contract; torch's bare cast maps
    overflow to NaN, so the clamp must never be removed.
    """
    _check_finite(x, "e4m3 input")
    clamped = torch.clamp(x.to(torch.float32), min=-E4M3_MAX, max=E4M3_MAX)
    return clamped.to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_decode(codes: torch.Tensor) -> torch.Tensor:
    return codes.view(torch.float8_e4m3fn).to(torch.float32)


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """FP32 -> E2M1 nibble codes (0..15, sign in bit 3), RNE with saturation."""
    _check_finite(x, "e2m1 input")
    x32 = x.to(torch.float32)
    sign = torch.signbit(x32)
    a = torch.clamp(x32.abs(), max=E2M1_MAX)
    boundaries = torch.tensor(_E2M1_BOUNDARIES, dtype=torch.float32, device=x32.device)
    # side='left': exact midpoints land on the lower code ...
    idx = torch.searchsorted(boundaries, a.reshape(-1), right=False).reshape(a.shape)
    # ... then bump the three midpoints whose round-to-even target is the upper code.
    for tie in _E2M1_TIES_UP:
        idx = torch.where(a == tie, idx + 1, idx)
    return (idx.to(torch.uint8)) | (sign.to(torch.uint8) << 3)


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=codes.device)
    mag = table[(codes & 0x7).to(torch.long)]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0).to(torch.float32)
    return mag * sign


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codes two per byte along the last dim, low nibble first."""
    if codes.shape[-1] % 2 != 0:
        raise ValueError("last dim must be even to pack nibbles")
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return lo | (hi << 4)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    lo = packed & 0xF
    hi = packed >> 4
    out = torch.stack((lo, hi), dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def mx_quantize(x: torch.Tensor, elem_format: str) -> MXTensor:
    """BF16/FP32 -> MX tensor with block-32 E8M0 scales along the last dim.

    The amax reduction is strictly within one 32-element block of one row;
    it never crosses rows (and therefore never crosses ranks).
    """
    if elem_format not in EMAX_ELEM:
        raise ValueError(f"unsupported elem_format {elem_format!r}")
    x32 = x.to(torch.float32)
    _check_finite(x32, "mx_quantize input")
    shape = tuple(x32.shape)
    if shape[-1] % MX_BLOCK != 0:
        raise ValueError(f"last dim {shape[-1]} not divisible by MX block {MX_BLOCK}")
    blocked = x32.reshape(*shape[:-1], shape[-1] // MX_BLOCK, MX_BLOCK)
    amax = blocked.abs().amax(dim=-1)
    scale_codes = e8m0_scale_from_amax(amax, elem_format)
    scale = e8m0_decode(scale_codes)
    scaled = (blocked / scale.unsqueeze(-1)).reshape(shape)
    if elem_format == "e4m3":
        codes = e4m3_encode(scaled)
    else:
        codes = pack_nibbles(e2m1_encode(scaled))
    return MXTensor(codes=codes, scales=scale_codes, elem_format=elem_format, shape=shape)


def mx_dequantize(t: MXTensor) -> torch.Tensor:
    """MX tensor -> FP32 (exact: element decode and power-of-two scale)."""
    if t.elem_format == "e4m3":
        elems = e4m3_decode(t.codes)
    else:
        elems = e2m1_decode(unpack_nibbles(t.codes))
    scale = e8m0_decode(t.scales)
    blocked = elems.reshape(*t.shape[:-1], t.shape[-1] // MX_BLOCK, MX_BLOCK)
    return (blocked * scale.unsqueeze(-1)).reshape(t.shape)
