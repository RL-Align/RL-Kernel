# SPDX-License-Identifier: Apache-2.0
"""Eager MXFP8 x checkpoint MXFP4 training on CDNA3.

Weights remain packed E2M1/E8M0 throughout forward and input VJP. Tiles decode
in registers; no persistent dense weight or TE FP4 capability override is used.
The profile pins 32-wide ascending block products, FP32 accumulation, and
identity STE for activation quantization. It is distinct from fused P5 MLP.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rl_engine.contracts.operators.expert_training import QuantizedExpertActivation
from rl_engine.kernels.ops.triton.moe.fused_mlp import (
    _block_prefix,
    _decode_e2m1,
    _decode_e4m3,
    _e8m0_code_and_inv,
    _e8m0_scale,
    _grid_blocks,
    _locate_token_block,
    _search_steps,
)


@triton.jit
def _quant(X, C, S, K: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    j = block * 32 + tl.arange(0, 32)
    v = tl.load(X + row * K + j).to(tl.float32)
    code, inv = _e8m0_code_and_inv(tl.max(tl.abs(v), 0))
    v = tl.minimum(tl.maximum(v * inv, -448.0), 448.0)
    # gfx942 does not implement the OCP FN cast in hardware. Triton lowers
    # this conversion; the stored bytes remain OCP E4M3, not FNUZ.
    q = v.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(C + row * K + j, q)
    tl.store(S + row * (K // 32) + block, code.to(tl.uint8))


@triton.jit
def _forward(
    AC,
    AS,
    WC,
    WS,
    OFF,
    PREFIX,
    Y,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
):
    e, start, end = _locate_token_block(OFF, PREFIX, E, tl.program_id(0), 16, SEARCH)
    if start >= end:
        return
    rows = start.to(tl.int64) + tl.arange(0, 16)
    cols = tl.program_id(1).to(tl.int64) * 32 + tl.arange(0, 32)
    kk = tl.arange(0, 32)
    acc = tl.zeros((16, 32), tl.float32)
    for block in range(K // 32):
        a = tl.load(AC + rows[:, None] * K + block * 32 + kk[None, :], rows[:, None] < end, other=0)
        a = _decode_e4m3(a).to(tl.bfloat16)
        packed = tl.load(
            WC + (e.to(tl.int64) * N + cols[None, :]) * (K // 2) + block * 16 + kk[:, None] // 2,
            cols[None, :] < N,
            other=0,
        )
        nib = tl.where(kk[:, None] % 2 == 0, packed & 15, packed >> 4)
        w = _decode_e2m1(nib).to(tl.bfloat16)
        sa = _e8m0_scale(tl.load(AS + rows * (K // 32) + block, rows < end, other=127))
        sw = _e8m0_scale(
            tl.load(WS + (e.to(tl.int64) * N + cols) * (K // 32) + block, cols < N, other=127)
        )
        partial = tl.dot(a, w)
        acc = acc + (partial * sa[:, None]) * sw[None, :]
    tl.store(
        Y + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < end) & (cols[None, :] < N)
    )


@triton.jit
def _backward(
    DY,
    WC,
    WS,
    OFF,
    PREFIX,
    DX,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
):
    e, start, end = _locate_token_block(OFF, PREFIX, E, tl.program_id(0), 16, SEARCH)
    if start >= end:
        return
    rows = start.to(tl.int64) + tl.arange(0, 16)
    cols = tl.program_id(1).to(tl.int64) * 32 + tl.arange(0, 32)
    nn = tl.arange(0, 32)
    acc = tl.zeros((16, 32), tl.float32)
    for n0 in range(triton.cdiv(N, 32)):
        n = n0 * 32 + nn
        dy = tl.load(
            DY + rows[:, None] * N + n[None, :], (rows[:, None] < end) & (n[None, :] < N), other=0
        ).to(tl.bfloat16)
        byte = tl.load(
            WC + (e.to(tl.int64) * N + n[:, None]) * (K // 2) + cols[None, :] // 2,
            (n[:, None] < N) & (cols[None, :] < K),
            other=0,
        )
        nib = tl.where(cols[None, :] % 2 == 0, byte & 15, byte >> 4)
        scale = tl.load(
            WS + (e.to(tl.int64) * N + n[:, None]) * (K // 32) + cols[None, :] // 32,
            (n[:, None] < N) & (cols[None, :] < K),
            other=127,
        )
        w = (_decode_e2m1(nib) * _e8m0_scale(scale)).to(tl.bfloat16)
        acc = tl.dot(dy, w, acc)
    tl.store(
        DX + rows[:, None] * K + cols[None, :], acc, (rows[:, None] < end) & (cols[None, :] < K)
    )


class RocmPackedMXBackend:
    profile = "dsv4-mxfp8-mxfp4-rocm-block32-fp32-v1"
    quant_backward = "identity-ste"

    def validate(self, config, device):
        if not torch.version.hip or device.type != "cuda":
            raise RuntimeError("packed MX requires a ROCm GPU")
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        if arch not in ("gfx942", "gfx950"):
            raise RuntimeError(f"packed MX requires gfx942/gfx950, got {arch}")
        if config.hidden % 32 or config.ffn % 32:
            raise ValueError("packed MX dimensions must be divisible by 32")

    def quantize(self, x):
        if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError("quantization requires contiguous BF16 [M,K]")
        if x.shape[1] % 32 or not bool(torch.isfinite(x).all()):
            raise ValueError("quantization requires finite block-32 activations")
        c = torch.empty_like(x, dtype=torch.uint8)
        s = torch.empty((x.shape[0], x.shape[1] // 32), device=x.device, dtype=torch.uint8)
        if x.shape[0]:
            _quant[(x.shape[0], x.shape[1] // 32)](x, c, s, x.shape[1], enable_fp_fusion=False)
        return QuantizedExpertActivation(c, s)

    @staticmethod
    def _check(w, offsets, rows, device):
        w.validate(w.shape, device)
        if not w.codes.is_contiguous() or not w.scales.is_contiguous():
            raise ValueError("packed weights must be contiguous")
        if offsets.device != device or offsets.dtype not in (torch.int32, torch.int64):
            raise ValueError("offsets must be GPU int32/int64")
        if offsets.shape != (w.shape[0] + 1,) or not offsets.is_contiguous():
            raise ValueError("offsets must cover every expert")
        if (
            offsets[0].item() != 0
            or offsets[-1].item() != rows
            or bool((offsets[1:] < offsets[:-1]).any())
        ):
            raise ValueError("invalid expert offsets")

    def forward(self, q, w, offsets):
        m, k = q.codes.shape
        self._check(w, offsets, m, q.codes.device)
        if k != w.shape[-1] or not q.codes.is_contiguous() or not q.scales.is_contiguous():
            raise ValueError("activation shape/layout differs from packed weight")
        q.validate(torch.empty((m, k), device=q.codes.device))
        e, n, _ = w.shape
        y = torch.empty((m, n), device=q.codes.device, dtype=torch.float32)
        if m:
            _forward[(_grid_blocks(e, m, 16), triton.cdiv(n, 32))](
                q.codes,
                q.scales,
                w.codes,
                w.scales,
                offsets,
                _block_prefix(offsets, 16),
                y,
                k,
                n,
                e,
                _search_steps(e),
                num_warps=4,
                enable_fp_fusion=False,
            )
        return y

    def backward(self, dy, w, offsets):
        self._check(w, offsets, dy.shape[0], dy.device)
        if dy.ndim != 2 or dy.shape[1] != w.shape[1] or dy.dtype != torch.bfloat16:
            raise ValueError("input VJP requires aligned BF16 dY")
        e, n, k = w.shape
        dx = torch.empty((dy.shape[0], k), device=dy.device, dtype=torch.float32)
        if dy.shape[0]:
            _backward[(_grid_blocks(e, dy.shape[0], 16), triton.cdiv(k, 32))](
                dy.contiguous(),
                w.codes,
                w.scales,
                offsets,
                _block_prefix(offsets, 16),
                dx,
                k,
                n,
                e,
                _search_steps(e),
                num_warps=4,
                enable_fp_fusion=False,
            )
        return dx
