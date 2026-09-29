# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton fused MoE MLP kernels: the shared expert, and the routed expert in two stages.

These are `tl.dot` counterparts to the hand-written kernels in
``csrc/cuda/moe/``. They are deterministic and batch-invariant, and each
declares its own numeric profile -- none of them is byte-equal to the CUDA
kernels, for two measured reasons:

* **Sigmoid.** No Triton exponential reproduces nvcc's ``expf``, which is what
  ``torch.sigmoid`` and the CUDA cores use. Measured over 262144 elements at
  |g| up to ~16: ``tl.sigmoid`` and ``1/(1+tl.exp(-g))`` differ on 6 BF16
  outputs, libdevice ``exp`` on 2, and every variant differs on some input.
  A fused kernel computes ``gate`` internally, so it cannot take
  ``torch.sigmoid(gate)`` as an input the way the unfused Triton path does.
* **Reduction order.** ``tl.dot`` accumulates a BLOCK_K tile at a time; the CUDA
  shared-expert kernel walks det_gemm's mid-split K-tree with a BF16 round per
  32-wide leaf. These are different reductions by construction.

The K-tree is deliberately absent here. It exists to make a contiguous half-K
shard one child of the tree, and that property cannot survive a fused
activation anyway: ``SiLU(g1+g2)*(u1+u2) != SiLU(g1)*u1 + SiLU(g2)*u2``. With
byte-equality already off the table, a flat FP32 accumulator is both simpler and
more accurate.

Batch invariance holds the same way it does in the CUDA kernels: BLOCK sizes are
compile-time constants and never chosen from the token count, there is no
split-K, no atomics, and every output element is reduced over the whole K inside
one program.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from rl_engine.moe.mx_format import MX_BLOCK, MXTensor

# Triton can only close over globals that are tl.constexpr instances, so the
# contract's constants are mirrored here rather than read from the modules.
_E8M0_BIAS = 127  # host side
_GATE_MAX = tl.constexpr(10.0)      # rl_engine.moe.contract.GATE_CLAMP_MAX
_UP_MIN = tl.constexpr(-10.0)       # rl_engine.moe.contract.UP_CLAMP_MIN
_UP_MAX = tl.constexpr(10.0)        # rl_engine.moe.contract.UP_CLAMP_MAX
_E4M3_MAX = tl.constexpr(448.0)     # rl_engine.moe.mx_format.E4M3_MAX
_BIAS = tl.constexpr(127)           # rl_engine.moe.mx_format.E8M0_BIAS
_EMAX = tl.constexpr(8)             # EMAX_ELEM["e4m3"]
_MXB = tl.constexpr(MX_BLOCK)       # rl_engine.moe.mx_format.MX_BLOCK

# Pinned. Autotuning would pick per-shape configs and break batch invariance.
_BM, _BF, _BK = 64, 64, 64
_BN_FC3 = 64


# ----------------------------------------------------------------- helpers ---


@triton.jit
def _swiglu_bf16(g, u, clamp: tl.constexpr, p_s):
    """h = BF16(SiLU(gate) * up [* p_s]), FP32 math, one round at the end."""
    if clamp:  # routed variant (P5-2): gate has an upper clamp, up is two-sided
        g = tl.minimum(g, _GATE_MAX)
        u = tl.minimum(tl.maximum(u, _UP_MIN), _UP_MAX)
    h = (g * tl.sigmoid(g)) * u
    if clamp:
        h = h * p_s
    return h.to(tl.bfloat16)


@triton.jit
def _e8m0_code_and_inv(amax):
    """E8M0 code for a block amax, plus 1/scale. Matches mx_format exactly.

    Integer arithmetic on the bit pattern: the biased exponent field of a normal
    float is floor(log2(x)) + 127, so no libm call is involved and nothing on
    this path can be flushed to zero.
    """
    bits = amax.to(tl.uint32, bitcast=True) & 0x7FFFFFFF
    exp = (tl.maximum(bits, 0x00800000) >> 23).to(tl.int32) - _BIAS - _EMAX
    code = tl.minimum(tl.maximum(exp, -_BIAS), _BIAS) + _BIAS
    code = tl.where(bits == 0, _BIAS, code)
    # 1 / 2**(code - 127) == 2**(127 - code); code 0 is the subnormal 2**-127.
    inv_bits = tl.where(code > 0, (254 - code).to(tl.uint32) << 23, 0x7F000000)
    return code, inv_bits.to(tl.float32, bitcast=True)


@triton.jit
def _locate_token_block(offsets, n_experts: tl.constexpr, pid, BM: tl.constexpr):
    """Map a program id to (expert, row0, row_end) over the sorted token rows."""
    seen = 0
    expert = 0
    row0 = 0
    row_end = 0
    for e in range(n_experts):
        lo = tl.load(offsets + e)
        hi = tl.load(offsets + e + 1)
        nb = tl.cdiv(hi - lo, BM)
        hit = (pid >= seen) & (pid < seen + nb)
        expert = tl.where(hit, e, expert)
        row0 = tl.where(hit, lo + (pid - seen) * BM, row0)
        row_end = tl.where(hit, hi, row_end)
        seen += nb
    return expert, row0, row_end


# ------------------------------------------- shared expert: fc1 + SwiGLU -----


@triton.jit
def _shared_fc1_swiglu_kernel(
    X, W1, H_OUT, T, H, F, BM: tl.constexpr, BF: tl.constexpr, BK: tl.constexpr
):
    """h = BF16(SiLU(gate) * up) with gate|up = x @ w_fc1^T, all BF16.

    One program owns [BM tokens, BF h-columns] and runs two accumulators so the
    gate and up halves of w_fc1 are reduced side by side; the SwiGLU pair is
    then local to the program and needs no second pass.
    """
    pid_m = tl.program_id(0).to(tl.int64)
    pid_f = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_f = pid_f * BF + tl.arange(0, BF)
    m_mask = offs_m < T
    f_mask = offs_f < F

    acc_g = tl.zeros((BM, BF), dtype=tl.float32)
    acc_u = tl.zeros((BM, BF), dtype=tl.float32)
    for k0 in tl.range(0, H, BK):
        offs_k = k0 + tl.arange(0, BK)
        k_mask = offs_k < H
        a = tl.load(
            X + offs_m[:, None] * H + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # w_fc1 is [2F, H] row-major, used as the [H, 2F] right operand.
        wg = tl.load(
            W1 + offs_f[None, :] * H + offs_k[:, None],
            mask=f_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        wu = tl.load(
            W1 + (F + offs_f)[None, :] * H + offs_k[:, None],
            mask=f_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        acc_g = tl.dot(a, wg, acc_g)
        acc_u = tl.dot(a, wu, acc_u)

    h = _swiglu_bf16(acc_g, acc_u, False, 0.0)
    tl.store(
        H_OUT + offs_m[:, None] * F + offs_f[None, :],
        h,
        mask=m_mask[:, None] & f_mask[None, :],
    )


def fused_shared_fc1_swiglu(x: Tensor, w_fc1: Tensor) -> Tensor:
    """BF16 ``x [T, H]`` and ``w_fc1 [2F, H]`` -> BF16 ``h [T, F]``."""
    if x.dtype is not torch.bfloat16 or w_fc1.dtype is not torch.bfloat16:
        raise TypeError(f"x and w_fc1 must be BF16, got {x.dtype} and {w_fc1.dtype}")
    if not (x.is_cuda and w_fc1.is_cuda):
        raise ValueError("fused_shared_fc1_swiglu requires CUDA tensors")
    x, w_fc1 = x.contiguous(), w_fc1.contiguous()
    t, h_dim = x.shape
    if w_fc1.shape[1] != h_dim:
        raise ValueError(f"K mismatch: x has H={h_dim}, w_fc1 has {w_fc1.shape[1]}")
    if w_fc1.shape[0] % 2:
        raise ValueError("w_fc1 must have an even number of rows (gate | up)")
    f_dim = w_fc1.shape[0] // 2
    out = torch.empty(t, f_dim, dtype=torch.bfloat16, device=x.device)
    if t == 0 or f_dim == 0:
        return out
    grid = (triton.cdiv(t, _BM), triton.cdiv(f_dim, _BF))
    _shared_fc1_swiglu_kernel[grid](
        x, w_fc1, out, t, h_dim, f_dim, BM=_BM, BF=_BF, BK=_BK, num_warps=4, num_stages=3
    )
    return out


# --------------------------------------------------- MX decode (portable) ----
# The routed kernels decode MX operands to BF16 with integer arithmetic and run
# a plain BF16 dot, instead of bitcasting to an FP8 type and using an FP8 dot.
# Two reasons, both about portability:
#
#   * ``tl.float8e4nv`` is NVIDIA's e4m3fn. CDNA's native FP8 is e4m3fnuz
#     (``tl.float8e4b8``), a different bias with no negative zero, so an FP8 dot
#     is not the same instruction -- or the same numerics -- across vendors. The
#     P5 start kit is explicit that ROCm registers its own profile rather than
#     silently relaxing to fnuz.
#   * A BF16 dot is the best-supported path on both vendors.
#
# The decode is exact, which is what makes this a free choice rather than a
# precision loss: E4M3 carries 4 significand bits and E2M1 carries 2, BF16
# carries 8, and an E8M0 scale is a power of two, so ``code * 2**(s-127)`` lands
# in BF16 with bits to spare. Folding the scale in here also removes the need
# for a fresh accumulator per 32-wide block: one accumulating tl.dot over the
# whole K is enough, which is the idiomatic Triton matmul.
#
# On a machine whose FP8 dot is available and wanted, swapping these two
# helpers for a bitcast is a local change.


@triton.jit
def _e8m0_scale(code):
    """2**(code - 127) from the bit pattern; code 0 is the subnormal 2**-127."""
    c = code.to(tl.uint32)
    return tl.where(c > 0, c << 23, 0x00400000).to(tl.float32, bitcast=True)


@triton.jit
def _decode_e4m3(code):
    """OCP E4M3 byte -> FP32. exp == 0 is the subnormal mant * 2**-9."""
    c = code.to(tl.uint32)
    sign = tl.where((c & 0x80) != 0, -1.0, 1.0)
    exp = (c >> 3) & 0xF
    mant = c & 0x7
    # Normal: (1 + mant/8) * 2**(exp-7), assembled as an FP32 bit pattern.
    normal = ((exp + 120) << 23 | mant << 20).to(tl.float32, bitcast=True)
    mag = tl.where(exp == 0, mant.to(tl.float32) * 0.001953125, normal)
    return sign * mag


@triton.jit
def _decode_e2m1(nib):
    """OCP E2M1 nibble -> FP32. Magnitudes {0,.5,1,1.5,2,3,4,6}."""
    n = nib.to(tl.uint32)
    idx = n & 0x7
    sign = tl.where((n & 0x8) != 0, -1.0, 1.0)
    # idx < 2 is {0, 0.5}; from idx 2 the value is (1 + .5*(idx&1)) * 2**(idx//2 - 1).
    mag = tl.where(
        idx < 2,
        idx.to(tl.float32) * 0.5,
        (1.0 + 0.5 * (idx & 1).to(tl.float32)) * tl.exp2(((idx >> 1) - 1).to(tl.float32)),
    )
    return sign * mag


@triton.jit
def _load_mxfp8_a(CODES, SCALES, offs_m, offs_k, m_mask, k_stride, nblk):
    """Activation tile [BM, BK] as BF16, with its block scale folded in."""
    code = tl.load(
        CODES + offs_m[:, None] * k_stride + offs_k[None, :], mask=m_mask[:, None], other=0
    )
    s = tl.load(
        SCALES + offs_m[:, None] * nblk + (offs_k // _MXB)[None, :],
        mask=m_mask[:, None], other=_BIAS,
    )
    return (_decode_e4m3(code) * _e8m0_scale(s)).to(tl.bfloat16)


@triton.jit
def _load_mxfp4_b(CODES, SCALES, offs_n, offs_k, n_mask, k_stride, nblk):
    """Weight tile [BK, BN] as BF16 from packed E2M1, block scale folded in.

    ``CODES`` is [..., K/2] with the low nibble holding the even k, so the byte
    for (n, k) sits at ``n * (K/2) + k//2``.
    """
    byte = tl.load(
        CODES + offs_n[None, :] * (k_stride // 2) + (offs_k // 2)[:, None],
        mask=n_mask[None, :], other=0,
    )
    nib = (byte >> ((offs_k % 2) * 4)[:, None]) & 0xF
    s = tl.load(
        SCALES + offs_n[None, :] * nblk + (offs_k // _MXB)[:, None],
        mask=n_mask[None, :], other=_BIAS,
    )
    return (_decode_e2m1(nib) * _e8m0_scale(s)).to(tl.bfloat16)


# ------------------------------ routed expert, stage 1: fc1 + SwiGLU + quant --


@triton.jit
def _routed_fc1_kernel(
    XC, XS, WC, WS, OFFS, PS, HC, HS, H, F, n_experts: tl.constexpr,
    BM: tl.constexpr, BF: tl.constexpr, BK: tl.constexpr,
):
    """MXFP8 x MXFP4 fc1, then clamp-SwiGLU * p_s, then MX re-quantization of h.

    Gate and up are reduced side by side so the SwiGLU pair stays inside the
    program, and ``h`` is quantized before it is ever stored: the FP32 z never
    reaches global memory.
    """
    expert, row0, row_end = _locate_token_block(OFFS, n_experts, tl.program_id(0), BM)
    if row0 >= row_end:
        return
    pid_f = tl.program_id(1).to(tl.int64)
    offs_m = row0.to(tl.int64) + tl.arange(0, BM)
    offs_f = pid_f * BF + tl.arange(0, BF)
    m_mask = offs_m < row_end
    f_mask = offs_f < F
    nblk = H // _MXB
    gate_rows = expert.to(tl.int64) * (2 * F) + offs_f
    up_rows = gate_rows + F

    acc_g = tl.zeros((BM, BF), dtype=tl.float32)
    acc_u = tl.zeros((BM, BF), dtype=tl.float32)
    for k0 in tl.range(0, H, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = _load_mxfp8_a(XC, XS, offs_m, offs_k, m_mask, H, nblk)
        acc_g = tl.dot(a, _load_mxfp4_b(WC, WS, gate_rows, offs_k, f_mask, H, nblk), acc_g)
        acc_u = tl.dot(a, _load_mxfp4_b(WC, WS, up_rows, offs_k, f_mask, H, nblk), acc_u)

    p_s = tl.load(PS + offs_m, mask=m_mask, other=0.0)
    h = _swiglu_bf16(acc_g, acc_u, True, p_s[:, None]).to(tl.float32)

    # MX re-quantization: one E8M0 scale per 32 h-columns.
    groups: tl.constexpr = BF // _MXB
    hg = tl.reshape(h, (BM, groups, _MXB))
    code, inv = _e8m0_code_and_inv(tl.max(tl.abs(hg), axis=2))
    q = tl.reshape(hg * inv[:, :, None], (BM, BF))
    q = tl.minimum(tl.maximum(q, -_E4M3_MAX), _E4M3_MAX).to(tl.float8e4nv)
    tl.store(
        HC + offs_m[:, None] * F + offs_f[None, :],
        q.to(tl.uint8, bitcast=True),
        mask=m_mask[:, None] & f_mask[None, :],
    )
    offs_s = pid_f * groups + tl.arange(0, groups)
    tl.store(
        HS + offs_m[:, None] * (F // _MXB) + offs_s[None, :],
        code.to(tl.uint8),
        mask=m_mask[:, None] & (offs_s < F // _MXB)[None, :],
    )


# ------------------------------------------- routed expert, stage 2: fc3 -----


@triton.jit
def _routed_fc3_kernel(
    HC, HS, WC, WS, OFFS, Y, F, Hdim, n_experts: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """MXFP8 x MXFP4 fc3 -> BF16 y."""
    expert, row0, row_end = _locate_token_block(OFFS, n_experts, tl.program_id(0), BM)
    if row0 >= row_end:
        return
    pid_n = tl.program_id(1).to(tl.int64)
    offs_m = row0.to(tl.int64) + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < row_end
    n_mask = offs_n < Hdim
    nblk = F // _MXB
    w_rows = expert.to(tl.int64) * Hdim + offs_n

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, F, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = _load_mxfp8_a(HC, HS, offs_m, offs_k, m_mask, F, nblk)
        acc = tl.dot(a, _load_mxfp4_b(WC, WS, w_rows, offs_k, n_mask, F, nblk), acc)

    tl.store(
        Y + offs_m[:, None] * Hdim + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


# ------------------------------------------------------------- wrappers ------


def _grid_blocks(expert_offsets: Tensor, rows: int) -> int:
    """Upper bound on token blocks over all experts; no device sync needed."""
    return (expert_offsets.numel() - 1) + triton.cdiv(rows, _BM)


def routed_fc1_swiglu_quant(
    x_q: MXTensor, w1: MXTensor, expert_offsets: Tensor, p_s: Tensor
) -> MXTensor:
    """fc1 -> clamp-SwiGLU * p_s -> MX quant. Returns ``h_q`` [M, F]."""
    m, h_dim = x_q.shape
    n_experts, two_f, wk = w1.shape
    if wk != h_dim:
        raise ValueError(f"K mismatch: x_q has H={h_dim}, w1 has {wk}")
    if h_dim % _BK or two_f % (2 * MX_BLOCK):
        raise ValueError(f"H must be a multiple of {_BK} and 2F a multiple of {2 * MX_BLOCK}")
    f_dim = two_f // 2
    dev = x_q.codes.device
    codes = torch.empty(m, f_dim, dtype=torch.uint8, device=dev)
    scales = torch.empty(m, f_dim // MX_BLOCK, dtype=torch.uint8, device=dev)
    if m and f_dim:
        grid = (_grid_blocks(expert_offsets, m), triton.cdiv(f_dim, _BF))
        _routed_fc1_kernel[grid](
            x_q.codes, x_q.scales, w1.codes, w1.scales, expert_offsets, p_s.contiguous(),
            codes, scales, h_dim, f_dim, n_experts=n_experts,
            BM=_BM, BF=_BF, BK=_BK, num_warps=4, num_stages=3,
        )
    return MXTensor(codes=codes, scales=scales, elem_format="e4m3", shape=(m, f_dim))


def routed_fc3(h_q: MXTensor, w2: MXTensor, expert_offsets: Tensor) -> Tensor:
    """fc3 -> BF16 ``y [M, H]``."""
    m, f_dim = h_q.shape
    n_experts, h_dim, wk = w2.shape
    if wk != f_dim:
        raise ValueError(f"K mismatch: h_q has F={f_dim}, w2 has {wk}")
    if f_dim % _BK:
        raise ValueError(f"F must be a multiple of {_BK}")
    y = torch.empty(m, h_dim, dtype=torch.bfloat16, device=h_q.codes.device)
    if m and h_dim:
        grid = (_grid_blocks(expert_offsets, m), triton.cdiv(h_dim, _BN_FC3))
        _routed_fc3_kernel[grid](
            h_q.codes, h_q.scales, w2.codes, w2.scales, expert_offsets, y,
            f_dim, h_dim, n_experts=n_experts, BM=_BM, BN=_BN_FC3, BK=_BK,
            num_warps=4, num_stages=3,
        )
    return y


def routed_mlp_forward(
    x_q: MXTensor, w1: MXTensor, w2: MXTensor, expert_offsets: Tensor, p_s: Tensor
) -> Tensor:
    """The two-stage routed expert: fc1+SwiGLU+quant, then fc3."""
    return routed_fc3(routed_fc1_swiglu_quant(x_q, w1, expert_offsets, p_s), w2, expert_offsets)
