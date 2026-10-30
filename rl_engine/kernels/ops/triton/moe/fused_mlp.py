# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton fused MoE MLP kernels: four launches for the DSv4 MoE block.

=====  ===============================  ==========================================
 #      kernel                           invariance
=====  ===============================  ==========================================
 1      ``fused_shared_fc1_swiglu``      batch-invariant; TP-invariant by layout
 2      ``shared_fc3``                   batch-invariant; TP-invariant by K-tree
 3      ``routed_fc1_swiglu_quant``      batch-invariant
 4      ``routed_fc3``                   batch-invariant
=====  ===============================  ==========================================

These are ``tl.dot`` counterparts to the hand-written kernels in
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

Batch invariance holds the same way it does in the CUDA kernels: BLOCK sizes are
compile-time constants, pinned per GPU architecture and never chosen from the
token count; there is no split-K, no atomics, and every output element is
reduced over the whole K inside one program.

**TP invariance.** Under tensor parallelism the shared expert's fc1 is
column-parallel (each rank owns F/tp columns of gate and of up) and fc2 is
row-parallel (each rank owns F/tp of the reduction). fc1 needs nothing: a
column's whole K is local to one program at every TP size. fc2 is the one that
must be arranged, and ``shared_fc3`` does it by fixing the reduction tree to the
deterministic all-reduce's (``csrc/cuda/distributed/deterministic_collective.cu``,
``fixed_tree_reduce``)::

    F = [ s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 ]    8 contiguous segments
    y = BF16(((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)))

Each segment is F/8 wide, reduced in FP32 with ascending BK tiles and rounded
to BF16 once; every ``+`` above is one BF16 add (FP32 add, round to nearest even), the
same add the collective performs on BF16 payloads. At TP=t a rank holds 8/t
adjacent segments, which is a whole subtree, so it reduces that subtree and
sends a BF16 partial; the collective's tree over t ranks is the top of the same
tree. TP=1, 2, 4 and 8 therefore produce identical bytes. The K-tree does not
apply to fc1: ``SiLU(g1+g2)*(u1+u2) != SiLU(g1)*u1 + SiLU(g2)*u2``, and fc1's K
is never the split axis anyway.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
from torch import Tensor

from rl_engine.moe.mx_format import MX_BLOCK, MXTensor

# Triton can only close over globals that are tl.constexpr instances, so the
# contract's constants are mirrored here rather than read from the modules.
_GATE_MAX = tl.constexpr(10.0)      # rl_engine.moe.contract.GATE_CLAMP_MAX
_UP_MIN = tl.constexpr(-10.0)       # rl_engine.moe.contract.UP_CLAMP_MIN
_UP_MAX = tl.constexpr(10.0)        # rl_engine.moe.contract.UP_CLAMP_MAX
_E4M3_MAX = tl.constexpr(448.0)     # rl_engine.moe.mx_format.E4M3_MAX
_BIAS = tl.constexpr(127)           # rl_engine.moe.mx_format.E8M0_BIAS
_EMAX = tl.constexpr(8)             # EMAX_ELEM["e4m3"]
_MXB = tl.constexpr(MX_BLOCK)       # rl_engine.moe.mx_format.MX_BLOCK

# The deterministic all-reduce supports TP sizes 1, 2, 4 and 8, so the shared
# fc3 K-tree has 8 leaves: one per rank at the largest supported TP size.
TP_SEGMENTS = 8

# ------------------------------------------------------------------ tiles ---
# Pinned per architecture. Autotuning would pick configs per shape and break
# batch invariance; the arch is fixed for a deployment, the token count is not.
# BK is part of the numeric contract (it fixes the reduction order inside a
# tl.dot chain); BM / BN / warps / stages only change speed in principle, but
# they can change the MFMA/MMA instruction Triton picks, so they are pinned too.
_TILES: dict[str, dict[str, dict[str, int]]] = {
    "default": {
        "shared_fc1": dict(BM=64, BN=64, BK=64, num_warps=4, num_stages=3),
        "shared_fc3": dict(BM=64, BN=64, BK=64, num_warps=4, num_stages=3),
        "routed_fc1": dict(BM=64, BN=64, BK=64, num_warps=4, num_stages=3),
        "routed_fc3": dict(BM=64, BN=64, BK=64, num_warps=4, num_stages=3),
    },
    # MI300X, tuned with E=8 and E=256 routing (32..1024 rows per expert) at
    # H=4096 F=2048 and picked by geometric mean over those shapes. Measured
    # there, every BM/BN/BK/warps combination produced identical bytes; the
    # pin is still what the batch-invariance guarantee rests on.
    "gfx942": {
        "shared_fc1": dict(BM=128, BN=64, BK=128, num_warps=8, num_stages=2),
        "shared_fc3": dict(BM=128, BN=64, BK=64, num_warps=8, num_stages=2),
        "routed_fc1": dict(BM=128, BN=64, BK=128, num_warps=8, num_stages=2),
        "routed_fc3": dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=2),
        "routed_fc1_fp8": dict(BM=128, BN=64, BK=64, num_warps=8, num_stages=2),
        "routed_fc3_fp8": dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=2),
    },
}


@functools.lru_cache(maxsize=None)
def _arch(device_index: int) -> str:
    props = torch.cuda.get_device_properties(device_index)
    gcn = getattr(props, "gcnArchName", "")
    return gcn.split(":")[0] if gcn else f"sm{props.major}{props.minor}"


def tiles(kernel: str, device: torch.device) -> dict[str, int]:
    """The pinned launch config for ``kernel`` on ``device``'s architecture."""
    table = _TILES.get(_arch(device.index if device.index is not None else 0), _TILES["default"])
    return dict(table[kernel])


def _hw_fnuz(device: torch.device) -> bool:
    """CDNA3 (gfx94x) converts e4m3fnuz in hardware; see ``_load_mxfp8_a``."""
    return _arch(device.index if device.index is not None else 0).startswith("gfx94")


def _launch_kw(cfg: dict[str, int]) -> dict[str, int]:
    return {k: v for k, v in cfg.items() if k in ("num_warps", "num_stages", "waves_per_eu")}


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
def _locate_token_block(offsets, block_prefix, n_experts, pid, BM: tl.constexpr,
                        SEARCH_STEPS: tl.constexpr):
    """Map a program id to (expert, row0, row_end) over the sorted token rows.

    ``block_prefix[e]`` is the number of BM-row blocks before expert ``e``. The
    owning expert is the last one whose prefix is <= pid -- an empty expert has
    the same prefix as its successor, so it is skipped -- found by a fixed-step
    binary search rather than a scan, which matters at DSv4's 256 experts.
    ``row0 == row_end`` marks a program past the last block.
    """
    total = tl.load(block_prefix + n_experts)
    lo = 0
    hi = n_experts
    for _ in tl.static_range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        go_right = tl.load(block_prefix + mid) <= pid
        lo = tl.where(go_right, mid, lo)
        hi = tl.where(go_right, hi, mid)
    expert = lo
    row0 = tl.load(offsets + expert) + (pid - tl.load(block_prefix + expert)) * BM
    row_end = tl.load(offsets + expert + 1)
    row_end = tl.where(pid < total, row_end, row0)
    return expert, row0, row_end


@triton.jit
def _block_prefix_kernel(OFFS, PREFIX, n_experts, BM: tl.constexpr, BLOCK_E: tl.constexpr):
    """PREFIX[0] = 0, PREFIX[e+1] = sum_{j<=e} cdiv(rows_j, BM). Integer, exact."""
    e = tl.arange(0, BLOCK_E)
    mask = e < n_experts
    lo = tl.load(OFFS + e, mask=mask, other=0)
    hi = tl.load(OFFS + e + 1, mask=mask, other=0)
    nb = (hi - lo + BM - 1) // BM
    tl.store(PREFIX + 1 + e, tl.cumsum(nb, axis=0), mask=mask)
    tl.store(PREFIX + e, tl.zeros_like(nb), mask=e == 0)


# ------------------------------------- kernel 1: shared expert fc1 + SwiGLU --


@triton.jit
def _shared_fc1_swiglu_kernel(
    X, W1, H_OUT, T, H, F, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    """h = BF16(SiLU(gate) * up) with gate|up = x @ w_fc1^T, all BF16.

    One program owns [BM tokens, BN h-columns] and runs two accumulators so the
    gate and up halves of w_fc1 are reduced side by side; the SwiGLU pair is
    then local to the program and needs no second pass.
    """
    pid_m = tl.program_id(0).to(tl.int64)
    pid_f = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_f = pid_f * BN + tl.arange(0, BN)
    m_mask = offs_m < T
    f_mask = offs_f < F

    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
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
    """BF16 ``x [T, H]`` and ``w_fc1 [2F, H]`` -> BF16 ``h [T, F]``.

    Under TP pass the rank's shard ``[gate_shard; up_shard]`` (``[2F/tp, H]``);
    the result is byte-equal to the matching column slice of the TP=1 output.
    """
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
    cfg = tiles("shared_fc1", x.device)
    grid = (triton.cdiv(t, cfg["BM"]), triton.cdiv(f_dim, cfg["BN"]))
    _shared_fc1_swiglu_kernel[grid](
        x, w_fc1, out, t, h_dim, f_dim, BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"],
        **_launch_kw(cfg),
    )
    return out


# ------------------------------- kernel 2: shared expert fc3, TP-invariant ---


@triton.jit
def _bf16_add(a, b):
    """One BF16 add: exact operands, FP32 add, round to nearest even.

    The same operation as ``__hadd`` on BF16 in the deterministic collective and
    as ``torch.add`` on BF16 tensors.
    """
    return (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _segment_dot(HP, WP, offs_m, offs_n, m_mask, n_mask, K, k_lo, seg_len,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, EVEN_K: tl.constexpr):
    """One K-tree leaf: FP32 ascending-BK reduction of [k_lo, k_lo+seg_len), BF16 out.

    The leaf boundary is F/8 whatever BK is; a ragged last tile is zero-masked,
    which adds exact zeros and so leaves the leaf's sum unchanged.
    """
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(k_lo, k_lo + seg_len, BK):
        offs_k = k0 + tl.arange(0, BK)
        if EVEN_K:
            a_mask = m_mask[:, None]
            b_mask = n_mask[None, :]
        else:
            k_mask = offs_k < k_lo + seg_len
            a_mask = m_mask[:, None] & k_mask[None, :]
            b_mask = n_mask[None, :] & k_mask[:, None]
        a = tl.load(HP + offs_m[:, None] * K + offs_k[None, :], mask=a_mask, other=0.0)
        # w_fc2 is [H, F] row-major, used as the [F, H] right operand.
        b = tl.load(WP + offs_n[None, :] * K + offs_k[:, None], mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
    return acc.to(tl.bfloat16)


@triton.jit
def _shared_fc3_kernel(
    HP, WP, Y, T, N, K, seg_len, NSEG: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, EVEN_K: tl.constexpr,
):
    """y = h @ w_fc2^T over this rank's NSEG adjacent leaves of the 8-leaf K-tree.

    The tree is written out rather than looped so that it is visibly the
    collective's ``((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))``; at most three BF16
    partials are live next to the FP32 leaf accumulator.
    """
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < T
    n_mask = offs_n < N

    s = _segment_dot(HP, WP, offs_m, offs_n, m_mask, n_mask, K, 0, seg_len, BM, BN, BK, EVEN_K)
    if NSEG >= 2:
        s1 = _segment_dot(HP, WP, offs_m, offs_n, m_mask, n_mask, K, seg_len, seg_len, BM, BN, BK, EVEN_K)
        s = _bf16_add(s, s1)
    if NSEG >= 4:
        s2 = _segment_dot(HP, WP, offs_m, offs_n, m_mask, n_mask, K, 2 * seg_len, seg_len, BM, BN, BK, EVEN_K)
        s3 = _segment_dot(HP, WP, offs_m, offs_n, m_mask, n_mask, K, 3 * seg_len, seg_len, BM, BN, BK, EVEN_K)
        s = _bf16_add(s, _bf16_add(s2, s3))
    if NSEG >= 8:
        s4 = _segment_dot(HP, WP, offs_m, offs_n, m_mask, n_mask, K, 4 * seg_len, seg_len, BM, BN, BK, EVEN_K)
        s5 = _segment_dot(HP, WP, offs_m, offs_n, m_mask, n_mask, K, 5 * seg_len, seg_len, BM, BN, BK, EVEN_K)
        s45 = _bf16_add(s4, s5)
        s6 = _segment_dot(HP, WP, offs_m, offs_n, m_mask, n_mask, K, 6 * seg_len, seg_len, BM, BN, BK, EVEN_K)
        s7 = _segment_dot(HP, WP, offs_m, offs_n, m_mask, n_mask, K, 7 * seg_len, seg_len, BM, BN, BK, EVEN_K)
        s = _bf16_add(s, _bf16_add(s45, _bf16_add(s6, s7)))

    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], s, mask=m_mask[:, None] & n_mask[None, :])


def shared_fc3(h: Tensor, w_fc2: Tensor, tp_size: int = 1) -> Tensor:
    """TP-invariant shared-expert down projection. BF16 in, BF16 out.

    ``h [T, F/tp]`` and ``w_fc2 [H, F/tp]`` are this rank's shards (the whole
    tensors at ``tp_size=1``). The result is the rank's BF16 partial; summing
    the partials with the deterministic all-reduce's fixed tree gives the same
    bytes as ``tp_size=1``, whose result is the final ``y [T, H]``.

    Requires F divisible by ``TP_SEGMENTS``, so that every rank at every TP
    size holds whole leaves. The leaves are F/8 wide independent of BK.
    """
    if h.dtype is not torch.bfloat16 or w_fc2.dtype is not torch.bfloat16:
        raise TypeError(f"h and w_fc2 must be BF16, got {h.dtype} and {w_fc2.dtype}")
    if not (h.is_cuda and w_fc2.is_cuda):
        raise ValueError("shared_fc3 requires CUDA tensors")
    if tp_size not in (1, 2, 4, 8):
        raise ValueError(f"tp_size must be 1, 2, 4 or 8 (the collective's sizes), got {tp_size}")
    h, w_fc2 = h.contiguous(), w_fc2.contiguous()
    t, k_local = h.shape
    n = w_fc2.shape[0]
    if w_fc2.shape[1] != k_local:
        raise ValueError(f"K mismatch: h has F={k_local}, w_fc2 has {w_fc2.shape[1]}")
    cfg = tiles("shared_fc3", h.device)
    nseg = TP_SEGMENTS // tp_size
    if k_local % nseg:
        raise ValueError(
            f"local F={k_local} must split into {nseg} K-tree leaves "
            f"(F must be a multiple of {TP_SEGMENTS})"
        )
    seg_len = k_local // nseg
    y = torch.empty(t, n, dtype=torch.bfloat16, device=h.device)
    if t == 0 or n == 0:
        return y
    if k_local == 0:
        return y.zero_()
    grid = (triton.cdiv(t, cfg["BM"]), triton.cdiv(n, cfg["BN"]))
    _shared_fc3_kernel[grid](
        h, w_fc2, y, t, n, k_local, seg_len, NSEG=nseg,
        BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], EVEN_K=seg_len % cfg["BK"] == 0,
        **_launch_kw(cfg),
    )
    return y


# --------------------------------------------------- MX decode (portable) ----
# The routed kernels decode MX operands to BF16 with integer arithmetic and run
# a plain BF16 dot, instead of bitcasting to an FP8 type and using an FP8 dot.
# Two reasons, both about portability:
#
#   * ``tl.float8e4nv`` is NVIDIA's e4m3fn. CDNA3's native FP8 is e4m3fnuz
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
# Both decoders assemble the FP32 bit pattern directly (sign | exponent |
# mantissa) instead of computing ``sign * mantissa * 2**exp``: no exp2, no
# multiply by +-1, only shifts, ors and one select for the subnormal row.
#
# The loaders matter more than the decoders. Measured on MI300X (M=8192,
# H=4096, 2F=4096, 128x128x64 tiles): gathering the packed weight bytes along k
# -- each byte loaded twice, one nibble per load -- ran at 63 TFLOPS; loading
# them contiguously and splitting the nibbles ran at 157, and the hardware fnuz
# convert for the activation lifts that to 182. All variants are bit-identical.


@triton.jit
def _e8m0_scale(code):
    """2**(code - 127) from the bit pattern; code 0 is the subnormal 2**-127."""
    c = code.to(tl.uint32)
    return tl.where(c > 0, c << 23, 0x00400000).to(tl.float32, bitcast=True)


@triton.jit
def _decode_e4m3(code):
    """OCP E4M3 byte -> FP32. exp == 0 is the subnormal mant * 2**-9."""
    c = code.to(tl.uint32)
    exp = (c >> 3) & 0xF
    mant = c & 0x7
    # Normal: (1 + mant/8) * 2**(exp-7) -> FP32 exponent exp + 120.
    normal = ((exp + 120) << 23) | (mant << 20)
    sub = (mant.to(tl.float32) * 0.001953125).to(tl.uint32, bitcast=True)
    bits = tl.where(exp == 0, sub, normal) | ((c & 0x80) << 24)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _decode_e2m1(nib):
    """OCP E2M1 nibble -> FP32. Magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}."""
    n = nib.to(tl.uint32)
    exp = (n >> 1) & 0x3
    mant = n & 0x1
    # Normal: (1 + mant/2) * 2**(exp-1) -> FP32 exponent exp + 126.
    normal = ((exp + 126) << 23) | (mant << 22)
    sub = tl.where(mant != 0, 0x3F000000, 0)  # exp == 0: 0.5 * mant
    bits = tl.where(exp == 0, sub, normal) | ((n & 0x8) << 28)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _load_mxfp8_a(CODES, SCALES, offs_m, k0, m_mask, k_stride, nblk,
                  BK: tl.constexpr, HW_FNUZ: tl.constexpr):
    """Activation tile [BM, BK] as BF16, with its block scale folded in.

    One E8M0 scale per (row, 32-block), broadcast over the block rather than
    gathered per element. With ``HW_FNUZ`` (CDNA3) the E4M3 byte goes through
    the hardware FP8 convert: an OCP e4m3fn bit pattern read as e4m3fnuz is
    exactly half its value (the bias is one higher, subnormals included), so
    ``x2`` restores it. The two encodings differ only at 0x80, which is -0 in
    fn and NaN in fnuz; it is mapped to +0, which no sum can tell apart.
    """
    offs_k = k0 + tl.arange(0, BK)
    code = tl.load(
        CODES + offs_m[:, None] * k_stride + offs_k[None, :], mask=m_mask[:, None], other=0
    )
    if HW_FNUZ:
        code = tl.where(code == 0x80, 0, code)
        v = code.to(tl.float8e4b8, bitcast=True).to(tl.float32) * 2.0
    else:
        v = _decode_e4m3(code)
    offs_b = k0 // _MXB + tl.arange(0, BK // _MXB)
    s = tl.load(
        SCALES + offs_m[:, None] * nblk + offs_b[None, :], mask=m_mask[:, None], other=_BIAS
    )
    bm: tl.constexpr = v.shape[0]
    v = tl.reshape(v, (bm, BK // _MXB, _MXB)) * _e8m0_scale(s)[:, :, None]
    return tl.reshape(v, (bm, BK)).to(tl.bfloat16)


@triton.jit
def _load_mxfp4_b(CODES, SCALES, offs_n, k0, n_mask, k_stride, nblk, BK: tl.constexpr):
    """Weight tile [BK, BN] as BF16 from packed E2M1, block scale folded in.

    ``CODES`` is [..., K/2] with the low nibble holding the even k. The bytes
    are loaded as a contiguous [BN, BK/2] tile -- each byte once -- and the two
    nibbles decoded side by side; ``join`` + ``reshape`` puts them back in k
    order (lo = 2j, hi = 2j+1) before the transpose to the dot's [BK, BN].
    Both nibbles of a byte sit in the same 32-block, so they share one scale.
    """
    offs_j = k0 // 2 + tl.arange(0, BK // 2)
    byte = tl.load(
        CODES + offs_n[:, None] * (k_stride // 2) + offs_j[None, :], mask=n_mask[:, None], other=0
    )
    s = _e8m0_scale(tl.load(
        SCALES + offs_n[:, None] * nblk + (offs_j // (_MXB // 2))[None, :],
        mask=n_mask[:, None], other=_BIAS,
    ))
    lo = _decode_e2m1(byte & 0xF) * s
    hi = _decode_e2m1(byte >> 4) * s
    bn: tl.constexpr = lo.shape[0]
    return tl.trans(tl.reshape(tl.join(lo, hi), (bn, BK)).to(tl.bfloat16))


# ------------------------------------------------- native FP8 (CDNA3) -------
# On gfx94x the routed GEMMs can run on FP8 MFMA instead of BF16. Both operands
# have to be FP8 for that, and CDNA3's FP8 is e4m3fnuz, so:
#
#   * Activation (MX E4M3, OCP fn): no decode at all. An fn byte read as fnuz is
#     exactly half its value (bias 8 instead of 7, subnormals included); the x2
#     is applied once in the epilogue. 0x80 (-0 in fn, NaN in fnuz) becomes +0.
#   * Weight (MX E2M1): converted to fnuz with its block scale folded in
#     relative to a per-column reference exponent, value' = e2m1 * 2**(sw - ref).
#     Every E2M1 magnitude times 2**r is an exact fnuz number for r in [-9, 5]
#     (6 * 2**5 = 192 <= 240; 0.5 * 2**-9 = 2**-10 is the smallest subnormal),
#     so the conversion is exact; ``prepare_mxfp4_fp8`` checks the range once
#     per frozen weight and fails closed. 2**(ref - 127) is applied in the
#     epilogue. This is the CUDA kernel's ref_c trick with fnuz's range.
#   * Activation scale: per (row, 32-block), so it cannot be folded into an FP8
#     operand without losing bits. Each 32-block is one FP8 dot into a fresh
#     FP32 tile, then ``acc += blk * 2**(sa - 127)`` -- a power-of-two multiply,
#     so exact, and the same whether or not it is contracted into an FMA.
#
# The rounding differs from the BF16 lane (MFMA sums 32 products per block and
# blocks are added in ascending order), so this path has its own profile,
# ``p5-triton-fused-fp8-v1``. Batch invariance holds by the same construction.
#
# Opt-in (``fp8=True``), because on MI300X it loses to the BF16 lane. Measured
# on one 8192x4096x4096 GEMM: a whole-tile FP8 dot runs at ~0.35 ms, but one
# dot per 32-block plus the scale promote takes ~1.0 ms even with pre-converted
# weights, and ~1.4 ms with the in-kernel E2M1 conversion -- against ~1.16 ms
# for the BF16 lane. 16x16x32 MFMA, waves_per_eu and kpack do not help. The
# per-block structure is forced by the activation scale, not by the weights.

_FNUZ_RES_MIN = tl.constexpr(-9)
_FNUZ_RES_MAX = tl.constexpr(5)
FNUZ_RES_RANGE = (-9, 5)  # host side, for prepare_mxfp4_fp8


@triton.jit
def _load_fnuz_a(CODES, SCALES, offs_m, kb, m_mask, k_stride, nblk):
    """One 32-block of activations as e4m3fnuz [BM, 32] (half the true value), plus its scale."""
    offs_k = kb + tl.arange(0, _MXB)
    code = tl.load(
        CODES + offs_m[:, None] * k_stride + offs_k[None, :], mask=m_mask[:, None], other=0
    )
    a = tl.where(code == 0x80, 0, code).to(tl.float8e4b8, bitcast=True)
    sa = tl.load(SCALES + offs_m * nblk + kb // _MXB, mask=m_mask, other=_BIAS)
    return a, _e8m0_scale(sa)


@triton.jit
def _load_fnuz_b(CODES, SCALES, rows, ref, kb, n_mask, k_stride, nblk):
    """One 32-block of packed E2M1 weights as e4m3fnuz [32, BN], scale folded against ``ref``.

    Same contiguous byte load and nibble join as ``_load_mxfp4_b``. A zero
    magnitude is forced to +0: -0 would convert to 0x80, which is NaN in fnuz.
    """
    offs_j = kb // 2 + tl.arange(0, _MXB // 2)
    byte = tl.load(
        CODES + rows[:, None] * (k_stride // 2) + offs_j[None, :], mask=n_mask[:, None], other=0
    )
    sw = tl.load(SCALES + rows * nblk + kb // _MXB, mask=n_mask, other=_BIAS).to(tl.int32)
    # Clamped only for blocks that the range check exempts (none) and for masked
    # columns; an in-range residual is unchanged.
    r = tl.minimum(tl.maximum(sw - ref, _FNUZ_RES_MIN), _FNUZ_RES_MAX)
    p = ((r + 127).to(tl.uint32) << 23).to(tl.float32, bitcast=True)[:, None]
    lo = tl.where((byte & 0x7) == 0, 0.0, _decode_e2m1(byte & 0xF) * p)
    hi = tl.where((byte & 0x70) == 0, 0.0, _decode_e2m1(byte >> 4) * p)
    bn: tl.constexpr = lo.shape[0]
    w = tl.reshape(tl.join(lo, hi), (bn, _MXB)).to(tl.float8e4b8)
    return tl.trans(w)


@triton.jit
def _unfold_fnuz(acc, ref):
    """Undo the operand folds: x2 for the fnuz activation, 2**(ref-127) per column."""
    return (acc * 2.0) * _e8m0_scale(ref)[None, :]


# ----------------------- kernel 3: routed expert fc1 + SwiGLU + MX quant -----


@triton.jit
def _routed_fc1_kernel(
    XC, XS, WC, WS, WREF, OFFS, PREFIX, PS, HC, HS, H, F, n_experts,
    SEARCH_STEPS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HW_FNUZ: tl.constexpr, FP8: tl.constexpr,
):
    """MXFP8 x MXFP4 fc1, then clamp-SwiGLU * p_s, then MX re-quantization of h.

    Gate and up are reduced side by side so the SwiGLU pair stays inside the
    program, and ``h`` is quantized before it is ever stored: the FP32 z never
    reaches global memory.
    """
    expert, row0, row_end = _locate_token_block(
        OFFS, PREFIX, n_experts, tl.program_id(0), BM, SEARCH_STEPS
    )
    if row0 >= row_end:
        return
    pid_f = tl.program_id(1).to(tl.int64)
    offs_m = row0.to(tl.int64) + tl.arange(0, BM)
    offs_f = pid_f * BN + tl.arange(0, BN)
    m_mask = offs_m < row_end
    f_mask = offs_f < F
    nblk = H // _MXB
    gate_rows = expert.to(tl.int64) * (2 * F) + offs_f
    up_rows = gate_rows + F

    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    if FP8:
        ref_g = tl.load(WREF + gate_rows, mask=f_mask, other=_BIAS).to(tl.int32)
        ref_u = tl.load(WREF + up_rows, mask=f_mask, other=_BIAS).to(tl.int32)
        for k0 in tl.range(0, H, BK):
            for j in tl.static_range(BK // _MXB):
                kb = k0 + j * _MXB
                a, sa = _load_fnuz_a(XC, XS, offs_m, kb, m_mask, H, nblk)
                bg = _load_fnuz_b(WC, WS, gate_rows, ref_g, kb, f_mask, H, nblk)
                bu = _load_fnuz_b(WC, WS, up_rows, ref_u, kb, f_mask, H, nblk)
                acc_g += tl.dot(a, bg) * sa[:, None]
                acc_u += tl.dot(a, bu) * sa[:, None]
        acc_g = _unfold_fnuz(acc_g, ref_g)
        acc_u = _unfold_fnuz(acc_u, ref_u)
    else:
        for k0 in tl.range(0, H, BK):
            a = _load_mxfp8_a(XC, XS, offs_m, k0, m_mask, H, nblk, BK, HW_FNUZ)
            acc_g = tl.dot(a, _load_mxfp4_b(WC, WS, gate_rows, k0, f_mask, H, nblk, BK), acc_g)
            acc_u = tl.dot(a, _load_mxfp4_b(WC, WS, up_rows, k0, f_mask, H, nblk, BK), acc_u)

    p_s = tl.load(PS + offs_m, mask=m_mask, other=0.0)
    h = _swiglu_bf16(acc_g, acc_u, True, p_s[:, None]).to(tl.float32)

    # MX re-quantization: one E8M0 scale per 32 h-columns.
    groups: tl.constexpr = BN // _MXB
    hg = tl.reshape(h, (BM, groups, _MXB))
    code, inv = _e8m0_code_and_inv(tl.max(tl.abs(hg), axis=2))
    q = tl.reshape(hg * inv[:, :, None], (BM, BN))
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


# ------------------------------------------ kernel 4: routed expert fc3 ------


@triton.jit
def _routed_fc3_kernel(
    HC, HS, WC, WS, WREF, OFFS, PREFIX, Y, F, Hdim, n_experts,
    SEARCH_STEPS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HW_FNUZ: tl.constexpr, FP8: tl.constexpr,
):
    """MXFP8 x MXFP4 fc3 -> BF16 y."""
    expert, row0, row_end = _locate_token_block(
        OFFS, PREFIX, n_experts, tl.program_id(0), BM, SEARCH_STEPS
    )
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
    if FP8:
        ref = tl.load(WREF + w_rows, mask=n_mask, other=_BIAS).to(tl.int32)
        for k0 in tl.range(0, F, BK):
            for j in tl.static_range(BK // _MXB):
                kb = k0 + j * _MXB
                a, sa = _load_fnuz_a(HC, HS, offs_m, kb, m_mask, F, nblk)
                acc += tl.dot(a, _load_fnuz_b(WC, WS, w_rows, ref, kb, n_mask, F, nblk)) * sa[:, None]
        acc = _unfold_fnuz(acc, ref)
    else:
        for k0 in tl.range(0, F, BK):
            a = _load_mxfp8_a(HC, HS, offs_m, k0, m_mask, F, nblk, BK, HW_FNUZ)
            acc = tl.dot(a, _load_mxfp4_b(WC, WS, w_rows, k0, n_mask, F, nblk, BK), acc)

    tl.store(
        Y + offs_m[:, None] * Hdim + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


# ------------------------------------------------------------- wrappers ------


def _search_steps(n_experts: int) -> int:
    """Binary-search iterations that narrow [0, n_experts) to one expert."""
    return max(1, (n_experts - 1).bit_length())


def _grid_blocks(n_experts: int, rows: int, bm: int) -> int:
    """Upper bound on token blocks over all experts; no device sync needed."""
    return n_experts + triton.cdiv(rows, bm)


def _block_prefix(expert_offsets: Tensor, bm: int) -> Tensor:
    """Per-expert BM-block prefix on device, in one launch and with no sync."""
    n_experts = expert_offsets.numel() - 1
    prefix = torch.empty(n_experts + 1, dtype=torch.int32, device=expert_offsets.device)
    _block_prefix_kernel[(1,)](
        expert_offsets, prefix, n_experts, BM=bm,
        BLOCK_E=triton.next_power_of_2(max(n_experts, 1)),
    )
    return prefix


def fp8_available(device: torch.device) -> bool:
    """Whether the native FP8 lane can run here (CDNA3: e4m3fnuz MFMA)."""
    return _hw_fnuz(device)


def prepare_mxfp4_fp8(w: MXTensor) -> Tensor:
    """Per-column reference exponents ``ref [E, N]`` (uint8) for the FP8 lane.

    ``ref = max_j sw[e, n, j] - 5``, so every block residual ``sw - ref`` is at
    most 5; it fails closed when a column's scales span more than 14 binades,
    which would push a residual below -9 and make the fnuz conversion inexact.
    Weights are frozen, so this runs once per weight (it syncs to check).
    """
    if w.elem_format != "e2m1":
        raise TypeError(f"expected an e2m1 (MXFP4) weight, got {w.elem_format!r}")
    s = w.scales.to(torch.int16)
    if bool((s == 255).any()):
        raise ValueError("E8M0 NaN code 255 is not allowed")
    hi, lo = s.amax(-1), s.amin(-1)
    span = int((hi - lo).max()) if s.numel() else 0
    width = FNUZ_RES_RANGE[1] - FNUZ_RES_RANGE[0]
    if span > width:
        raise ValueError(
            f"weight block scales span {span} binades within a column; the FP8 lane folds "
            f"at most {width} exactly (use the BF16 lane, fp8=False)"
        )
    return (hi - FNUZ_RES_RANGE[1]).clamp(0, 254).to(torch.uint8).contiguous()


def _resolve_fp8(fp8: bool, w: MXTensor, w_ref: Tensor | None, dev: torch.device):
    use = fp8
    if use and not fp8_available(dev):
        raise NotImplementedError("the FP8 lane needs e4m3fnuz MFMA (gfx94x); use fp8=False")
    if not use:
        return False, w.scales  # placeholder pointer, never read
    return True, prepare_mxfp4_fp8(w) if w_ref is None else w_ref


def routed_fc1_swiglu_quant(
    x_q: MXTensor, w1: MXTensor, expert_offsets: Tensor, p_s: Tensor,
    *, fp8: bool = False, w1_ref: Tensor | None = None,
) -> MXTensor:
    """fc1 -> clamp-SwiGLU * p_s -> MX quant. Returns ``h_q`` [M, F].

    ``fp8=True`` selects the native FP8 lane (gfx94x only, profile
    ``p5-triton-fused-fp8-v1``). It is opt-in because it is slower than the
    BF16 lane on MI300X: exact MX semantics force one FP8 dot per 32-wide block
    (the activation scale changes every 32 k), and in Triton that costs ~3x a
    whole-tile FP8 dot. ``w1_ref`` is ``prepare_mxfp4_fp8(w1)``; pass it to skip
    recomputing (and re-checking) it.
    """
    m, h_dim = x_q.shape
    n_experts, two_f, wk = w1.shape
    if wk != h_dim:
        raise ValueError(f"K mismatch: x_q has H={h_dim}, w1 has {wk}")
    dev = x_q.codes.device
    use_fp8, wref = _resolve_fp8(fp8, w1, w1_ref, dev)
    cfg = tiles("routed_fc1_fp8" if use_fp8 else "routed_fc1", dev)
    if h_dim % cfg["BK"] or two_f % (2 * MX_BLOCK):
        raise ValueError(f"H must be a multiple of {cfg['BK']} and 2F a multiple of {2 * MX_BLOCK}")
    f_dim = two_f // 2
    codes = torch.empty(m, f_dim, dtype=torch.uint8, device=dev)
    scales = torch.empty(m, f_dim // MX_BLOCK, dtype=torch.uint8, device=dev)
    if m and f_dim:
        grid = (_grid_blocks(n_experts, m, cfg["BM"]), triton.cdiv(f_dim, cfg["BN"]))
        _routed_fc1_kernel[grid](
            x_q.codes, x_q.scales, w1.codes, w1.scales, wref, expert_offsets,
            _block_prefix(expert_offsets, cfg["BM"]), p_s.contiguous(),
            codes, scales, h_dim, f_dim, n_experts, SEARCH_STEPS=_search_steps(n_experts),
            BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], HW_FNUZ=_hw_fnuz(dev), FP8=use_fp8,
            **_launch_kw(cfg),
        )
    return MXTensor(codes=codes, scales=scales, elem_format="e4m3", shape=(m, f_dim))


def routed_fc3(
    h_q: MXTensor, w2: MXTensor, expert_offsets: Tensor,
    *, fp8: bool = False, w2_ref: Tensor | None = None,
) -> Tensor:
    """fc3 -> BF16 ``y [M, H]``. ``fp8`` / ``w2_ref`` as in ``routed_fc1_swiglu_quant``."""
    m, f_dim = h_q.shape
    n_experts, h_dim, wk = w2.shape
    if wk != f_dim:
        raise ValueError(f"K mismatch: h_q has F={f_dim}, w2 has {wk}")
    dev = h_q.codes.device
    use_fp8, wref = _resolve_fp8(fp8, w2, w2_ref, dev)
    cfg = tiles("routed_fc3_fp8" if use_fp8 else "routed_fc3", dev)
    if f_dim % cfg["BK"]:
        raise ValueError(f"F must be a multiple of {cfg['BK']}")
    y = torch.empty(m, h_dim, dtype=torch.bfloat16, device=dev)
    if m and h_dim:
        grid = (_grid_blocks(n_experts, m, cfg["BM"]), triton.cdiv(h_dim, cfg["BN"]))
        _routed_fc3_kernel[grid](
            h_q.codes, h_q.scales, w2.codes, w2.scales, wref, expert_offsets,
            _block_prefix(expert_offsets, cfg["BM"]), y, f_dim, h_dim, n_experts,
            SEARCH_STEPS=_search_steps(n_experts),
            BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], HW_FNUZ=_hw_fnuz(dev), FP8=use_fp8,
            **_launch_kw(cfg),
        )
    return y


def routed_mlp_forward(
    x_q: MXTensor, w1: MXTensor, w2: MXTensor, expert_offsets: Tensor, p_s: Tensor,
    *, fp8: bool = False, w1_ref: Tensor | None = None, w2_ref: Tensor | None = None,
) -> Tensor:
    """The two-stage routed expert: fc1+SwiGLU+quant, then fc3."""
    h_q = routed_fc1_swiglu_quant(x_q, w1, expert_offsets, p_s, fp8=fp8, w1_ref=w1_ref)
    return routed_fc3(h_q, w2, expert_offsets, fp8=fp8, w2_ref=w2_ref)
