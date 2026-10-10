# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton backend for the Qwen-Image MLP up projection + GELU (WS1).

``pre = fp32_accum(x @ weight.T) + bias``
``y   = bf16_rne(gelu_tanh_fp32(pre))``
with a fixed K reduction order, an FP32 accumulator, no split-K, no atomics, the
bias added once in FP32 after the complete reduction, the tanh-approximate GELU
applied to that FP32 pre-activation and exactly one BF16 cast at the store.  Five
Triton kernels implement the operator and its VJP:

``_mlp_up_gemm_gelu_forward_kernel``
    one program per ``(BLOCK_M, BLOCK_N)`` output tile walks the whole K in
    ascending ``BLOCK_K`` chunks (``tl.dot(x_tile, w_tile)``, BF16 inputs, FP32
    accumulator, ``input_precision="ieee"`` so no TF32 path is ever selected),
    adds the bias once in FP32, then applies the frozen GELU sequence and casts
    to BF16.  When ``EMIT_PRE`` is set the same kernel also stores the *pre-GELU*
    FP32 accumulator (GEMM + bias) into a separate FP32 buffer -- no second K
    pass, so ``pre`` is exactly the operand the GELU above consumed.
``_mlp_up_gemm_gelu_gate_kernel``
    the activation's backward, ``gate = bf16(fp32(grad) * gelu'(pre))``, on the
    FP32 pre-activation: elementwise, no reduction.  This is the ROCm/MUSA slot's
    gate too, so it uses the backend-agnostic ``libdevice`` namespace and no
    CUDA-only symbol.
``_mlp_up_gemm_gelu_dx_kernel``
    ``dx = gate @ weight``, reduction over ``N`` in ascending chunks.
``_mlp_up_gemm_gelu_dw_kernel``
    ``dW = gate.T @ x``, reduction over the batch rows in ascending ``BLOCK_M``
    chunks *inside one program*, so no atomics and no inter-program reduction
    order exists to vary.
``_mlp_up_gemm_gelu_db_kernel``
    ``db`` as a strict ascending-row FP32 fold: the rows are consumed one at a
    time in index order, which is bit-identical to
    ``left_fold_bias_gradient`` from the reference model.

Batch invariance (mandatory for this row) is the reason every launch
configuration below is a module-level pinned constant and why autotuning is
deliberately absent: ``triton.autotune`` keys on the runtime shapes, so a
different ``M`` would silently pick a different ``BLOCK_K``/``num_warps`` and
therefore a different FP32 accumulation order, and the same logical row would
change bytes with batch size, batch position or launch geometry.  The batch
dimension is also marked ``do_not_specialize`` on every kernel so that a single
compiled binary, and hence a single arithmetic order, serves every ``M``
(including ``M < 16``, where the mask -- not the tile shape -- is what shrinks).
Padding rows are always masked to exact zeros; a zero operand contributes an
exact ``+0`` to the FP32 accumulator, so no padded tile can perturb a valid
row.  Each contraction owns its own pinned tile because the three contractions
have different shapes, but none of the three depends on ``M``.

Byte equality with the CUDA backends, not merely tolerance agreement, is the
design target of this schedule: both sides chain the reduction through
``k``-chunks of 16 into one accumulator per output element, in ascending k
order, and both evaluate the epilogue through the *same* sequence of explicitly
rounded operations (``__fmul_rn``/``__fmaf_rn``/``__nv_tanhf`` on the CUDA side,
``libdevice.mul_rn``/``fma``/``tanh`` here), so the FP32 pre-activation is
byte-equal to the reference's.  ``tanhf`` itself is the operator's single
transcendental: the fp32 CPU reference evaluates it correctly rounded while the
device is within its documented 2 ulp, so ``y``/``gate`` (and therefore the
gradients) are declared-tolerance comparisons while everything upstream of the
tanh is byte-equal.  ``db`` is a strict ascending-row FP32 fold and reproduces
``left_fold_bias_gradient`` bit for bit when the kernel writes FP32, so its BF16
store is exactly one rounding away from the reference fold.

Measured with ``benchmarks/benchmark_mlp_up_gemm_gelu.py`` (shape ``K = 3072``,
``N = 12288``, BF16); the pinned tiles below were chosen offline from a sweep
over block sizes, warps and stages at ``M`` in {4096, 6032, 6889}, and every
candidate in that sweep was byte-identical to the CUDA mma backend, so the
choice is purely a throughput decision.  The accuracy gate in that report runs
in the same process.

bf16 only: this is the model's runtime dtype, and an fp32 call fails closed
instead of silently running an SGEMM-shaped path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    # Backend-agnostic libdevice: the same namespace resolves to the CUDA
    # ``__nv_*`` functions on NVIDIA and to the native math library on ROCm/MUSA,
    # which is what lets the gate kernel be shared across those slots.
    from triton.language.extra import libdevice

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without Triton
    triton = None
    tl = None
    libdevice = None
    _TRITON_AVAILABLE = False

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.utils.logger import logger

# The row contract this backend implements (shared with the CUDA backend).
MLP_UP_GEMM_GELU_CONTRACT = "mlp-up-gemm-gelu-mma"
# Backend-specific provenance recorded by ``record_backward``.
TRITON_BACKEND_IMPL = "triton_mlp_up_gemm_gelu_pinned_config"


@dataclass(frozen=True)
class _TritonTile:
    """One pinned launch configuration. Never derived from ``M``."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


# Pinned. NOT autotuned (autotune selects per-shape configs -> changes the FP32
# reduction order -> breaks batch invariance).
#
# Chosen offline from a sweep over block sizes, warps and stages at S in
# {4096, 6032, 6889}, K = 3072, N = 12288, then re-measured with the bench
# below; the sweep also confirmed that every candidate is byte-identical to the
# CUDA mma backend, so the choice is purely a throughput decision.  The one
# constant a re-sweep moved for this row's (transposed) shape is the forward's
# ``num_stages``: 4 beats the down row's 3 by 7.9% under a paired A/B rotation
# (median A/B 0.921 over 30 pairs, A faster in 29 of them), which is also how the
# 5-9% run-to-run spread of the development host was ruled out.
_FORWARD_TILE = _TritonTile(block_m=128, block_n=256, block_k=64, num_warps=8, num_stages=4)
_DX_TILE = _TritonTile(block_m=128, block_n=128, block_k=128, num_warps=4, num_stages=3)
# ``block_m`` is the reduction chunk over the batch rows; the other two are the
# ``(N, K)`` output tile of ``dW``.  ``dW`` is the short-reduction/huge-output
# contraction, so it wants a narrow reduction chunk and a wide output tile.
_DW_TILE = _TritonTile(block_m=32, block_n=256, block_k=128, num_warps=8, num_stages=4)
# ``db`` has no ``tl.dot``; ``block_m`` is the statically unrolled fold width and
# does not affect the (strictly ascending) association order.
_DB_TILE = _TritonTile(block_m=64, block_n=128, block_k=1, num_warps=4, num_stages=1)
# The gate is elementwise: ``block_m`` is a flat block of elements and, like the
# other tiles, carries no ``M`` dependence and no reduction order.
_GATE_TILE = _TritonTile(block_m=1024, block_n=1, block_k=1, num_warps=4, num_stages=1)

#: GELU constants as the fp32 bit patterns the CUDA header
#: (``csrc/cuda/gemm/mlp_up_gemm_gelu_math.cuh``) and the fp32 CPU reference use:
#: ``_GELU_C = 0.044715``, ``_GELU_S = sqrt(2 / pi)`` (PyTorch's tanh-GELU
#: ``kBeta = M_SQRT2 * M_2_SQRTPI * 0.5``), ``_GELU_K3 = 3 * _GELU_C``.  Triton
#: folds a Python float constant into the fp32 type, so these are the same
#: literals the device kernels compile.
_GELU_C = 0.044715
_GELU_S = 0.7978845608028654
_GELU_K3 = 0.134145


if _TRITON_AVAILABLE:

    @triton.jit
    def _mlp_up_gemm_gelu_tanh_fp32(pre, C: tl.constexpr, S: tl.constexpr):
        """Frozen fp32 tanh-approximate GELU of an fp32 pre-activation.

        ``C`` / ``S`` are the module-level ``_GELU_C`` / ``_GELU_S``, passed in
        as ``tl.constexpr`` because Triton 3.6 cannot resolve module-level scalar
        globals inside a jit body; they fold to the identical fp32 literals.

        Operation for operation the sequence pinned in the spec /
        ``mlp_up_gemm_gelu_math.cuh`` -- every step is an explicitly rounded
        libdevice call so the compiler can neither reassociate nor contract it::

            q  = C * x
            t  = fma(q, x * x, x)          x + 0.044715 x^3
            th = tanh(S * t)
            y  = (0.5 * x) * (1 + th)
        """

        q = libdevice.mul_rn(C, pre)
        t = libdevice.fma(q, libdevice.mul_rn(pre, pre), pre)
        th = libdevice.tanh(libdevice.mul_rn(S, t))
        return libdevice.mul_rn(libdevice.mul_rn(0.5, pre), libdevice.add_rn(1.0, th))

    @triton.jit
    def _mlp_up_gemm_gelu_tanh_grad_fp32(pre, C: tl.constexpr, S: tl.constexpr, K3: tl.constexpr):
        """Frozen fp32 derivative of :func:`_mlp_up_gemm_gelu_tanh_fp32`.

        ``C`` / ``S`` / ``K3`` are the module-level ``_GELU_C`` / ``_GELU_S`` /
        ``_GELU_K3`` threaded in as ``tl.constexpr`` (Triton 3.6 cannot resolve
        module-level scalar globals inside a jit body).

        The backward's gate coefficient, on the fp32 pre-activation::

            a  = 1 + th
            b  = 1 - th * th
            e  = fma(K3, x * x, 1)
            d  = fma(0.5, a, 0.5 * S * x * b * e)
        """

        x2 = libdevice.mul_rn(pre, pre)
        q = libdevice.mul_rn(C, pre)
        t = libdevice.fma(q, x2, pre)
        th = libdevice.tanh(libdevice.mul_rn(S, t))
        a = libdevice.add_rn(1.0, th)
        # ``sub_rn(1, th * th)`` expressed through the fma with a -1 multiplier:
        # the product is exact, so the fused multiply-add is the single rounding
        # of ``1 - th * th``, i.e. the identical operation.
        b = libdevice.fma(-1.0, libdevice.mul_rn(th, th), 1.0)
        e = libdevice.fma(x2, K3, 1.0)
        h = libdevice.mul_rn(libdevice.mul_rn(libdevice.mul_rn(S, pre), b), e)
        return libdevice.fma(0.5, a, libdevice.mul_rn(0.5, h))

    @triton.jit(do_not_specialize=["M"])
    def _mlp_up_gemm_gelu_forward_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        out_ptr,
        pre_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        stride_pm,
        stride_pn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        EMIT_PRE: tl.constexpr,
        GELU_C: tl.constexpr,
        GELU_S: tl.constexpr,
    ):
        # One program = one output tile, walks the whole K in fixed ascending
        # BLOCK_K chunks.  No split-K -> the accumulation order of a logical row
        # does not depend on M, on the grid, or on which tile holds it.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        # weight is [N, K]: the reduction dim is the fast axis, so a [BLOCK_K,
        # BLOCK_N] tile is a plain (k-major) load of the same rows.
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_rem = K - k * BLOCK_K
            a = tl.load(
                x_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem),
                other=0.0,
            )
            b = tl.load(
                w_ptrs,
                mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N),
                other=0.0,
            )
            # BF16 x BF16 products are exact in FP32; "ieee" pins the intent
            # (never TF32, never a reduced-precision input path).
            acc += tl.dot(a, b, input_precision="ieee")
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
        if EMIT_PRE:
            # The pre-activation (GEMM + bias, *before* the GELU) is the backward's
            # gate operand; it is written from this same accumulator, so there is
            # no second reduction pass and no second rounding between the two.
            # ``EMIT_PRE=False`` instantiates this kernel without the store, and
            # the launcher then allocates no buffer for it.
            pre_ptrs = pre_ptr + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
            tl.store(pre_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
        act = _mlp_up_gemm_gelu_tanh_fp32(acc, GELU_C, GELU_S)
        c = act.to(out_ptr.dtype.element_ty)
        c_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.jit(do_not_specialize=["numel"])
    def _mlp_up_gemm_gelu_gate_kernel(
        g_ptr,
        pre_ptr,
        out_ptr,
        numel,
        BLOCK: tl.constexpr,
        GELU_C: tl.constexpr,
        GELU_S: tl.constexpr,
        GELU_K3: tl.constexpr,
    ):
        # gate = bf16(fp32(grad) * gelu'(pre)): pure elementwise, no reduction,
        # hence no association order to pin.  Uses the backend-agnostic libdevice
        # namespace, so the ROCm/MUSA slot shares this exact kernel.
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        grad = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        pre = tl.load(pre_ptr + offs, mask=mask, other=0.0)
        gate = libdevice.mul_rn(
            grad, _mlp_up_gemm_gelu_tanh_grad_fp32(pre, GELU_C, GELU_S, GELU_K3)
        )
        tl.store(out_ptr + offs, gate.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit(do_not_specialize=["M"])
    def _mlp_up_gemm_gelu_dx_kernel(
        g_ptr,
        w_ptr,
        dx_ptr,
        M,
        K: tl.constexpr,
        N: tl.constexpr,
        stride_gm,
        stride_gn,
        stride_wn,
        stride_wk,
        stride_dm,
        stride_dk,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # dx = gate @ weight: the reduction runs over N in ascending BLOCK_N
        # chunks, i.e. the same fixed order for every M.
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_r = tl.arange(0, BLOCK_N)
        g_ptrs = g_ptr + offs_m[:, None] * stride_gm + offs_r[None, :] * stride_gn
        w_ptrs = w_ptr + offs_r[:, None] * stride_wn + offs_k[None, :] * stride_wk
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for r in range(0, tl.cdiv(N, BLOCK_N)):
            r_rem = N - r * BLOCK_N
            a = tl.load(
                g_ptrs,
                mask=(offs_m[:, None] < M) & (offs_r[None, :] < r_rem),
                other=0.0,
            )
            b = tl.load(
                w_ptrs,
                mask=(offs_r[:, None] < r_rem) & (offs_k[None, :] < K),
                other=0.0,
            )
            acc += tl.dot(a, b, input_precision="ieee")
            g_ptrs += BLOCK_N * stride_gn
            w_ptrs += BLOCK_N * stride_wn
        c_ptrs = dx_ptr + offs_m[:, None] * stride_dm + offs_k[None, :] * stride_dk
        tl.store(
            c_ptrs,
            acc.to(dx_ptr.dtype.element_ty),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        )

    @triton.jit(do_not_specialize=["M"])
    def _mlp_up_gemm_gelu_dw_kernel(
        g_ptr,
        x_ptr,
        dw_ptr,
        M,
        K: tl.constexpr,
        N: tl.constexpr,
        stride_gm,
        stride_gn,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # dW = gate.T @ x.  One program owns a full (BLOCK_N, BLOCK_K) output
        # tile and walks the batch rows itself in ascending BLOCK_M chunks:
        # no atomics, no cross-program reduction, no split over the reduction
        # dim, so dW[n, k] has one fixed accumulation order for every M.
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_m = tl.arange(0, BLOCK_M)
        g_ptrs = g_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        n_in = offs_n < N
        k_in = offs_k < K
        for m0 in range(0, M, BLOCK_M):
            m_in = ((m0 + offs_m) < M)[:, None]
            a = tl.load(g_ptrs, mask=m_in & n_in[None, :], other=0.0)
            b = tl.load(x_ptrs, mask=m_in & k_in[None, :], other=0.0)
            acc += tl.dot(tl.trans(a), b, input_precision="ieee")
            g_ptrs += BLOCK_M * stride_gm
            x_ptrs += BLOCK_M * stride_xm
        c_ptrs = dw_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        tl.store(c_ptrs, acc.to(dw_ptr.dtype.element_ty), mask=n_in[:, None] & k_in[None, :])

    @triton.jit(do_not_specialize=["M"])
    def _mlp_up_gemm_gelu_db_kernel(
        g_ptr,
        db_ptr,
        M,
        N: tl.constexpr,
        stride_gm,
        stride_gn,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        # db = the ascending-row FP32 fold of gate, one row at a time, exactly
        # the association order of ``left_fold_bias_gradient``.
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for m0 in range(0, M, BLOCK_M):
            base = g_ptr + m0 * stride_gm + offs_n * stride_gn
            for i in tl.static_range(0, BLOCK_M):
                row = tl.load(
                    base + i * stride_gm,
                    mask=n_mask & (m0 + i < M),
                    other=0.0,
                )
                acc += row.to(tl.float32)
        tl.store(db_ptr + offs_n, acc.to(db_ptr.dtype.element_ty), mask=n_mask)


def _require_2d_bf16(x: torch.Tensor, weight: torch.Tensor) -> None:
    """Validate the operand contract shared with the CUDA backend."""

    supported_devices = ("cuda", "hip", "xpu", "musa")
    if x.device.type not in supported_devices or weight.device.type not in supported_devices:
        raise RuntimeError(
            "TritonMlpUpGemmGeluOp requires accelerator tensors (CUDA / ROCm / XPU / MUSA)"
        )
    if x.device != weight.device:
        raise ValueError("x and weight must live on one device")
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        raise ValueError(
            "mlp_up_gemm_gelu is bf16 only on the kernel paths: x/weight "
            f"required bf16, got x={x.dtype}, weight={weight.dtype}"
        )
    if x.dim() < 1 or weight.dim() != 2:
        raise ValueError("mlp_up_gemm_gelu expects x [*, K] and weight [N, K]")
    if x.size(-1) != weight.size(-1):
        raise ValueError("x K must match weight K")


def _check_bias(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
    if bias is None:
        return
    if bias.device != x.device or bias.dtype is not torch.bfloat16:
        raise ValueError("bias must be bf16 on the same device as x")
    if bias.dim() != 1 or bias.size(0) != weight.size(0):
        raise ValueError("bias length must match weight N")


def _launch_forward(
    x2d: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    emit_pre: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward launch.  Returns ``(out_bf16, pre)``.

    ``pre`` is the fp32 pre-activation ``[M, N]`` (GEMM + bias, before the GELU)
    when ``emit_pre`` is true; otherwise it is the empty ``fp32`` placeholder of
    the operator contract and no buffer is allocated for it inside the kernel's
    instantiation.
    """

    tile = _FORWARD_TILE
    rows, k_dim = x2d.shape
    n_dim = weight.size(0)
    out = torch.empty((rows, n_dim), dtype=torch.bfloat16, device=x2d.device)
    if emit_pre:
        pre = torch.empty((rows, n_dim), dtype=torch.float32, device=x2d.device)
        pre_arg, stride_pm, stride_pn = pre, pre.stride(0), pre.stride(1)
    else:
        # ``EMIT_PRE=False`` has no store at all, so the placeholder pointer is
        # never dereferenced; only the (unused) strides have to be well-formed.
        pre = torch.empty((0,), dtype=torch.float32, device=x2d.device)
        pre_arg, stride_pm, stride_pn = x2d, x2d.stride(0), x2d.stride(1)
    grid = (triton.cdiv(rows, tile.block_m), triton.cdiv(n_dim, tile.block_n))
    # The kernel indexes the bias at unit stride, so a view has to be made
    # contiguous here -- exactly as the CUDA backend's ``bias->to(kFloat)`` does.
    # Passing a strided bias through would read the wrong elements and silently
    # return wrong values and gradients.
    bias_arg = bias.contiguous() if bias is not None else x2d
    _mlp_up_gemm_gelu_forward_kernel[grid](
        x2d,
        weight,
        bias_arg,
        out,
        pre_arg,
        rows,
        N=n_dim,
        K=k_dim,
        stride_xm=x2d.stride(0),
        stride_xk=x2d.stride(1),
        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),
        stride_om=out.stride(0),
        stride_on=out.stride(1),
        stride_pm=stride_pm,
        stride_pn=stride_pn,
        BLOCK_M=tile.block_m,
        BLOCK_N=tile.block_n,
        BLOCK_K=tile.block_k,
        HAS_BIAS=bias is not None,
        EMIT_PRE=emit_pre,
        GELU_C=_GELU_C,
        GELU_S=_GELU_S,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
    )
    return out, pre


def _launch_gate(grad2d: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """``gate = bf16(fp32(grad) * gelu'(pre))``; both operands must be contiguous."""

    tile = _GATE_TILE
    rows, n_dim = grad2d.shape
    gate = torch.empty((rows, n_dim), dtype=torch.bfloat16, device=grad2d.device)
    numel = rows * n_dim
    grid = (triton.cdiv(numel, tile.block_m),)
    _mlp_up_gemm_gelu_gate_kernel[grid](
        grad2d,
        pre,
        gate,
        numel,
        BLOCK=tile.block_m,
        GELU_C=_GELU_C,
        GELU_S=_GELU_S,
        GELU_K3=_GELU_K3,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
    )
    return gate


def _launch_dx(grad2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    tile = _DX_TILE
    rows, n_dim = grad2d.shape
    k_dim = weight.size(1)
    dx = torch.empty((rows, k_dim), dtype=torch.bfloat16, device=grad2d.device)
    grid = (triton.cdiv(rows, tile.block_m), triton.cdiv(k_dim, tile.block_k))
    _mlp_up_gemm_gelu_dx_kernel[grid](
        grad2d,
        weight,
        dx,
        rows,
        K=k_dim,
        N=n_dim,
        stride_gm=grad2d.stride(0),
        stride_gn=grad2d.stride(1),
        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),
        stride_dm=dx.stride(0),
        stride_dk=dx.stride(1),
        BLOCK_M=tile.block_m,
        BLOCK_N=tile.block_n,
        BLOCK_K=tile.block_k,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
    )
    return dx


def _launch_dw(grad2d: torch.Tensor, x2d: torch.Tensor) -> torch.Tensor:
    tile = _DW_TILE
    rows, n_dim = grad2d.shape
    k_dim = x2d.size(1)
    dw = torch.empty((n_dim, k_dim), dtype=torch.bfloat16, device=grad2d.device)
    grid = (triton.cdiv(n_dim, tile.block_n), triton.cdiv(k_dim, tile.block_k))
    _mlp_up_gemm_gelu_dw_kernel[grid](
        grad2d,
        x2d,
        dw,
        rows,
        K=k_dim,
        N=n_dim,
        stride_gm=grad2d.stride(0),
        stride_gn=grad2d.stride(1),
        stride_xm=x2d.stride(0),
        stride_xk=x2d.stride(1),
        stride_wn=dw.stride(0),
        stride_wk=dw.stride(1),
        BLOCK_M=tile.block_m,
        BLOCK_N=tile.block_n,
        BLOCK_K=tile.block_k,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
    )
    return dw


def _launch_db(grad2d: torch.Tensor) -> torch.Tensor:
    tile = _DB_TILE
    rows, n_dim = grad2d.shape
    db = torch.empty((n_dim,), dtype=torch.bfloat16, device=grad2d.device)
    grid = (triton.cdiv(n_dim, tile.block_n),)
    _mlp_up_gemm_gelu_db_kernel[grid](
        grad2d,
        db,
        rows,
        N=n_dim,
        stride_gm=grad2d.stride(0),
        stride_gn=grad2d.stride(1),
        BLOCK_N=tile.block_n,
        BLOCK_M=tile.block_m,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
    )
    return db


class _TritonMlpUpGemmGeluFunction(torch.autograd.Function):
    """Forward/backward through the pinned Triton tiles.

    The forward additionally materializes the fp32 pre-activation whenever a
    gradient could be requested; the backward's gate is formed from it and the
    ``dx``/``dW``/``db`` contractions consume that bf16 gate.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        emit_pre: bool,
    ) -> torch.Tensor:
        x2d = x.reshape(-1, x.size(-1)).contiguous()
        weight = weight.contiguous()
        # ``pre`` is only needed if some input may receive a gradient; the GELU
        # arithmetic (and therefore ``y``) is identical either way. The flag is
        # decided at the call site because torch runs a Function's forward with
        # grad mode disabled, where ``torch.is_grad_enabled()`` is always False.
        out, pre = _launch_forward(x2d, weight, bias, emit_pre=emit_pre)
        ctx.save_for_backward(x2d, weight, bias, pre)
        ctx.lead_shape = x.shape[:-1]
        return out.reshape(*ctx.lead_shape, weight.size(0))

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], None]:
        x2d, weight, bias, pre = ctx.saved_tensors
        if pre.numel() == 0:
            raise RuntimeError(
                "mlp_up_gemm_gelu: the forward ran without materializing the fp32 "
                "pre-activation (it was inferred that no gradient could be requested), "
                "so the GELU gate cannot be formed; re-run the forward under grad mode"
            )
        grad_2d = grad_output.reshape(-1, weight.size(0))
        if grad_2d.dtype is not torch.bfloat16:
            # Autograd hands back the forward dtype; anything else (a hand-built
            # fp32 cotangent) is rounded once before it enters the kernels.
            grad_2d = grad_2d.to(torch.bfloat16)
        grad_2d = grad_2d.contiguous()
        # The gate is the activation's gradient at the gradient dtype boundary;
        # dx/dW/db all consume this one bf16 tensor.
        gate = _launch_gate(grad_2d, pre)

        grad_x: Optional[torch.Tensor] = None
        grad_w: Optional[torch.Tensor] = None
        grad_b: Optional[torch.Tensor] = None
        if ctx.needs_input_grad[0]:
            grad_x = _launch_dx(gate, weight).reshape(*ctx.lead_shape, weight.size(1))
        if ctx.needs_input_grad[1]:
            grad_w = _launch_dw(gate, x2d)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_b = _launch_db(gate).to(bias.dtype)
        record_backward(
            "mlp_up_gemm_gelu",
            kernel_id=MLP_UP_GEMM_GELU_CONTRACT,
            impl=TRITON_BACKEND_IMPL,
            family="triton",
        )
        return grad_x, grad_w, grad_b, None


def _needs_pre(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> bool:
    """Whether the forward has to materialize the fp32 pre-activation.

    The backward's gate needs it, so it is written only when a gradient can
    actually be requested; an inference-only call (``torch.no_grad()``, or a
    frozen model) skips the store entirely. The forward *bytes* do not depend on
    this decision -- only on whether the extra fp32 tensor exists. Evaluated at
    the call site because torch runs a Function's forward with grad mode
    disabled, where ``torch.is_grad_enabled()`` is always ``False``.
    """

    if not torch.is_grad_enabled():
        return False
    if x.requires_grad or weight.requires_grad:
        return True
    return bias is not None and bias.requires_grad


class TritonMlpUpGemmGeluOp:
    """Triton backend for the row contract: pinned tiles, bf16 only."""

    op_class = "reduction"
    is_batch_invariant = True  # per pinned tiles; verified by tests/test_mlp_up_gemm_gelu_triton.py
    backward_impl = TRITON_BACKEND_IMPL

    def __init__(self) -> None:
        if not _TRITON_AVAILABLE:
            raise RuntimeError(
                "Triton is not importable; TritonMlpUpGemmGeluOp has no fallback "
                "(a non-pinned schedule would break the row's batch invariance)."
            )
        logger.info(
            "TritonMlpUpGemmGeluOp ready (pinned tiles: fwd %s, dx %s, dw %s).",
            _FORWARD_TILE,
            _DX_TILE,
            _DW_TILE,
        )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(x, weight, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _require_2d_bf16(x, weight)
        _check_bias(x, weight, bias)
        return _TritonMlpUpGemmGeluFunction.apply(x, weight, bias, _needs_pre(x, weight, bias))

    def _forward_with_pre(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward entry that also returns the fp32 pre-activation (parity tests).

        Runs the same pinned schedule with ``emit_pre=True``; the returned
        pre-activation is what the forward kernel wrote before the GELU, i.e.
        exactly what the backward's gate consumes.
        """

        x2d = x.reshape(-1, x.size(-1)).contiguous()
        w = weight.contiguous()
        b = None if bias is None else bias.contiguous()
        out, pre = _launch_forward(x2d, w, b, emit_pre=True)
        return out.reshape(*x.shape[:-1], weight.size(0)), pre

    def _grads_for_test(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        grad_output: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
        pre: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward entry used by the byte-equality tests against the spec.

        When ``pre`` is supplied it is used verbatim as the gate's pre-activation
        (the tests' way of isolating the transcendental out of the comparison);
        otherwise the backward is driven through autograd like any other call.
        """

        if pre is not None:
            x2d = x.reshape(-1, x.size(-1)).contiguous()
            w = weight.contiguous()
            grad_2d = grad_output.reshape(-1, weight.size(0)).to(torch.bfloat16).contiguous()
            gate = _launch_gate(grad_2d, pre.contiguous())
            grad_x = _launch_dx(gate, w).reshape(*x.shape[:-1], weight.size(1))
            grad_w = _launch_dw(gate, x2d)
            return grad_x, grad_w
        x = x.detach().requires_grad_(True)
        weight = weight.detach().requires_grad_(True)
        self.forward(x, weight, bias=bias).backward(grad_output)
        return x.grad, weight.grad
