# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton backend for the Qwen-Image MLP down projection (WS1).

``y = bf16(fp32_accum(x @ weight.T) + bias)``
with a fixed K reduction order, an FP32 accumulator, no split-K, no atomics, the
bias added once in FP32 after the complete reduction and exactly one BF16 cast at
the store.  Four Triton kernels implement the operator and its VJP:

``_mlp_down_gemm_forward_kernel``
    one program per ``(BLOCK_M, BLOCK_N)`` output tile walks the whole K in
    ascending ``BLOCK_K`` chunks (``tl.dot(x_tile, w_tile)``, BF16 inputs, FP32
    accumulator, ``input_precision="ieee"`` so no TF32 path is ever selected).
``_mlp_down_gemm_dx_kernel``
    ``dx = dout @ weight``, reduction over ``N`` in ascending chunks.
``_mlp_down_gemm_dw_kernel``
    ``dW = dout.T @ x``, reduction over the batch rows in ascending ``BLOCK_M``
    chunks *inside one program*, so no atomics and no inter-program reduction
    order exists to vary.
``_mlp_down_gemm_db_kernel``
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

Byte equality with the CUDA backend, not merely tolerance agreement, is a
reachable and measured property of this schedule: both sides chain the
reduction through k-chunks of 16 into one accumulator per output element, in
ascending k order.  The dumped PTX for the shipped tiles shows Triton on SM90
issuing ``wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16`` (forward) and
``wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16`` (``dx``, ``dW``), the
sub-chunks of one ``BLOCK_K`` chunk all writing the same accumulator register
set inside the ascending K loop, with no k-split across warps or warpgroups;
below ``BLOCK_M = 64`` Triton falls back to
``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`` -- and that
configuration is byte-identical too.  Both
lowerings were measured to produce the *same bytes* as the CUDA Hopper
``wgmma`` chain, at every tile/warp/stage setting swept (128 forward, 32 ``dx``, 16
``dW``, 15 ``db`` configurations, the forward at ``S`` in {1, 7, 129, 512, 4096,
6889}).  ``db`` is a strict ascending-row FP32 fold and reproduces
``left_fold_bias_gradient`` bit for bit when the kernel writes FP32, so its BF16
store is exactly one rounding away from the reference fold.

Measured with ``benchmarks/operators/gemm/benchmark_mlp_down_gemm.py`` (the accuracy gate in
that report passes in the same run) on the pinned configurations (H100 PCIe,
BF16, ``K = 12288``, ``N = 3072``, Triton 3.6):

===========  =========  ===============  =============  ===============
tokens       fwd ms     fwd TFLOP/s      fwd+bwd ms     fwd+bwd TFLOP/s
===========  =========  ===============  =============  ===============
4096         0.656      471.5            2.445          379.4
6889         1.075      483.7            4.238          368.2
===========  =========  ===============  =============  ===============

Against the independent fp32 CPU reference at ``K = 12288`` (256-row gate, the
same numbers at both token counts): forward 99.32% of elements bit-identical to
the correctly-rounded bf16 reference and every element within 0.83 bf16-ulp of
max|reference|; ``dx`` 99.82%/1.19 ulp; ``dW`` 99.99%/1.26 ulp; ``db`` 100%/0
ulp.  The single bf16 cast moves the stored value by at most 0.81 ulp from the
raw fp32 result, i.e. under half of the declared 8-ulp budget.

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

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without Triton
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.utils.logger import logger

# The row contract this backend implements (shared with the CUDA backend).
MLP_DOWN_GEMM_CONTRACT = "mlp-down-gemm-mma"
# Backend-specific provenance recorded by ``record_backward``.
TRITON_BACKEND_IMPL = "triton_mlp_down_gemm_pinned_config"


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
# {4096, 6889}, K = 12288, N = 3072, then re-measured with the bench below; the
# sweep also confirmed that every candidate is byte-identical to the CUDA mma
# backend, so the choice is purely a throughput decision.
_FORWARD_TILE = _TritonTile(block_m=128, block_n=256, block_k=64, num_warps=8, num_stages=3)
_DX_TILE = _TritonTile(block_m=128, block_n=128, block_k=128, num_warps=4, num_stages=3)
# ``block_m`` is the reduction chunk over the batch rows; the other two are the
# ``(N, K)`` output tile of ``dW``.  ``dW`` is the short-reduction/huge-output
# contraction, so it wants a narrow reduction chunk and a wide output tile.
_DW_TILE = _TritonTile(block_m=32, block_n=256, block_k=128, num_warps=8, num_stages=4)
# ``db`` has no ``tl.dot``; ``block_m`` is the statically unrolled fold width and
# does not affect the (strictly ascending) association order.
_DB_TILE = _TritonTile(block_m=64, block_n=128, block_k=1, num_warps=4, num_stages=1)


if _TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=["M"])
    def _mlp_down_gemm_forward_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        out_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
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
        c = acc.to(out_ptr.dtype.element_ty)
        c_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.jit(do_not_specialize=["M"])
    def _mlp_down_gemm_dx_kernel(
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
        # dx = dout @ weight: the reduction runs over N in ascending BLOCK_N
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
    def _mlp_down_gemm_dw_kernel(
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
        # dW = dout.T @ x.  One program owns a full (BLOCK_N, BLOCK_K) output
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
    def _mlp_down_gemm_db_kernel(
        g_ptr,
        db_ptr,
        M,
        N: tl.constexpr,
        stride_gm,
        stride_gn,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        # db = the ascending-row FP32 fold of dout, one row at a time, exactly
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
            "TritonMlpDownGemmOp requires accelerator tensors (CUDA / ROCm / XPU / MUSA)"
        )
    if x.device != weight.device:
        raise ValueError("x and weight must live on one device")
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        raise ValueError(
            "mlp_down_gemm is bf16 only on the kernel paths: x/weight "
            f"required bf16, got x={x.dtype}, weight={weight.dtype}"
        )
    if x.dim() < 1 or weight.dim() != 2:
        raise ValueError("mlp_down_gemm expects x [*, K] and weight [N, K]")
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
    x2d: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    tile = _FORWARD_TILE
    rows, k_dim = x2d.shape
    n_dim = weight.size(0)
    out = torch.empty((rows, n_dim), dtype=torch.bfloat16, device=x2d.device)
    grid = (triton.cdiv(rows, tile.block_m), triton.cdiv(n_dim, tile.block_n))
    # The kernel indexes the bias at unit stride, so a view has to be made
    # contiguous here -- exactly as the CUDA backend's ``bias->to(kFloat)`` does.
    # Passing a strided bias through would read the wrong elements and silently
    # return wrong values and gradients.
    bias_arg = bias.contiguous() if bias is not None else x2d
    _mlp_down_gemm_forward_kernel[grid](
        x2d,
        weight,
        bias_arg,
        out,
        rows,
        N=n_dim,
        K=k_dim,
        stride_xm=x2d.stride(0),
        stride_xk=x2d.stride(1),
        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),
        stride_om=out.stride(0),
        stride_on=out.stride(1),
        BLOCK_M=tile.block_m,
        BLOCK_N=tile.block_n,
        BLOCK_K=tile.block_k,
        HAS_BIAS=bias is not None,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
    )
    return out


def _launch_dx(grad2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    tile = _DX_TILE
    rows, n_dim = grad2d.shape
    k_dim = weight.size(1)
    dx = torch.empty((rows, k_dim), dtype=torch.bfloat16, device=grad2d.device)
    grid = (triton.cdiv(rows, tile.block_m), triton.cdiv(k_dim, tile.block_k))
    _mlp_down_gemm_dx_kernel[grid](
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
    _mlp_down_gemm_dw_kernel[grid](
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
    _mlp_down_gemm_db_kernel[grid](
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


class _TritonMlpDownGemmFunction(torch.autograd.Function):
    """Forward/backward through the pinned Triton tiles."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x2d = x.reshape(-1, x.size(-1)).contiguous()
        weight = weight.contiguous()
        out = _launch_forward(x2d, weight, bias)
        ctx.save_for_backward(x2d, weight, bias)
        ctx.lead_shape = x.shape[:-1]
        return out.reshape(*ctx.lead_shape, weight.size(0))

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        x2d, weight, bias = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, weight.size(0))
        if grad_2d.dtype is not torch.bfloat16:
            # Autograd hands back the forward dtype; anything else (a hand-built
            # fp32 cotangent) is rounded once before it enters the kernels.
            grad_2d = grad_2d.to(torch.bfloat16)
        grad_2d = grad_2d.contiguous()

        grad_x: Optional[torch.Tensor] = None
        grad_w: Optional[torch.Tensor] = None
        grad_b: Optional[torch.Tensor] = None
        if ctx.needs_input_grad[0]:
            grad_x = _launch_dx(grad_2d, weight).reshape(*ctx.lead_shape, weight.size(1))
        if ctx.needs_input_grad[1]:
            grad_w = _launch_dw(grad_2d, x2d)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_b = _launch_db(grad_2d).to(bias.dtype)
        record_backward(
            "mlp_down_gemm",
            kernel_id=MLP_DOWN_GEMM_CONTRACT,
            impl=TRITON_BACKEND_IMPL,
            family="triton",
        )
        return grad_x, grad_w, grad_b


class TritonMlpDownGemmOp:
    """Triton backend for the row contract: pinned tiles, bf16 only."""

    op_class = "reduction"
    is_batch_invariant = True  # per pinned tiles; verified by tests/ops/gemm/test_mlp_down_gemm_triton.py
    backward_impl = TRITON_BACKEND_IMPL

    def __init__(self) -> None:
        if not _TRITON_AVAILABLE:
            raise RuntimeError(
                "Triton is not importable; TritonMlpDownGemmOp has no fallback "
                "(a non-pinned schedule would break the row's batch invariance)."
            )
        logger.info(
            "TritonMlpDownGemmOp ready (pinned tiles: fwd %s, dx %s, dw %s).",
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
        return _TritonMlpDownGemmFunction.apply(x, weight, bias)

    def _grads_for_test(
        self, x: torch.Tensor, weight: torch.Tensor, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward entry used by the invariance/accuracy tests."""

        x = x.detach().requires_grad_(True)
        weight = weight.detach().requires_grad_(True)
        self.forward(x, weight).backward(grad_output)
        return x.grad, weight.grad
