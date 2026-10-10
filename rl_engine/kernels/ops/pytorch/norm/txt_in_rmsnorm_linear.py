# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Qwen-Image WS1 txt_in_rmsnorm_linear: RMSNorm(3584) + Linear(3584->3072)+bias.

Contract ``txt-in-rmsnorm-linear-v1`` (frozen v3.6): the RMSNorm->Linear
internal seam is FP32 with no intermediate cast; a single RNE cast happens at
the output boundary. Reference, CUDA and Triton run one discipline; outputs
and gradients are compared bitwise per the contract.

Forward (every step a single correctly-rounded FP32 op):

  sumsq = Tree112(x*x)  112 wide-32 leaves, ascending-h addcmul chains
                        (addcmul == correctly-rounded FMA, fmaf-oracle
                        verified), mid-split tree merges (plain fp32 adds)
  var   = sumsq / 3584  true division, never reciprocal-multiply (torch CUDA
                        tensor/scalar silently reciprocal-multiplies, so the
                        division runs through ``true_div_rn`` -- 0-dim tensor
                        divisor, tensor/tensor kernel, correctly rounded)
  rstd  = three-step sqrt: t = var + eps32; sq = fp32(sqrt_fp64(t)) -- only
                        the sqrt step goes through fp64 (innocuous double
                        rounding, empirically exact vs sqrtf); computing the
                        whole 1/sqrt in fp64 is forbidden (1471/5000 sampled
                        bit differences -- it skips the sq32 rounding);
                        rstd = 1.0f / sq in fp32
  xhat  = x * rstd ;  u = xhat * gamma (left-assoc isolated muls);
  z     = u (frozen FP32 seam, no cast)
  y     = cast(TreeGEMM(z, Wf) + bf)  -- the single output cast

Backward: dz = TreeN(g, Wf.t()) (same mid-split tree over N); du := dz
(plain identity -- no internal cast, no STE); dxhat = du*gamma; dot = TreeH
of FMA(dxhat, xhat); t1 = dot/3584; t2 = xhat*t1; t3 = dxhat-t2 (isolated
ops -- contraction into FMS forbidden); dx = rstd*t3; dgamma/dW = ascending
row FMA left folds; db = pure-add fold; each gradient cast once at return.
"""

from __future__ import annotations

import struct
from typing import Optional

import torch

from rl_engine.kernels.ops.pytorch.linear.attn_out_bias_gemm import (
    finalize_tree_output,
    tree_gemm_fp32,
)
from rl_engine.utils.logger import logger

TXT_IN_RMSNORM_LINEAR_CONTRACT_VERSION = "txt-in-rmsnorm-linear-v1"
TXT_IN_HIDDEN = 3584
TXT_IN_OUT = 3072

# Fixed FP32 bit patterns (single rounding of the decimal constants; every
# backend must materialise the same bits).
EPS32 = struct.unpack("<f", struct.pack("<I", 0x358637BD))[0]  # 1e-6
DIVISOR_F32 = 3584.0  # exactly representable in FP32 (0x45600000)


def true_div_rn(t: torch.Tensor, scalar: float) -> torch.Tensor:
    """Correctly-rounded division by a scalar, on EVERY device.

    torch CUDA ``tensor / python_float`` silently multiplies by the
    single-rounded reciprocal instead of dividing (CPU divides): the two
    devices disagree on ~55% of random samples, violating the contract's
    "true division, never reciprocal-multiply" rule. A device-resident 0-dim
    tensor divisor keeps the tensor/tensor division kernel, which is
    correctly rounded on every backend (verified: cpu == cuda == div.rn over
    2^20 random samples; see the regression tests).
    """
    return t / torch.tensor(scalar, dtype=torch.float32, device=t.device)


def row_sumsq_tree(x_f: torch.Tensor) -> torch.Tensor:
    """sumsq[s] = Tree(x_f[s, :]^2) -- 112 leaves of width 32 (short tail
    allowed), ascending-h addcmul chains, mid-split fp32 adds."""
    rows, hidden = x_f.shape
    leaf_count = (hidden + 31) // 32

    def leaf_sum(leaf: int) -> torch.Tensor:
        h0 = leaf * 32
        h1 = min(h0 + 32, hidden)
        acc = torch.zeros(rows, device=x_f.device, dtype=torch.float32)
        for h in range(h0, h1):
            col = x_f[:, h]
            acc = torch.addcmul(acc, col, col)
        return acc

    def reduce_range(lo: int, hi: int) -> torch.Tensor:
        if hi - lo == 1:
            return leaf_sum(lo)
        mid = lo + (hi - lo) // 2
        return reduce_range(lo, mid) + reduce_range(mid, hi)

    return reduce_range(0, leaf_count)


def three_step_rstd(var: torch.Tensor) -> torch.Tensor:
    """rstd under the frozen three-step sqrt form (see module docstring)."""
    t32 = var + EPS32
    sq32 = torch.sqrt(t32.double()).float()
    return 1.0 / sq32


def txt_in_norm_stats(x_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen norm statistics: returns (xhat, rstd), both fp32 [S, H]/[S]."""
    sumsq = row_sumsq_tree(x_f)
    var = true_div_rn(sumsq, DIVISOR_F32)
    rstd = three_step_rstd(var)
    xhat = x_f * rstd.unsqueeze(1)
    return xhat, rstd


def txt_in_row_dot_tree(a_f: torch.Tensor, b_f: torch.Tensor) -> torch.Tensor:
    """Per-row dot under the frozen H-dim FMA tree: leaf chains accumulate
    ``acc = addcmul(acc, a_col, b_col)`` (never ``p = a*b`` then sum), leaf
    merges are plain fp32 adds in mid-split order. a_f/b_f: [S, H]."""
    rows, hidden = a_f.shape
    leaf_count = (hidden + 31) // 32

    def leaf_sum(leaf: int) -> torch.Tensor:
        h0 = leaf * 32
        h1 = min(h0 + 32, hidden)
        acc = torch.zeros(rows, device=a_f.device, dtype=torch.float32)
        for h in range(h0, h1):
            acc = torch.addcmul(acc, a_f[:, h], b_f[:, h])
        return acc

    def reduce_range(lo: int, hi: int) -> torch.Tensor:
        if hi - lo == 1:
            return leaf_sum(lo)
        mid = lo + (hi - lo) // 2
        return reduce_range(lo, mid) + reduce_range(mid, hi)

    return reduce_range(0, leaf_count)


def txt_in_reference_backward(
    x_f: torch.Tensor,
    gamma_f: torch.Tensor,
    grad_output_f: torch.Tensor,
    xhat: torch.Tensor,
    rstd: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen-semantics dx and dgamma (both fp32; caller casts at return).

    dz = TreeN(g, Wf.t()); du := dz; dxhat = du*gamma; dot = TreeH FMA;
    t1 = dot/3584; t2 = xhat*t1; t3 = dxhat-t2; dx = rstd*t3;
    dgamma = ascending-row FMA fold of (du, xhat).
    """
    rows, hidden = x_f.shape
    dxhat = grad_output_f * gamma_f.unsqueeze(0)  # du applied via gamma mul
    dot = txt_in_row_dot_tree(dxhat, xhat)
    t1 = true_div_rn(dot, DIVISOR_F32)
    t2 = xhat * t1.unsqueeze(1)
    t3 = dxhat - t2
    dx = rstd.unsqueeze(1) * t3
    dgamma = torch.zeros(hidden, device=x_f.device, dtype=torch.float32)
    for s in range(rows):
        dgamma = torch.addcmul(dgamma, grad_output_f[s], xhat[s])
    return dx, dgamma


class _TxtInRMSNormLinearFunction(torch.autograd.Function):
    """Autograd wrapper: frozen forward + frozen-semantics backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x2d = x.reshape(-1, x.size(-1))
        x_f = x2d.float()
        gamma_f = norm_weight.float()
        w_f = weight.float()
        b_f = None if bias is None else bias.float()

        xhat, rstd = txt_in_norm_stats(x_f)
        z = xhat * gamma_f.unsqueeze(0)  # u; frozen FP32 seam -- no cast
        tree_sum = tree_gemm_fp32(z, w_f)
        out = finalize_tree_output(tree_sum, b_f, x.dtype)
        ctx.save_for_backward(x_f, gamma_f, w_f, xhat, rstd, z)
        ctx.lead_shape = x.shape[:-1]
        ctx.has_bias = bias is not None
        return out.reshape(*ctx.lead_shape, weight.size(0))

    @staticmethod
    def backward(ctx, grad_output):
        from rl_engine.kernels.ops.backward_runtime import record_backward
        from rl_engine.kernels.ops.vjp_fp32 import row_local_bias_fp32

        x_f, gamma_f, w_f, xhat, rstd, z = ctx.saved_tensors
        lead_shape = ctx.lead_shape
        rows, hidden = x_f.shape
        g = grad_output.reshape(-1, grad_output.size(-1)).float().contiguous()
        out_dim = g.size(1)

        # dz[s,h] = sum_n g[s,n] * W[n,h] under the N-dim tree (96 leaves);
        # wt is a pure transposed copy (zero rounding).
        dz = tree_gemm_fp32(g, w_f.t().contiguous())
        du = dz  # plain identity (frozen seam -- no cast, no STE)

        # Reuse the frozen backward chain with du as the upstream gradient.
        dxhat = du * gamma_f.unsqueeze(0)
        dot = txt_in_row_dot_tree(dxhat, xhat)
        t1 = true_div_rn(dot, DIVISOR_F32)
        t2 = xhat * t1.unsqueeze(1)
        t3 = dxhat - t2
        dx = rstd.unsqueeze(1) * t3

        dgamma = torch.zeros(hidden, device=x_f.device, dtype=torch.float32)
        for s in range(rows):
            dgamma = torch.addcmul(dgamma, du[s], xhat[s])

        dw = torch.zeros(out_dim, hidden, device=x_f.device, dtype=torch.float32)
        for s in range(rows):
            dw = torch.addcmul(dw, g[s].unsqueeze(1), z[s].unsqueeze(0))

        db = row_local_bias_fp32(g) if ctx.has_bias else None

        record_backward(
            "txt_in_rmsnorm_linear",
            kernel_id=TXT_IN_RMSNORM_LINEAR_CONTRACT_VERSION,
            impl="pytorch_tree_reference",
            family="pytorch",
        )
        grad_x = dx.reshape(*lead_shape, hidden).to(grad_output.dtype)
        grad_gamma = dgamma.to(grad_output.dtype)
        grad_w = dw.to(grad_output.dtype)
        grad_b = None if db is None else db.to(grad_output.dtype)
        return grad_x, grad_gamma, grad_w, grad_b


# norm dtype: gradients return in the OUTPUT dtype == input dtype (all four
# input tensors share one dtype per the frozen contract), so grad_output.dtype
# is the input dtype by construction.
_NORM_DTYPE_NOTE = (
    "All four inputs share one dtype (fail-closed), so the upstream gradient "
    "dtype equals the input dtype and serves as the return-cast target."
)


class NativeTxtInRMSNormLinearOp:
    """FP32-CPU bit-exactness gold and PyTorch dispatch backend.

    ``forward`` returns the input dtype (single RNE cast at the output);
    ``forward_fp32`` keeps fp32 output and is the gtest gold method.
    """

    op_class = "reduction"
    is_batch_invariant = True
    backward_impl = "pytorch_tree_reference"
    contract_version = TXT_IN_RMSNORM_LINEAR_CONTRACT_VERSION

    def __init__(self) -> None:
        logger.info(
            "NativeTxtInRMSNormLinearOp ready (contract %s).",
            TXT_IN_RMSNORM_LINEAR_CONTRACT_VERSION,
        )

    def __call__(
        self,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(x, norm_weight, weight, bias=bias)

    @staticmethod
    def _check_inputs(
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> None:
        if x.dim() < 1 or x.size(-1) != TXT_IN_HIDDEN:
            raise ValueError(f"x must be [*lead, {TXT_IN_HIDDEN}], got {tuple(x.shape)}")
        if norm_weight.shape != (TXT_IN_HIDDEN,):
            raise ValueError(
                f"norm_weight must be [{TXT_IN_HIDDEN},], got {tuple(norm_weight.shape)}"
            )
        if weight.shape != (TXT_IN_OUT, TXT_IN_HIDDEN):
            raise ValueError(
                f"weight must be [{TXT_IN_OUT}, {TXT_IN_HIDDEN}] ([out,in]), "
                f"got {tuple(weight.shape)}"
            )
        dtypes = {x.dtype, norm_weight.dtype, weight.dtype}
        if bias is not None:
            if bias.shape != (TXT_IN_OUT,):
                raise ValueError(f"bias must be [{TXT_IN_OUT},], got {tuple(bias.shape)}")
            dtypes.add(bias.dtype)
            if bias.device != x.device:
                raise ValueError("bias must live on the same device as x")
        if len(dtypes) > 1:
            raise ValueError(f"all inputs must share one dtype, got {sorted(map(str, dtypes))}")
        if norm_weight.device != x.device or weight.device != x.device:
            raise ValueError("all tensors must live on the same device")

    def forward(
        self,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._check_inputs(x, norm_weight, weight, bias)
        return _TxtInRMSNormLinearFunction.apply(x, norm_weight, weight, bias)

    def forward_fp32(
        self,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Ground truth: full frozen pipeline, fp32 output (no cast)."""
        self._check_inputs(x, norm_weight, weight, bias)
        x2d = x.reshape(-1, x.size(-1)).float()
        gamma_f = norm_weight.float()
        w_f = weight.float()
        b_f = None if bias is None else bias.float()
        xhat, _ = txt_in_norm_stats(x2d)
        z = xhat * gamma_f.unsqueeze(0)
        tree_sum = tree_gemm_fp32(z, w_f)
        out = finalize_tree_output(tree_sum, b_f, torch.float32)
        return out.reshape(*x.shape[:-1], TXT_IN_OUT)
