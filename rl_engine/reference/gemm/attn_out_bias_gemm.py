# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Qwen-Image WS1 attn-out bias GEMM: ``y = single_cast(x @ W.T + b)``.

Covers the MMDiT attention output projections ``to_out`` (image stream) and
``to_add_out`` (text stream) -- mathematically identical ``[3072, 3072]`` bias
GEMMs applied to the already-dtype-rounded joint-attention output.

Frozen numeric contract (issue #386, contract version
``attn-out-bias-gemm-tree-v1``; the human-readable contract lives in
``docs/operators/attn-out-bias-gemm.md``):

1. **K-dim reduction tree.** The reduction length ``R`` splits into 32-wide
   leaves (the last leaf may be short). Each leaf is summed by an ascending-k
   fp32 chain that starts from ``+0.0`` (so ``0 + (-0.0)`` normalises the first
   product to ``+0.0``). Leaves combine through a mid-split tree
   ``T(l, r) = T(l, m) + T(m, r)`` over leaf indices. The tree depends only on
   ``R`` -- never on batch size, token count, or tiling -- so row outputs are
   batch-invariant. A two-way reduction composes when each shard evaluates
   one root subtree. At R=3072 this splits 96 leaves into two groups of 48.
2. **Multiply-add discipline.** Leaf chains and the weight-gradient fold
   use FP32 FMA. Triton uses ``tl.fma``, native CUDA uses ``__fmaf_rn``,
   and the PyTorch reference uses a compensated FP64 sum rounded to FP32. Tests compare
   the reference against ``libm fmaf`` on the tested toolchains.
3. **bias.** Added once per output element, in fp32, after the complete K
   tree; then the single output cast. Never inside a leaf chain or tree node.
4. **S-dim reductions** (parameter gradients only) use an ascending-row fp32
   left fold, matching ``rl_engine.ops.autograd.vjp_fp32``.

The forward uses ``R = K`` (3072 -> 96 leaves) and ``dx`` uses ``R = N`` under
the mid-split tree; ``dW`` reduces over the row dim S with an ascending-row
left fold (contract c-prime, shared with ``db``).

The reference computes on the input device using PyTorch elementwise ops.
Triton computes the tree per output tile and shares the helper for bias and
the final cast. Native CUDA evaluates the tree, bias and cast in
the device kernel.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from rl_engine.ops.autograd.vjp_fp32 import row_local_bias_fp32
from rl_engine.utils.logger import logger

LEAF_WIDTH = 32
ATTN_OUT_BIAS_GEMM_CONTRACT_VERSION = "attn-out-bias-gemm-tree-v1"


def num_leaves(reduction_len: int, leaf: int = LEAF_WIDTH) -> int:
    """Number of leaves for a reduction of ``reduction_len`` (last may be short)."""
    if reduction_len < 0:
        raise ValueError(f"reduction_len must be >= 0, got {reduction_len}")
    return (reduction_len + leaf - 1) // leaf


def tree_from_leaf_evaluator(
    evaluate_leaf: Callable[[int], torch.Tensor], leaf_count: int
) -> torch.Tensor:
    """Mid-split tree combine over an evaluator of leaf partials.

    ``T(l, r) = T(l, m) + T(m, r)`` with ``m = l + (r - l) // 2``; a single
    leaf returns its own partial. Every add is an fp32 elementwise op (RNE).
    Depth-first evaluation keeps peak memory at O(log leaves) partials.
    """
    if leaf_count < 0:
        raise ValueError(f"leaf_count must be >= 0, got {leaf_count}")

    def reduce_range(lo: int, hi: int) -> torch.Tensor:
        if hi - lo == 1:
            return evaluate_leaf(lo)
        mid = lo + (hi - lo) // 2
        left = reduce_range(lo, mid)
        right = reduce_range(mid, hi)
        return left + right

    if leaf_count == 0:
        raise ValueError("tree_from_leaf_evaluator requires at least one leaf")
    return reduce_range(0, leaf_count)


def tree_combine(leaf_sums: Sequence[torch.Tensor]) -> torch.Tensor:
    """Mid-split tree combine over an explicit sequence of leaf partials."""
    if len(leaf_sums) == 0:
        raise ValueError("tree_combine requires at least one leaf partial")
    return tree_from_leaf_evaluator(lambda index: leaf_sums[index], len(leaf_sums))


def finalize_tree_output(
    tree_sum: torch.Tensor, bias_f: Optional[torch.Tensor], output_dtype: torch.dtype
) -> torch.Tensor:
    """Add bias once (fp32) after the complete tree, then the single RNE cast."""
    if bias_f is not None:
        tree_sum = tree_sum + bias_f
    return tree_sum.to(output_dtype)


def _require_gradual_underflow(values: torch.Tensor) -> None:
    """Check on each CPU invocation because denormal handling is thread-local."""
    if values.device.type == "cpu":
        tiny = torch.tensor(torch.finfo(torch.float32).tiny, dtype=torch.float32, device="cpu")
        if (tiny * 0.5).item() == 0.0:
            raise RuntimeError(
                "attn_out_bias_gemm CPU reference requires gradual underflow; call "
                "torch.set_flush_denormal(False) before execution and CPU worker initialization"
            )


def _fma_fp32(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One FP32 rounding of a*b + acc for FP32 operands."""

    # A binary32 product is exactly representable in binary64, including at
    # the extremes of the binary32 exponent range.
    product = a.double() * b.double()
    addend = acc.double()
    total = product + addend
    with torch.no_grad():
        # FastTwoSum recovers the error of the binary64 addition when the
        # larger-magnitude operand is subtracted first.
        product_larger = product.abs() >= addend.abs()
        residual = torch.where(
            product_larger, (product - total) + addend, (addend - total) + product
        )
        # Round-to-odd before narrowing: an inexact sum with an even low bit
        # moves one binary64 ulp toward the exact value. This puts it on the
        # correct side of every binary32 midpoint, including the overflow tie.
        even = (total.view(torch.int64) & 1) == 0
        adjust = even & ((residual > 0) | (residual < 0))
        direction = torch.copysign(torch.full_like(total, float("inf")), residual)
        step = torch.nextafter(total, direction) - total
    # Keep the differentiable sum and the sign of an unadjusted exact zero.
    return torch.where(adjust, total + step, total).float()


def _leaf_sum_fp32(a_f: torch.Tensor, b_f: torch.Tensor, leaf_index: int) -> torch.Tensor:
    """One leaf of ``a_f @ b_f.T`` (both fp32): ascending-k correctly-rounded FMA chain.

    Each broadcast outer-product step uses the same FP32 FMA primitive as dW.
    """
    k0 = leaf_index * LEAF_WIDTH
    k1 = min(k0 + LEAF_WIDTH, a_f.size(1))
    acc = torch.zeros(a_f.size(0), b_f.size(0), device=a_f.device, dtype=torch.float32)
    for k in range(k0, k1):
        acc = _fma_fp32(acc, a_f[:, k].unsqueeze(1), b_f[:, k].unsqueeze(0))
    return acc


def tree_gemm_fp32(a_f: torch.Tensor, b_f: torch.Tensor) -> torch.Tensor:
    """``a_f @ b_f.T`` for 2-D fp32 inputs under the frozen K-dim tree.

    ``a_f`` is ``[S, R]`` and ``b_f`` is ``[M, R]``. Forward reduces over
    R=K and ``dx`` reduces over R=N. The weight gradient uses a separate
    ascending-row fold over S.
    """
    if a_f.dim() != 2 or b_f.dim() != 2:
        raise ValueError(
            f"tree_gemm_fp32 expects 2-D inputs, got {tuple(a_f.shape)} and {tuple(b_f.shape)}"
        )
    if a_f.size(1) != b_f.size(1):
        raise ValueError(f"reduction dims must match: a has {a_f.size(1)}, b has {b_f.size(1)}")
    _require_gradual_underflow(a_f)
    if a_f.size(1) == 0:
        return torch.zeros(a_f.size(0), b_f.size(0), device=a_f.device, dtype=torch.float32)
    return tree_from_leaf_evaluator(
        lambda index: _leaf_sum_fp32(a_f, b_f, index), num_leaves(a_f.size(1))
    )


def _dw_left_fold_reference(g_f: torch.Tensor, x_f: torch.Tensor) -> torch.Tensor:
    """dW by the ascending-row left fold, one correctly-rounded FMA per row.

    Same single-rounding primitive as ``_leaf_sum_fp32``: the fold accumulates
    ``dw = _fma_fp32(dw, g_row, x_row)`` per row in ascending order.
    """
    dw = torch.zeros(g_f.size(1), x_f.size(1), device=g_f.device, dtype=torch.float32)
    for row in range(g_f.size(0)):
        dw = _fma_fp32(dw, g_f[row].unsqueeze(1), x_f[row].unsqueeze(0))
    return dw


def attn_out_bias_gemm_reference_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    needs_grad_x: bool = True,
    needs_grad_weight: bool = True,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Reference ``dx`` and ``dW``; ``db`` is the shared left fold.

    ``dx = dY @ W`` reduces over ``R = N`` under the mid-split tree (same
    K-dim rule as the forward). ``dW = dY.T @ x`` reduces over the row dim S
    with the **ascending-row left fold** (contract c-prime), one correctly-
    rounded FMA per row (same primitive as the leaf chains).
    """
    _require_gradual_underflow(grad_output)
    g_f = grad_output.reshape(-1, grad_output.size(-1)).float()
    dx = None
    dw = None
    if needs_grad_x:
        dx = tree_gemm_fp32(g_f, weight.float().t().contiguous())
    if needs_grad_weight:
        x_f = x.reshape(-1, x.size(-1)).float()
        dw = _dw_left_fold_reference(g_f, x_f)
    return dx, dw


class _AttnOutBiasGemmTreeFunction(torch.autograd.Function):
    """Autograd wrapper with the tree-disciplined forward and backward."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x2d = x.reshape(-1, x.size(-1))
        lead_shape = x.shape[:-1]
        out_dim = weight.size(0)
        tree_sum = tree_gemm_fp32(x2d.float(), weight.float())
        out = finalize_tree_output(tree_sum, None if bias is None else bias.float(), x.dtype)
        ctx.save_for_backward(x2d, weight)
        ctx.lead_shape = lead_shape
        ctx.out_dim = out_dim
        return out.reshape(*lead_shape, out_dim)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        from rl_engine.ops.autograd.backward_runtime import record_backward

        x2d, weight = ctx.saved_tensors
        grad_x, grad_w = attn_out_bias_gemm_reference_backward(
            x2d,
            weight,
            grad_output,
            needs_grad_x=ctx.needs_input_grad[0],
            needs_grad_weight=ctx.needs_input_grad[1],
        )
        if grad_x is not None:
            grad_x = grad_x.reshape(*ctx.lead_shape, x2d.size(-1)).to(x2d.dtype)
        if grad_w is not None:
            grad_w = grad_w.to(weight.dtype)
        grad_b: Optional[torch.Tensor] = None
        if ctx.needs_input_grad[2]:
            grad_b = row_local_bias_fp32(grad_output).to(x2d.dtype)
        record_backward(
            "attn_out_bias_gemm",
            kernel_id=ATTN_OUT_BIAS_GEMM_CONTRACT_VERSION,
            impl="pytorch_tree_reference",
            family="pytorch",
            node_metadata=ctx.metadata,
        )
        return grad_x, grad_w, grad_b


class AttnOutBiasGemmBackend:
    """Selection metadata for a concrete backend with no internal fallback."""

    backend_id: str
    provenance: Mapping[str, Any] = MappingProxyType({})

    @property
    def trace(self) -> dict[str, Any]:
        selection = self.provenance
        return {
            "actual_backend": self.backend_id,
            "fallback": False,
            "fallback_reason": "; ".join(selection.get("prior_rejections", ())) or None,
            **selection,
        }


class NativeAttnOutBiasGemmOp(AttnOutBiasGemmBackend):
    """FP32-CPU bit-exactness gold and PyTorch dispatch backend.

    ``forward`` returns the input dtype (single RNE cast at the output, after
    bias); ``forward_fp32`` keeps fp32 output and is the gtest gold method.
    The explicit tree evaluation makes every output row a pure function of
    that row and the weights, so batch invariance holds bitwise.
    """

    backend_id = "pytorch"
    op_class = "reduction"
    is_batch_invariant = True
    backward_impl = "pytorch_tree_reference"
    contract_version = ATTN_OUT_BIAS_GEMM_CONTRACT_VERSION

    def __init__(self) -> None:
        logger.info(
            "NativeAttnOutBiasGemmOp ready (contract %s).", ATTN_OUT_BIAS_GEMM_CONTRACT_VERSION
        )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(x, weight, bias=bias)

    @staticmethod
    def _check_inputs(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        if x.device.type != "cpu" and not (x.is_cuda and torch.version.cuda is not None):
            raise ValueError("attn_out_bias_gemm requires NVIDIA CUDA tensors or a CPU reference")
        if x.dim() < 1:
            raise ValueError(f"x must be at least 1-D [*lead, K], got shape {tuple(x.shape)}")
        if weight.dim() != 2:
            raise ValueError(f"weight must be 2-D [N, K], got shape {tuple(weight.shape)}")
        if x.size(-1) != weight.size(1):
            raise ValueError(
                f"x trailing dim {x.size(-1)} must match weight input dim {weight.size(1)}"
            )
        if x.size(-1) == 0 or weight.size(0) == 0:
            raise ValueError("input and output feature dimensions must be positive")
        if x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"supported dtypes are bf16/fp32 (contract scope), got {x.dtype}")
        if x.device != weight.device:
            raise ValueError(
                f"x and weight must live on the same device: {x.device} vs {weight.device}"
            )
        if x.dtype != weight.dtype:
            raise ValueError(f"x and weight dtypes must match: {x.dtype} vs {weight.dtype}")
        if bias is not None:
            if bias.dim() != 1:
                raise ValueError(f"bias must be 1-D [N], got shape {tuple(bias.shape)}")
            if bias.numel() != weight.size(0):
                raise ValueError(
                    f"bias size {bias.numel()} must match weight output dim {weight.size(0)}"
                )
            if bias.device != weight.device or bias.dtype != weight.dtype:
                raise ValueError("bias must share the device and dtype of weight")

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Canonical entry: fp32 tree reduction, bias once, single RNE cast."""
        self._check_inputs(x, weight, bias)
        return _AttnOutBiasGemmTreeFunction.apply(x, weight, bias)

    def forward_fp32(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Ground truth: fp32 tree reduction with bias, fp32 output (no cast)."""
        self._check_inputs(x, weight, bias)
        x2d = x.reshape(-1, x.size(-1))
        tree_sum = tree_gemm_fp32(x2d.float(), weight.float())
        out = finalize_tree_output(tree_sum, None if bias is None else bias.float(), torch.float32)
        return out.reshape(*x.shape[:-1], weight.size(0))
