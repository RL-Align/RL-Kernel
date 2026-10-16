# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Qwen-Image WS1 joint attention QK GEMM: ``scores = single_cast((Q Kᵀ) * scale)``.

First of the three unfused joint-attention operators in issue #386::

    joint_attn_qk_gemm -> joint_attn_softmax -> joint_attn_av_gemm

Qwen-Image MMDiT runs **non-causal** attention over the joint sequence
``[text, image]``: both streams are concatenated before attention, so this
operator sees one sequence and needs no causal mask. Per-stream Q/K projection,
learned-gain Q/K RMSNorm and three-axis RoPE are all applied *before* the call,
so the ``q``/``k`` passed in are already normalised and rotated.

Shapes follow the model contract: ``q`` is ``[B, H, Sq, D]`` and ``k`` is
``[B, H, Sk, D]`` with ``H = 24`` heads of ``D = 128``, and ``scale =
1/sqrt(D)``. At the three reference resolutions the image stream contributes
4096, 6032 and 6889 tokens, to which the padded text prompt is prepended.

Frozen numeric contract (contract version ``joint-attn-qk-gemm-tree-v1``; the
human-readable contract lives in ``docs/operators/joint-attn-qk-gemm.md``):

1. **Forward reduction tree (D-dim).** Each score element reduces over the head
   dimension ``D`` only. ``D`` splits into 32-wide leaves (the last leaf may be
   short); each leaf is an ascending-``d`` fp32 FMA chain started from ``+0.0``
   (so ``0 + (-0.0)`` normalises the first product to ``+0.0``), and leaves
   combine through a mid-split tree ``T(l, r) = T(l, m) + T(m, r)``. Because the
   tree depends only on ``D`` -- a per-head architectural constant, never on
   batch size, token count, prompt padding or tiling -- **every forward score
   element is batch-, position- and padding-invariant by construction**. At the
   production ``D = 128`` the tree is exactly four leaves.
2. **Multiply-add discipline.** One correctly-rounded FP32 FMA at every
   multiply-into-add site, on both sides: device backends use explicit
   ``fma_rn`` intrinsics and this reference uses ``torch.addcmul``, a
   single-rounding FP32 FMA on the pinned toolchains. No intermediate double
   rounding and no tolerance path. RNE rounding everywhere, including the single
   fp32 -> input-dtype output cast.
3. **scale.** Applied once per output element, in fp32, *after* the complete
   D tree and before the single output cast. Never folded into ``q``, never
   applied inside a leaf chain or a tree node.
4. **Backward reductions (sequence-dim).** ``dQ`` reduces over ``Sk`` and ``dK``
   over ``Sq``, each under the same 32-wide-leaf mid-split tree. These lengths
   *do* depend on prompt padding, so padding invariance requires the explicit
   padding masks: valid positions are compacted into ascending logical order
   before the tree is built, so inserting padding cannot regroup the surviving
   terms. ``scale`` is likewise applied once in fp32 after the tree.

The tree discipline is deliberately identical to ``attn-out-bias-gemm-tree-v1``
(issue #386 row ``attn_out_bias_gemm``, PR #459) so the two GEMM rows share one
reduction definition; once that PR lands, ``num_leaves`` /
``tree_from_leaf_evaluator`` should be de-duplicated into a shared module.

This module is the bit-exactness gold: the trees are evaluated with explicit
torch elementwise ops only -- no ``matmul``, no ``scaled_dot_product_attention``,
no implicit reductions -- so it is bit-identical on every device and serves as
the FP32 CPU reference that the Triton and CUDA backends are compared against
byte for byte.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch

LEAF_WIDTH = 32
JOINT_ATTN_QK_GEMM_CONTRACT_VERSION = "joint-attn-qk-gemm-tree-v1"


# --------------------------------------------------------------------------- #
# FMA regime probe
# --------------------------------------------------------------------------- #
# The leaf chains below use ``torch.addcmul`` as their multiply-into-add
# primitive. On CPU that is a correctly-rounded single-rounding FP32 FMA,
# verified against an exact-rational oracle (see the test suite). It is NOT
# portable: on CUDA, ``torch.addcmul`` lowers to a separate multiply and add and
# is bit-identical to ``c + a * b``, so the same inputs produce different bits on
# CPU and CUDA. The canonical double-rounding probe is ``a = b = 1 + 2**-12``,
# ``c = 2**-60``: a true FMA returns ``0x3F801001`` and a double-rounded
# multiply-then-add returns ``0x3F801000``.
#
# Consequence for this contract: **CPU is the bit-defining device for the gold.**
# ``forward_fp32`` fails closed on any device whose ``addcmul`` is not a true
# FMA, rather than silently returning different bytes. Device backends use
# explicit ``fma_rn`` intrinsics and are compared byte for byte against the CPU
# gold, which is exactly the "independent FP32 CPU reference" issue #386 asks
# for.
_FMA_PROBE_A = float(torch.tensor(1.0 + 2.0**-12, dtype=torch.float32))
_FMA_PROBE_C = float(torch.tensor(2.0**-60, dtype=torch.float32))
_FMA_PROBE_SINGLE_ROUNDING_BITS = 0x3F801001
_fma_regime_cache: dict[str, bool] = {}


def fma_is_single_rounding(device: torch.device | str) -> bool:
    """True when ``torch.addcmul`` is a correctly-rounded FP32 FMA on ``device``.

    Probes the classic double-rounding counterexample. Cached per device type.
    """
    resolved = torch.device(device)
    cached = _fma_regime_cache.get(resolved.type)
    if cached is not None:
        return cached
    a = torch.tensor([_FMA_PROBE_A], dtype=torch.float32, device=resolved)
    c = torch.tensor([_FMA_PROBE_C], dtype=torch.float32, device=resolved)
    observed = int(torch.addcmul(c, a, a).view(torch.int32).cpu().item()) & 0xFFFFFFFF
    result = observed == _FMA_PROBE_SINGLE_ROUNDING_BITS
    _fma_regime_cache[resolved.type] = result
    return result


def require_single_rounding_fma(device: torch.device | str) -> None:
    """Fail closed when the bit-defining FMA primitive is unavailable on ``device``."""
    if not fma_is_single_rounding(device):
        raise RuntimeError(
            f"{JOINT_ATTN_QK_GEMM_CONTRACT_VERSION}: torch.addcmul is not a correctly-rounded "
            f"FP32 FMA on device type '{torch.device(device).type}', so the bit-exact fp32 gold "
            "cannot be produced there. Run the gold on CPU (the bit-defining device for this "
            "contract) and compare device backends against it."
        )


# --------------------------------------------------------------------------- #
# Declared arithmetic policy (trace metadata)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JointAttnQkGemmPlan:
    """Readback of the arithmetic policy actually used for a call.

    Mirrors ``rl_engine.ops.attention.projection.ProjectionPlan`` so the WS1
    trace can record reduction order, accumulator precision, cast points and the
    selected backend without the caller inspecting the implementation.
    """

    backend_id: str
    contract_version: str = JOINT_ATTN_QK_GEMM_CONTRACT_VERSION
    forward_reduction_dim: str = "head_dim"
    forward_reduction_order: str = "leaf32_ascending_d_midsplit_tree"
    backward_dq_reduction_dim: str = "key_seq"
    backward_dk_reduction_dim: str = "query_seq"
    backward_reduction_order: str = "leaf32_ascending_midsplit_tree_logical_order"
    accumulation_dtype: str = "torch.float32"
    multiply_add: str = "correctly_rounded_fp32_fma"
    scale_applied: str = "once_fp32_after_tree"
    cast_points: str = "single_output_cast_rne"
    split_k: bool = False
    atomics: bool = False
    tf32: bool = False
    fast_math: bool = False
    causal: bool = False
    batch_invariant: bool = True
    padding_invariant_backward: str = "requires_explicit_masks"
    fallback: bool = False
    fallback_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.accumulation_dtype != "torch.float32":
            raise ValueError("joint_attn_qk_gemm requires FP32 accumulation")
        if self.split_k or self.atomics or self.tf32 or self.fast_math:
            raise ValueError(
                "joint_attn_qk_gemm forbids Split-K, atomic accumulation, TF32 and fast math"
            )
        if self.scale_applied != "once_fp32_after_tree":
            raise ValueError("scale must be applied once in fp32 after the complete tree")

    def to_dict(self) -> dict[str, object]:
        """Flat mapping for the WS1 operator trace."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Frozen reduction tree
# --------------------------------------------------------------------------- #
def num_leaves(reduction_len: int, leaf: int = LEAF_WIDTH) -> int:
    """Number of leaves for a reduction of ``reduction_len`` (last may be short)."""
    if reduction_len < 0:
        raise ValueError(f"reduction_len must be >= 0, got {reduction_len}")
    return (reduction_len + leaf - 1) // leaf


def tree_from_leaf_evaluator(
    evaluate_leaf: Callable[[int], torch.Tensor], leaf_count: int
) -> torch.Tensor:
    """Mid-split tree combine over an evaluator of leaf partials.

    ``T(l, r) = T(l, m) + T(m, r)`` with ``m = l + (r - l) // 2``; a single leaf
    returns its own partial. Every add is an fp32 elementwise op (RNE).
    Depth-first evaluation keeps peak memory at O(log leaves) partials.
    """
    if leaf_count <= 0:
        raise ValueError(f"tree_from_leaf_evaluator requires at least one leaf, got {leaf_count}")

    def reduce_range(lo: int, hi: int) -> torch.Tensor:
        if hi - lo == 1:
            return evaluate_leaf(lo)
        mid = lo + (hi - lo) // 2
        return reduce_range(lo, mid) + reduce_range(mid, hi)

    return reduce_range(0, leaf_count)


def finalize_scaled_output(
    tree_sum: torch.Tensor, scale: float, output_dtype: torch.dtype
) -> torch.Tensor:
    """Apply ``scale`` once in fp32 after the complete tree, then the single RNE cast."""
    return (tree_sum * scale).to(output_dtype)


# --------------------------------------------------------------------------- #
# Forward: reduction over the head dimension D
# --------------------------------------------------------------------------- #
def _qk_leaf_fp32(q_f: torch.Tensor, k_f: torch.Tensor, leaf_index: int) -> torch.Tensor:
    """One D-leaf of ``q_f @ k_f.transpose(-1, -2)``: ascending-``d`` FMA chain.

    ``q_f`` is ``[..., Sq, D]`` and ``k_f`` is ``[..., Sk, D]``, both fp32. Each
    step is one ``torch.addcmul`` -- a single-rounding correctly-rounded FP32 FMA
    -- over the broadcast outer product of column ``d``, matching the device
    backends' ``__fmaf_rn`` chains bit for bit.
    """
    head_dim = q_f.size(-1)
    d0 = leaf_index * LEAF_WIDTH
    d1 = min(d0 + LEAF_WIDTH, head_dim)
    acc = torch.zeros((*q_f.shape[:-1], k_f.size(-2)), device=q_f.device, dtype=torch.float32)
    for d in range(d0, d1):
        acc = torch.addcmul(acc, q_f[..., d].unsqueeze(-1), k_f[..., d].unsqueeze(-2))
    return acc


def qk_scores_tree_fp32(q_f: torch.Tensor, k_f: torch.Tensor) -> torch.Tensor:
    """Unscaled ``Q Kᵀ`` in fp32 under the frozen D-dim tree.

    ``q_f`` is ``[..., Sq, D]``, ``k_f`` is ``[..., Sk, D]``; returns
    ``[..., Sq, Sk]``. The reduction runs over ``D`` for every output element
    independently, so the result does not depend on ``Sq``, ``Sk``, the batch
    layout or any tiling of the output.
    """
    if q_f.size(-1) != k_f.size(-1):
        raise ValueError(f"head dims must match: q has {q_f.size(-1)}, k has {k_f.size(-1)}")
    head_dim = q_f.size(-1)
    if head_dim == 0:
        return torch.zeros((*q_f.shape[:-1], k_f.size(-2)), device=q_f.device, dtype=torch.float32)
    return tree_from_leaf_evaluator(
        lambda index: _qk_leaf_fp32(q_f, k_f, index), num_leaves(head_dim)
    )


# --------------------------------------------------------------------------- #
# Backward: reductions over the sequence dimensions
# --------------------------------------------------------------------------- #
# A "leaf evaluator" computes one 32-wide leaf partial in fp32. The host tree
# and the padding compaction below are shared by every backend: a device backend
# supplies only its own leaf evaluator, so the merge order is identical by
# construction instead of being reimplemented per backend.
LeafEvaluator = Callable[[torch.Tensor, torch.Tensor, int, int], torch.Tensor]


def _dq_leaf_range_fp32(g_f: torch.Tensor, k_f: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """One ``Sk``-leaf of ``dQ = dS @ K`` over keys ``[lo, hi)``: ascending-``j`` FMA chain.

    ``g_f`` is ``[..., Sq, Sk]`` and ``k_f`` is ``[..., Sk, D]``, both fp32;
    returns ``[..., Sq, D]``.
    """
    acc = torch.zeros((*g_f.shape[:-1], k_f.size(-1)), device=g_f.device, dtype=torch.float32)
    for j in range(lo, hi):
        acc = torch.addcmul(acc, g_f[..., j].unsqueeze(-1), k_f[..., j, :].unsqueeze(-2))
    return acc


def _dk_leaf_range_fp32(g_f: torch.Tensor, q_f: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """One ``Sq``-leaf of ``dK = dSᵀ @ Q`` over queries ``[lo, hi)``: ascending-``i`` FMA chain.

    ``g_f`` is ``[..., Sq, Sk]`` and ``q_f`` is ``[..., Sq, D]``, both fp32;
    returns ``[..., Sk, D]``.
    """
    acc = torch.zeros(
        (*g_f.shape[:-2], g_f.size(-1), q_f.size(-1)), device=g_f.device, dtype=torch.float32
    )
    for i in range(lo, hi):
        acc = torch.addcmul(acc, g_f[..., i, :].unsqueeze(-1), q_f[..., i, :].unsqueeze(-2))
    return acc


def dq_tree_fp32(
    g_f: torch.Tensor, k_f: torch.Tensor, leaf_fn: Optional[LeafEvaluator] = None
) -> torch.Tensor:
    """``dS @ K`` in fp32 under the frozen mid-split tree over ``Sk``."""
    evaluate = leaf_fn if leaf_fn is not None else _dq_leaf_range_fp32
    key_len = k_f.size(-2)
    if key_len == 0:
        return torch.zeros((*g_f.shape[:-1], k_f.size(-1)), device=g_f.device, dtype=torch.float32)
    return tree_from_leaf_evaluator(
        lambda index: evaluate(
            g_f, k_f, index * LEAF_WIDTH, min((index + 1) * LEAF_WIDTH, key_len)
        ),
        num_leaves(key_len),
    )


def dk_tree_fp32(
    g_f: torch.Tensor, q_f: torch.Tensor, leaf_fn: Optional[LeafEvaluator] = None
) -> torch.Tensor:
    """``dSᵀ @ Q`` in fp32 under the frozen mid-split tree over ``Sq``."""
    evaluate = leaf_fn if leaf_fn is not None else _dk_leaf_range_fp32
    query_len = q_f.size(-2)
    if query_len == 0:
        return torch.zeros(
            (*g_f.shape[:-2], g_f.size(-1), q_f.size(-1)),
            device=g_f.device,
            dtype=torch.float32,
        )
    return tree_from_leaf_evaluator(
        lambda index: evaluate(
            g_f, q_f, index * LEAF_WIDTH, min((index + 1) * LEAF_WIDTH, query_len)
        ),
        num_leaves(query_len),
    )


def _valid_indices(mask_row: torch.Tensor) -> torch.Tensor:
    """Ascending logical positions of the valid entries of a ``[S]`` bool mask."""
    return torch.nonzero(mask_row, as_tuple=False).flatten()


def joint_attn_qk_gemm_backward_with_leaves(
    q: torch.Tensor,
    k: torch.Tensor,
    grad_scores: torch.Tensor,
    *,
    scale: Optional[float] = None,
    query_padding_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    dq_leaf_fn: Optional[LeafEvaluator] = None,
    dk_leaf_fn: Optional[LeafEvaluator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared host backward: padding compaction, mid-split tree, scale and cast.

    ``dQ = (dS @ K) * scale`` reduces over ``Sk``; ``dK = (dSᵀ @ Q) * scale``
    reduces over ``Sq``. Both use the 32-wide-leaf mid-split tree. Backends pass
    their own leaf evaluators and inherit this merge order unchanged.

    When a padding mask is supplied the corresponding reduction is taken over the
    **compacted** valid positions in ascending logical order, so inserting prompt
    padding cannot regroup the surviving terms; padded output rows are defined as
    exactly zero. Without masks the reduction follows physical positions, which is
    correct but only padding-invariant if the caller keeps the layout fixed.
    """
    head_dim = q.size(-1)
    scale = scale if scale is not None else (1.0 / math.sqrt(head_dim))
    q_f = q.float()
    k_f = k.float()
    g_f = grad_scores.float()

    if query_padding_mask is None and key_padding_mask is None:
        dq_sum = dq_tree_fp32(g_f, k_f, dq_leaf_fn)
        dk_sum = dk_tree_fp32(g_f, q_f, dk_leaf_fn)
    else:
        dq_sum = torch.zeros_like(q_f)
        dk_sum = torch.zeros_like(k_f)
        for b in range(q_f.size(0)):
            keep_j = (
                _valid_indices(key_padding_mask[b])
                if key_padding_mask is not None
                else torch.arange(k_f.size(-2), device=k_f.device)
            )
            keep_i = (
                _valid_indices(query_padding_mask[b])
                if query_padding_mask is not None
                else torch.arange(q_f.size(-2), device=q_f.device)
            )
            # dQ: rows are independent, so only the j-reduction is compacted.
            if keep_j.numel() > 0:
                dq_sum[b : b + 1] = dq_tree_fp32(
                    g_f[b : b + 1].index_select(-1, keep_j).contiguous(),
                    k_f[b : b + 1].index_select(-2, keep_j).contiguous(),
                    dq_leaf_fn,
                )
            # dK: columns are independent, so only the i-reduction is compacted,
            # and the result lands back on the valid key rows.
            if keep_i.numel() > 0:
                dk_sum[b : b + 1] = dk_tree_fp32(
                    g_f[b : b + 1].index_select(-2, keep_i).contiguous(),
                    q_f[b : b + 1].index_select(-2, keep_i).contiguous(),
                    dk_leaf_fn,
                )
        if query_padding_mask is not None:
            dq_sum = torch.where(query_padding_mask[:, None, :, None], dq_sum, 0.0)
        if key_padding_mask is not None:
            dk_sum = torch.where(key_padding_mask[:, None, :, None], dk_sum, 0.0)

    return (dq_sum * scale).to(q.dtype), (dk_sum * scale).to(k.dtype)


def joint_attn_qk_gemm_reference_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    grad_scores: torch.Tensor,
    *,
    scale: Optional[float] = None,
    query_padding_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``dQ``/``dK``: the shared host backward with the torch leaf chains."""
    return joint_attn_qk_gemm_backward_with_leaves(
        q,
        k,
        grad_scores,
        scale=scale,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)


def validate_joint_attn_qk_gemm_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    query_padding_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> None:
    """Fail closed on any geometry or dtype outside the declared contract."""
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError(f"q and k must be [B, H, S, D], got {tuple(q.shape)} and {tuple(k.shape)}")
    if q.dtype != k.dtype:
        raise ValueError(f"q and k dtypes must match, got {q.dtype} and {k.dtype}")
    if q.dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"dtype {q.dtype} is outside the WS1 contract for this operator; "
            f"supported: {SUPPORTED_DTYPES}"
        )
    if q.device != k.device:
        raise ValueError(f"q and k must share a device, got {q.device} and {k.device}")
    if q.size(0) != k.size(0):
        raise ValueError(f"batch mismatch: q has {q.size(0)}, k has {k.size(0)}")
    if q.size(1) != k.size(1):
        raise ValueError(
            f"head count mismatch: q has {q.size(1)}, k has {k.size(1)}; this operator is "
            "joint non-causal attention with one head set per stream, not GQA"
        )
    if q.size(3) != k.size(3):
        raise ValueError(f"head dim mismatch: q has {q.size(3)}, k has {k.size(3)}")
    for name, mask, length in (
        ("query_padding_mask", query_padding_mask, q.size(2)),
        ("key_padding_mask", key_padding_mask, k.size(2)),
    ):
        if mask is None:
            continue
        if mask.dtype != torch.bool:
            raise ValueError(f"{name} must be bool (True=valid), got {mask.dtype}")
        if tuple(mask.shape) != (q.size(0), length):
            raise ValueError(
                f"{name} must be [B, S] = {(q.size(0), length)}, got {tuple(mask.shape)}"
            )


# --------------------------------------------------------------------------- #
# Autograd wrapper and op
# --------------------------------------------------------------------------- #
class _JointAttnQkGemmTreeFunction(torch.autograd.Function):
    """Autograd wrapper with the tree-disciplined forward and backward."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        scale: float,
        query_padding_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        tree_sum = qk_scores_tree_fp32(q.float(), k.float())
        ctx.save_for_backward(q, k, query_padding_mask, key_padding_mask)
        ctx.scale = scale
        return finalize_scaled_output(tree_sum, scale, output_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_scores: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
        q, k, query_padding_mask, key_padding_mask = ctx.saved_tensors
        grad_q, grad_k = joint_attn_qk_gemm_reference_backward(
            q,
            k,
            grad_scores,
            scale=ctx.scale,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
        )
        return grad_q, grad_k, None, None, None, None


class NativeJointAttnQkGemmOp:
    """Pure PyTorch fixed-order reference for the Qwen-Image joint QK GEMM.

    ``scores = single_cast((Q Kᵀ) * scale)`` over the joint ``[text, image]``
    sequence, non-causal. Hand-written tree reduction -- deliberately NOT
    ``torch.matmul`` / ``F.scaled_dot_product_attention``, whose reduction order
    is unspecified and would break the batch-invariance contract. This op
    defines the *correct answer* the Triton and CUDA backends align to.

    No masking happens here: this operator produces raw scores and
    ``joint_attn_softmax`` owns the mask. The padding masks accepted below are
    used only to compact the backward sequence reductions, which is what makes
    ``dQ``/``dK`` padding-invariant.
    """

    contract_version = JOINT_ATTN_QK_GEMM_CONTRACT_VERSION

    def __init__(self) -> None:
        """No state; the op is a pure function over (q, k, ...)."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        scale: Optional[float] = None,
        query_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for ``forward`` so the op is callable like a module."""
        return self.forward(
            q,
            k,
            scale=scale,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        scale: Optional[float] = None,
        query_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Canonical entry: fp32 tree accumulation, single cast to the input dtype."""
        return self._run(
            q,
            k,
            scale=scale,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            output_dtype=q.dtype,
        )

    def forward_fp32(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        scale: Optional[float] = None,
        query_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Ground truth: fp32 tree accumulation, fp32 output, autocast and TF32 off.

        Fails closed on a device whose ``addcmul`` is not a correctly-rounded FMA;
        CPU is the bit-defining device for this contract.
        """
        require_single_rounding_fma(q.device)
        with _strict_fp32_math(q.device.type):
            return self._run(
                q,
                k,
                scale=scale,
                query_padding_mask=query_padding_mask,
                key_padding_mask=key_padding_mask,
                output_dtype=torch.float32,
            )

    @staticmethod
    def _run(
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        scale: Optional[float],
        query_padding_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        validate_joint_attn_qk_gemm_inputs(
            q,
            k,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
        )
        resolved = scale if scale is not None else (1.0 / math.sqrt(q.size(3)))
        return _JointAttnQkGemmTreeFunction.apply(
            q, k, resolved, query_padding_mask, key_padding_mask, output_dtype
        )


@contextmanager
def _strict_fp32_math(device_type: str):
    """Disable autocast and TF32 for a true fp32 path, restoring state after."""
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


__all__ = [
    "JOINT_ATTN_QK_GEMM_CONTRACT_VERSION",
    "JointAttnQkGemmPlan",
    "LEAF_WIDTH",
    "NativeJointAttnQkGemmOp",
    "SUPPORTED_DTYPES",
    "dk_tree_fp32",
    "joint_attn_qk_gemm_backward_with_leaves",
    "fma_is_single_rounding",
    "dq_tree_fp32",
    "finalize_scaled_output",
    "joint_attn_qk_gemm_reference_backward",
    "num_leaves",
    "qk_scores_tree_fp32",
    "require_single_rounding_fma",
    "tree_from_leaf_evaluator",
    "validate_joint_attn_qk_gemm_inputs",
]
