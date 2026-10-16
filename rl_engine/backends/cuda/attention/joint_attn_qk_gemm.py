# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA backend for the Qwen-Image WS1 joint attention QK GEMM (issue #386).

``scores = single_cast((Q Kᵀ) * scale)`` over the non-causal ``[text, image]``
joint sequence, under contract ``joint-attn-qk-gemm-tree-v1``. The authoritative
contract definition lives in
``rl_engine.reference.attention.joint_attn_qk_gemm``.

Forward calls the device kernel, which closes the whole head-dim tree in
registers with explicit ``__fmaf_rn`` chains. Backward calls one device kernel
per 32-wide sequence leaf and then merges the partials through
``joint_attn_qk_gemm_backward_with_leaves`` -- the *reference's own* host tree
and padding-compaction code. The merge order is therefore shared by
construction, not reimplemented, which is what makes dQ/dK byte-equal to the
fp32 CPU gold.

Unsupported head-dim geometry is an explicitly declared fallback to the
reference, recorded on the plan readback, never a silent one.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.reference.attention.joint_attn_qk_gemm import (
    JOINT_ATTN_QK_GEMM_CONTRACT_VERSION,
    LEAF_WIDTH,
    JointAttnQkGemmPlan,
    NativeJointAttnQkGemmOp,
    joint_attn_qk_gemm_backward_with_leaves,
    validate_joint_attn_qk_gemm_inputs,
)
from rl_engine.utils.logger import logger

_REQUIRED_SYMBOLS = (
    "joint_attn_qk_gemm_forward",
    "joint_attn_qk_gemm_dq_leaf",
    "joint_attn_qk_gemm_dk_leaf",
)
_MAX_LEAVES = 32  # mirrors kMaxLeaves in csrc/cuda/attention/joint_attn_qk_gemm.cu


def cuda_geometry_support(head_dim: int) -> tuple[bool, Optional[str]]:
    """Whether the device forward kernel covers ``head_dim``, and why not if it does not.

    The in-register halving combine equals the mid-split tree only for a
    power-of-two leaf count, so other head dims fall back rather than silently
    using a different reduction order. Qwen-Image runs ``head_dim = 128``.
    """
    if head_dim % LEAF_WIDTH != 0:
        return False, (
            f"head_dim={head_dim} is not a multiple of the {LEAF_WIDTH}-wide leaf; "
            "the device tree would need a short tail leaf"
        )
    leaf_count = head_dim // LEAF_WIDTH
    if leaf_count > _MAX_LEAVES:
        return False, f"head_dim={head_dim} exceeds the {_MAX_LEAVES * LEAF_WIDTH} register budget"
    if leaf_count & (leaf_count - 1):
        return False, (
            f"head_dim={head_dim} gives {leaf_count} leaves, which is not a power of two; "
            "the in-register halving combine would not equal the mid-split tree"
        )
    return True, None


def _dq_leaf_cuda(g_f: torch.Tensor, k_f: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """One dQ sequence leaf on the device, returned as an fp32 partial."""
    return _C.joint_attn_qk_gemm_dq_leaf(g_f, k_f, lo, hi)


def _dk_leaf_cuda(g_f: torch.Tensor, q_f: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """One dK sequence leaf on the device, returned as an fp32 partial."""
    return _C.joint_attn_qk_gemm_dk_leaf(g_f, q_f, lo, hi)


class _JointAttnQkGemmCudaFunction(Function):
    """Device forward; backward = device leaves merged by the shared host tree."""

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
        scores = _C.joint_attn_qk_gemm_forward(q, k, scale, output_dtype == torch.float32)
        ctx.save_for_backward(q, k, query_padding_mask, key_padding_mask)
        ctx.scale = scale
        return scores

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx, grad_scores: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
        q, k, query_padding_mask, key_padding_mask = ctx.saved_tensors
        grad_q, grad_k = joint_attn_qk_gemm_backward_with_leaves(
            q,
            k,
            grad_scores,
            scale=ctx.scale,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            dq_leaf_fn=_dq_leaf_cuda,
            dk_leaf_fn=_dk_leaf_cuda,
        )
        return grad_q, grad_k, None, None, None, None


class JointAttnQkGemmCudaOp:
    """Joint QK GEMM on a CUDA GPU, byte-equal to the fp32 CPU gold.

    Public surface matches ``NativeJointAttnQkGemmOp`` so the validation harness
    can call ``forward(**inputs)`` and ``forward_fp32(**inputs)`` interchangeably.
    """

    backend_id = "rlkernel.cuda.attention.joint_qk_gemm.v1"
    contract_version = JOINT_ATTN_QK_GEMM_CONTRACT_VERSION

    def __init__(self) -> None:
        if not _EXT_AVAILABLE:
            raise RuntimeError(
                "joint_attn_qk_gemm CUDA kernel is unavailable: the native extension (_C) "
                "did not load. Rebuild it for the active GPU platform."
            )
        missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(_C, name)]
        if missing:
            raise RuntimeError(
                f"joint_attn_qk_gemm CUDA kernel is unavailable: _C is missing {missing}. "
                "Rebuild the native extension."
            )
        self._reference = NativeJointAttnQkGemmOp()
        self._last_plan: Optional[JointAttnQkGemmPlan] = None
        logger.info("Successfully linked to _C.joint_attn_qk_gemm_forward/dq_leaf/dk_leaf.")

    @property
    def last_plan(self) -> Optional[JointAttnQkGemmPlan]:
        """Arithmetic policy actually used by the most recent call."""
        return self._last_plan

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
        """Device forward with the single RNE cast landing on the input dtype."""
        return self._run(
            q,
            k,
            scale=scale,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            output_dtype=q.dtype,
            strict_gold=False,
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
        """Device forward with an fp32 output, for comparison against the CPU gold."""
        return self._run(
            q,
            k,
            scale=scale,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            output_dtype=torch.float32,
            strict_gold=True,
        )

    def _run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        scale: Optional[float],
        query_padding_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        strict_gold: bool,
    ) -> torch.Tensor:
        validate_joint_attn_qk_gemm_inputs(
            q,
            k,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
        )
        resolved = scale if scale is not None else (1.0 / math.sqrt(q.size(3)))

        reason: Optional[str] = None
        if not q.is_cuda:
            reason = f"q is on {q.device.type}, not a CUDA device"
        else:
            supported, geometry_reason = cuda_geometry_support(int(q.size(3)))
            if not supported:
                reason = geometry_reason

        if reason is not None:
            # Declared fallback: recorded on the plan, never silent. The reference
            # is evaluated on CPU even when the inputs are on a GPU, because
            # `torch.addcmul` only behaves as a correctly-rounded FMA there; that
            # keeps the fallback's bytes equal to the gold instead of silently
            # switching to the device's double-rounded multiply-then-add. The
            # device copies stay inside autograd, so gradients still flow.
            reason = f"{reason}; reference evaluated on CPU to preserve gold bits"
            logger.warning(f"joint_attn_qk_gemm CUDA backend falling back to reference: {reason}")
            self._last_plan = JointAttnQkGemmPlan(
                backend_id=self._reference_backend_id(),
                fallback=True,
                fallback_reason=reason,
            )
            runner = self._reference.forward_fp32 if strict_gold else self._reference.forward
            scores = runner(
                q.cpu(),
                k.cpu(),
                scale=resolved,
                query_padding_mask=(
                    None if query_padding_mask is None else query_padding_mask.cpu()
                ),
                key_padding_mask=None if key_padding_mask is None else key_padding_mask.cpu(),
            )
            return scores.to(q.device)

        self._last_plan = JointAttnQkGemmPlan(backend_id=self.backend_id)
        return _JointAttnQkGemmCudaFunction.apply(
            q, k, resolved, query_padding_mask, key_padding_mask, output_dtype
        )

    @staticmethod
    def _reference_backend_id() -> str:
        """Backend id recorded when the call falls back to the PyTorch reference."""
        return "rlkernel.reference.attention.joint_qk_gemm.v1"


__all__ = ["JointAttnQkGemmCudaOp", "cuda_geometry_support"]
