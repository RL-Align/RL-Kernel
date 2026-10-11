# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Qwen3-Next RMSNorm references (WS1 ground truth for RFC #428 C1).

Qwen3-Next ships two RMSNorm conventions. They differ in how the weight is applied
and in where the dtype casts sit, so they are separate operators rather than a flag:

``Qwen3NextRMSNorm`` (decoder / final norm)
    Normalize in fp32, scale by ``(1 + weight)`` in fp32, cast once at the end. The
    stored weight is zero-centred, so the ``1 +`` must be applied after the fp32
    upcast; folding it into a bf16 weight first rounds the offset.

``Qwen3NextRMSNormGated`` (inside the Gated DeltaNet block)
    Normalize in fp32, scale by a plain weight, then gate by ``silu(gate)``.
    vLLM's ``RMSNormGated`` multiplies the weight in fp32; transformers casts the
    normalized value back to the input dtype first. ``Qwen3NextRMSNormGatedOp``
    follows vLLM; ``Qwen3NextRMSNormGatedHFOp`` keeps the transformers convention
    as a witness, and ``test_gated_conventions_diverge_in_low_precision`` pins that
    the two differ in bf16 and agree bitwise in fp32.

All three reuse :func:`shape_invariant_rstd`, a fixed-order reduction. They
reproduce the weight convention and cast order, not vLLM's reduction tree, and are
not bitwise equal to any vLLM path probed so far. Claim levels, measurements and
limitations are in ``docs/operators/qwen3-next-rms-norm.md`` and, for the gated
pair, ``docs/operators/qwen3-next-rms-norm-gated.md``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rl_engine.reference.norm.rms_norm import (
    NativeRMSNormOp,
    check_norm_weight,
    shape_invariant_rstd,
)

__all__ = [
    "Qwen3NextRMSNormOp",
    "Qwen3NextRMSNormGatedOp",
    "Qwen3NextRMSNormGatedHFOp",
]


class Qwen3NextRMSNormOp(NativeRMSNormOp):
    """Zero-centred RMSNorm: ``out = x * rstd * (1 + weight)``.

    Only the weight convention differs from :class:`NativeRMSNormOp`, so that is
    all this overrides. The base applies the offset in fp32, after the upcast,
    which is what ``transformers`` and vLLM both do -- folding ``1 +`` into a
    bf16 weight beforehand would round the offset away.
    """

    weight_offset = 1.0


class Qwen3NextRMSNormGatedOp:
    """Gated RMSNorm used by the Gated DeltaNet block (vLLM/strict convention).

    ``out = (x * rstd * weight) * silu(gate)``, with every multiply in fp32 and
    a single cast on the way out. This is the convention vLLM's ``RMSNormGated``
    uses with ``norm_before_gate=True``. Which gated convention is the strict
    default is still open; see ``docs/operators/qwen3-next-rms-norm-gated.md``.

    Not a subclass of the plain op: it takes an extra tensor and its epilogue
    differs, so it is not a drop-in substitute for one.

    The weight is plain, NOT zero-centred -- upstream initializes it to ones.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        gate: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        return self.forward(x, weight, gate, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        gate: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        return self._rms_norm_gated(x, weight, gate, eps=eps, output_dtype=x.dtype)

    def forward_fp32(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        gate: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Ground-truth: fp32 in, fp32 out."""
        return self._rms_norm_gated(x, weight, gate, eps=eps, output_dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Shared by both conventions; only `_scale_by_weight` differs.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalized(
        x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, *, eps: float
    ) -> torch.Tensor:
        check_norm_weight(x, weight)
        if gate.shape != x.shape:
            raise ValueError(
                f"gate must match x, got tuple(gate.shape)={tuple(gate.shape)} "
                f"vs tuple(x.shape)={tuple(x.shape)}"
            )
        if gate.dtype != x.dtype or gate.device != x.device:
            raise ValueError("gate must have the same dtype and device as x")
        x_f = x.float()
        rstd = shape_invariant_rstd(x_f, float(eps)).unsqueeze(-1)
        return x_f * rstd

    @staticmethod
    def _scale_by_weight(
        normed: torch.Tensor, weight: torch.Tensor, input_dtype: torch.dtype
    ) -> torch.Tensor:
        """vLLM: the weight multiply stays in fp32."""
        del input_dtype
        return normed * weight.float()

    @classmethod
    def _rms_norm_gated(
        cls,
        x: torch.Tensor,
        weight: torch.Tensor,
        gate: torch.Tensor,
        *,
        eps: float,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        normed = cls._normalized(x, weight, gate, eps=eps)
        scaled = cls._scale_by_weight(normed, weight, x.dtype)
        # The gate activation is evaluated in fp32 and promotes the product.
        gated = scaled * F.silu(gate.float())
        return gated.to(output_dtype)


class Qwen3NextRMSNormGatedHFOp(Qwen3NextRMSNormGatedOp):
    """``transformers`` gated convention: weight multiply in the input dtype.

    Kept so the HF-vs-vLLM divergence documented in the module docstring stays
    covered by a test rather than discovered in a drift report. Do NOT use this
    for an L2 claim against vLLM rollout.
    """

    @staticmethod
    def _scale_by_weight(
        normed: torch.Tensor, weight: torch.Tensor, input_dtype: torch.dtype
    ) -> torch.Tensor:
        """transformers: round-trip through the input dtype first."""
        return weight * normed.to(input_dtype)
