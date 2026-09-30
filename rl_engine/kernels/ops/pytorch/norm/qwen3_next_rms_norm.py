# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Qwen3-Next RMSNorm references (WS1 ground truth for RFC #428 C1).

Qwen3-Next ships two RMSNorm conventions that differ both in how the weight is
applied and in where the dtype casts sit. They are kept as separate operators
because the cast order is part of the contract, not a flag:

``Qwen3NextRMSNorm`` (decoder / final norm)
    normalize in fp32, scale by ``(1 + weight)`` in fp32, cast once at the end.
    The stored weight is zero-centred, so the ``1 +`` offset MUST be applied
    after the fp32 upcast -- folding it into a bf16 weight first rounds the
    offset and silently breaks the bitwise claim.

``Qwen3NextRMSNormGated`` (inside the Gated DeltaNet block)
    normalize in fp32, scale by a plain (non zero-centred) weight, then gate by
    ``silu(gate)``. Where the weight multiply happens is NOT agreed upstream --
    see below -- so it is an explicit part of the operator identity here.

Divergence at the gated-norm boundary (measured)
-------------------------------------------------
``transformers`` and vLLM do not agree on the gated variant:

* vLLM (``RMSNormGated``, both ``forward_native`` and the FLA Triton
  ``forward_cuda``) keeps the normalized value in fp32 for the weight multiply.
* ``transformers`` (``Qwen3NextRMSNormGated``) casts the normalized value back
  to the input dtype *before* the weight multiply.

On B200 / bf16 / ``head_v_dim=128`` these differ in 35% of elements with
``max|diff| = 6.25e-2``; swapping only the cast order reproduces the gap
(``5.3e-2``), so the cast order -- not the reduction order -- dominates. Because
RFC #428 claims L2 exactness against **vLLM rollout**, the fp32 multiply is the
strict default; the transformers convention is kept as a named witness so the
divergence stays testable instead of being silently picked.

What "agrees with vLLM" means here, precisely
----------------------------------------------
Only the *convention* is reproduced, not the bits. vLLM's own two paths are not
bitwise equal to each other: over 40 seeds (bf16, ``head_v_dim=128``, 512 rows),
``forward_native`` and ``forward_cuda`` disagreed on 21, worst
``max|diff| = 1.56e-2``. Against this operator the figures were 6/40 and 18/40.
So "bitwise equal to vLLM" is undefined until a single provider is named, and
this operator does not claim it -- see the reduction-order note below.

Both reuse :func:`shape_invariant_rstd` so the reduction order -- and hence the
result -- never depends on the batch layout (RFC #428 section 6, item 2). The
reduction order therefore deliberately differs from upstream's ``mean(-1)``;
what is reproduced exactly is the weight convention and the cast order.

Why the chunked reduction is kept on CUDA too
---------------------------------------------
:func:`shape_invariant_rstd` was introduced for NPU, where ``mean``/``sum`` pick
shape-dependent kernels. Measured on B200 (sm_100) it is needed on CUDA as well:
over 20 seeds at ``H=2048`` in bf16 -- Qwen3-Next's own ``hidden_size`` and dtype
-- a plain ``mean(-1)`` broke slice invariance (``rstd(x[3:5]) != rstd(x)[3:5]``)
on 1 of 20, while the chunked reduction broke on 0 of 20. Failures were also seen
at ``H=5120`` in fp32.

The cost is that the decoder norm is NOT bitwise equal to stock vLLM: 7 elements
of 1048576 differ (``max|diff| = 1.56e-2``, bf16, H=2048). That gap is inherent --
matching stock vLLM bitwise would mean adopting a reduction that is itself not
batch-invariant, i.e. trading L1 for L2. These operators therefore claim L0 and L1
only; an L2 claim needs the strict provider on both sides, per RFC #428 section 1,
item 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.pytorch.norm.rms_norm import (
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
    uses with ``norm_before_gate=True``, which is what the RFC #428 L2 claim is
    measured against.

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
