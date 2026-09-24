# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA backend for the routed-expert base GEMM and the LoRA delta.

``CudaP5GemmProvider`` overrides two operators and leaves the other seven on
the FP32 oracle, so ``scripts/check_p5.py`` still runs end to end:

===========================  ===============================================
``mxfp8_mxfp4_grouped_gemm`` ``csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm.cu``
``shared_grouped_lora_delta``composed from the ``det_gemm`` primitives
===========================  ===============================================

The two operators do NOT share a reduction, so a single ``numeric_profile``
string cannot describe both honestly:

* the grouped GEMM is a strict one-row FFMA kernel, byte-exact with the oracle;
* the LoRA delta reuses ``det_gemm``'s mid-split K-tree, which rounds every
  32-wide leaf to BF16 and therefore sits ~5e-3 from the oracle. That tree is
  what makes the result survive tensor parallelism (a contiguous half-K shard
  is one child of the tree), and it is batch-invariant, but it is not
  byte-equal.

The class therefore declares a composite profile and reports the per-operator
profiles in :meth:`provenance`. Callers that need byte-equality on the LoRA
delta should use :class:`~rl_engine.moe.backends.lora_delta.LoRADeltaTritonProvider`
instead, which keeps one FP32 accumulator across the whole K loop.

This is a reference backend. The production path for the routed expert is the
single fused kernel in :mod:`rl_engine.moe.backends.sm90_fused_mlp`.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.moe.backends.lora_delta import (
    DET_GEMM_PROFILE as _LORA_DELTA_PROFILE,
    det_gemm_lora_delta_bwd,
    det_gemm_lora_delta_fwd,
)
from rl_engine.moe.mx_format import MXTensor
from rl_engine.moe.provider import ReferenceProvider

_GROUPED_GEMM_PROFILE = "p5-strict-ffma-v1"


class CudaP5GemmProvider(ReferenceProvider):
    """Strict CUDA grouped GEMM plus the det_gemm-composed LoRA delta.

    Fail-closed: raises if the compiled extension is missing instead of
    silently falling back to the Python oracle.
    """

    name = "cuda-p5-gemm"
    numeric_profile = "p5-cuda-composite-v1"

    @staticmethod
    def _extension():
        try:
            from rl_engine import _C
        except ImportError as exc:
            raise RuntimeError("P5 CUDA extension is not built") from exc
        return _C

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "cuda-p5-strict",
            "geometry": ["one-row"],
            "devices": ["cuda"],
            "implemented": [
                "mxfp8_mxfp4_grouped_gemm_fwd",
                "mxfp8_mxfp4_grouped_gemm_bwd",
                "shared_grouped_lora_delta_fwd",
                "shared_grouped_lora_delta_bwd",
            ],
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": "cuda-strict",
            "numeric_profile": self.numeric_profile,
            # The two operators reduce differently; see the module docstring.
            "operator_profiles": {
                "mxfp8_mxfp4_grouped_gemm": _GROUPED_GEMM_PROFILE,
                "shared_grouped_lora_delta": _LORA_DELTA_PROFILE,
            },
            "geometry": "one-row-unpadded",
            "tile": None,  # no tiling in the strict one-row path
            "split_k": 1,  # no split-K in either operator
            "kernel_fingerprint": "mxfp8-mxfp4-grouped-gemm-strict-v1",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }

    # -- base GEMM: strict one-row FFMA kernel, byte-exact with the oracle ---

    def mxfp8_mxfp4_grouped_gemm_fwd(
        self, a: MXTensor, w: MXTensor, expert_offsets: torch.Tensor
    ) -> torch.Tensor:
        return self._extension().moe_mxfp8_mxfp4_grouped_gemm_forward(
            a.codes, a.scales, w.codes, w.scales, expert_offsets
        )

    def mxfp8_mxfp4_grouped_gemm_bwd(
        self, dy: torch.Tensor, w: MXTensor, expert_offsets: torch.Tensor
    ) -> torch.Tensor:
        return self._extension().moe_mxfp8_mxfp4_grouped_gemm_backward(
            dy, w.codes, w.scales, expert_offsets
        )

    # -- LoRA delta: composed from det_gemm, batch-invariant, not byte-equal --
    # Shared with LoRADeltaCudaProvider so the composition has one definition.

    shared_grouped_lora_delta_fwd = staticmethod(det_gemm_lora_delta_fwd)
    shared_grouped_lora_delta_bwd = staticmethod(det_gemm_lora_delta_bwd)
