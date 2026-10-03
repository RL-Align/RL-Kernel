# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Portable fused routed-expert MLP (MXFP8 x MXFP4, Triton) -- inference forward.

The ROCm-facing counterpart of :mod:`rl_engine.moe.backends.sm90_fused_mlp`,
with the same call surface (``fc1_swiglu_quant`` / ``fc3`` / ``forward``) so
the two can be swapped behind one dispatch. There is no ``fc1_z``: the CUDA
backend has one to validate its fc1 in isolation, and this backend is validated
against the oracle end to end instead. Nothing here is
Hopper-specific: no TMA, no ``wgmma``, no shared-memory choreography, and no
FP8 ``tl.dot`` -- ``tl.float8e4nv`` is NVIDIA's e4m3**fn** whereas CDNA's
native FP8 is e4m3**fnuz**, so the kernels decode both MX operands to BF16
with integer arithmetic and run a plain BF16 ``tl.dot``. That decode is exact
(E4M3 keeps 4 significand bits, E2M1 keeps 2, BF16 has 8, and the E8M0 scale is
a power of two), and the block scale is folded out of the inner sum, so no
dequantized tensor is ever materialized.

Numeric profile ``p5-triton-fused-v1``: deterministic and batch-invariant, and
deliberately NOT byte-aligned to either ``oracle-fp32-serial-v1`` or the CUDA
backend's ``p5-sm90-fused-mlp-v1``. Measured against an FP64 reference at
E=8 H=4096 F=2048 it lands at the same max relative error as the CUDA path
(3.4e-2 at M=512, 3.7e-2 at M=2048 -- the MX quantization floor, identical to
four digits), and it is ~6x slower on an H100 because the CUDA kernel has FP8
tensor cores, scale folding and warp specialization that this one gives up for
portability. Tuning happens on a CDNA host; see ``docs/operators/dsv4-moe.md``.

Fail-closed: no oracle fallback. A batch must carry this provider's numeric
profile, base weights only (no LoRA), and device tensors.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.moe.backends.routed_checks import check_routed_batch
from rl_engine.moe.contract import ExpertBatch
from rl_engine.moe.mx_format import MX_BLOCK, MXTensor

PROFILE = "p5-triton-fused-v1"


class TritonFusedMoeMlp:
    """Routed-expert MLP forward: y = fc3(mx_quant(swiglu(fc1(x_q)) * p_s))."""

    name = "triton-fused-moe-mlp"
    numeric_profile = PROFILE

    def __init__(self) -> None:
        from rl_engine.kernels.ops.triton.moe import fused_mlp as tf
        from rl_engine.kernels.ops.triton.moe import shared_expert as probe

        if not probe.TRITON_AVAILABLE:
            raise NotImplementedError("triton is not installed (fail-closed, no fallback)")
        self._tf = tf

    # ------------------------------------------------------------ metadata
    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "operators": ["routed_mlp_fwd"],
            "paths": ["two_launch"],  # no single-launch variant: Triton has no
            # user-managed shared memory, so h_q cannot be kept resident
            "geometry": ["packed"],
            "devices": ["cuda", "rocm"],
            "lora": False,
            "backward": False,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": self.name,
            "numeric_profile": self.numeric_profile,
            "torch_version": torch.__version__,
            "tensor_core": "tl.dot bf16 (MX operands decoded exactly, scale folded)",
            "tiles": {
                k: self._tf.tiles(k, torch.device("cuda", torch.cuda.current_device()))
                for k in ("routed_fc1", "routed_fc3")
            },
            "mx_scaling": "exact per 32-block: both scales folded out of the inner sum",
            "split_k": 1,
            "batch_invariant": True,
        }

    # ------------------------------------------------------------ weights
    def prepare(self, w: MXTensor) -> MXTensor:
        """No-op: the kernels read the MXFP4 codes and E8M0 scales directly.

        The CUDA backend pre-folds the weight scale into its FP8 bytes to save
        an FMA per element inside a WGMMA pipeline. Here the decode already
        produces BF16, so there is nothing to fold and nothing to cache.
        """
        if w.elem_format != "e2m1":
            raise TypeError(f"expected an e2m1 (MXFP4) weight, got {w.elem_format!r}")
        return w

    # ------------------------------------------------------------ checks
    def _check_batch(self, batch: ExpertBatch, x_q: MXTensor) -> None:
        check_routed_batch(batch, x_q, name=self.name, profile=PROFILE, align=MX_BLOCK)
        # The pinned K tiles are per architecture, so the alignment is too.
        dev = batch.x.device
        bk1 = self._tf.tiles("routed_fc1", dev)["BK"]
        bk3 = self._tf.tiles("routed_fc3", dev)["BK"]
        if batch.hidden % bk1 or batch.ffn % bk3:
            raise NotImplementedError(
                f"{self.name}: hidden must be a multiple of {bk1} and ffn of {bk3} on this "
                f"device, got {batch.hidden} / {batch.ffn}"
            )

    # ------------------------------------------------------------ forward
    def fc1_swiglu_quant(self, batch: ExpertBatch, x_q: MXTensor) -> MXTensor:
        """fc1 with clamp-SwiGLU * p_s and MX re-quantization fused: h_q [M, F]."""
        self._check_batch(batch, x_q)
        return self._tf.routed_fc1_swiglu_quant(
            x_q, batch.w1, batch.expert_offsets, batch.p_s.contiguous()
        )

    def fc3(self, batch: ExpertBatch, h_q: MXTensor) -> torch.Tensor:
        """fc3 (down projection): BF16 y [M, H]."""
        return self._tf.routed_fc3(h_q, batch.w2, batch.expert_offsets)

    def forward(self, batch: ExpertBatch, x_q: MXTensor, path: str = "two_launch") -> torch.Tensor:
        """Routed MLP forward for base weights. Only ``two_launch`` exists."""
        self._check_batch(batch, x_q)
        if path != "two_launch":
            raise ValueError(
                f"unknown path {path!r}: this backend has only 'two_launch' "
                "(Triton cannot keep h_q in shared memory across the two GEMMs)"
            )
        return self.fc3(batch, self.fc1_swiglu_quant(batch, x_q))
