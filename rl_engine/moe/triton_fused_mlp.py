# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Portable fused routed-expert MLP (MXFP8 x MXFP4, Triton) -- inference forward.

The ROCm-facing routed-expert path (``fc1_swiglu_quant`` / ``fc3`` /
``forward``), validated against the oracle end to end. Nothing here is
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

from rl_engine.moe.contract import ExpertBatch
from rl_engine.moe.mx_format import MX_BLOCK, MXTensor

PROFILE = "p5-triton-fused-v1"


def _check_routed_batch(
    batch: ExpertBatch,
    x_q: MXTensor,
    *,
    name: str,
    profile: str,
    align: int,
) -> None:
    """Cheap per-launch checks: host-side only, no device sync, no hashing.

    Full contract validation (``ExpertBatch.validate``) belongs where the batch
    is built: it walks the offsets on device and SHA-256s every weight byte,
    which costs far more than the kernels it guards.

    ``align`` is the backend's tile granularity -- ``hidden`` and ``ffn`` must
    both be multiples of it, so no launch ever needs a partial-K tile.
    """
    m, hidden = batch.x.shape
    if batch.p_s.dtype != torch.float32 or batch.p_s.shape != (m,):
        raise TypeError(f"p_s must be FP32 [{m}], got {batch.p_s.dtype} {tuple(batch.p_s.shape)}")
    if batch.expert_offsets.dtype != torch.int32 or batch.expert_offsets.dim() != 1:
        raise TypeError("expert_offsets must be a 1-D int32 tensor")
    n_experts = batch.expert_offsets.numel() - 1
    if tuple(batch.w1.shape) != (n_experts, 2 * batch.ffn, hidden):
        raise ValueError(f"w1 shape {batch.w1.shape} != {(n_experts, 2 * batch.ffn, hidden)}")
    if tuple(batch.w2.shape) != (n_experts, hidden, batch.ffn):
        raise ValueError(f"w2 shape {batch.w2.shape} != {(n_experts, hidden, batch.ffn)}")
    if batch.w1.elem_format != "e2m1" or batch.w2.elem_format != "e2m1":
        raise TypeError("base weights must be MXFP4 (e2m1)")
    if batch.numeric_profile != profile:
        raise NotImplementedError(
            f"{name} implements {profile!r}; batch declares "
            f"{batch.numeric_profile!r} (fail-closed, no fallback)"
        )
    if batch.lora is not None:
        raise NotImplementedError(f"{name}: LoRA is not supported in v1 (base weights only)")
    if not batch.x.is_cuda:
        raise NotImplementedError(f"{name} requires CUDA tensors, got {batch.x.device}")
    if x_q.elem_format != "e4m3":
        raise TypeError("x_q must be an e4m3 (MXFP8) activation")
    if tuple(x_q.shape) != tuple(batch.x.shape):
        raise ValueError(f"x_q shape {x_q.shape} != x shape {tuple(batch.x.shape)}")
    if batch.hidden % align != 0 or batch.ffn % align != 0:
        raise NotImplementedError(
            f"{name}: hidden and ffn must be multiples of {align}, got "
            f"{batch.hidden} / {batch.ffn}"
        )


class TritonFusedMoeMlp:
    """Routed-expert MLP forward: y = fc3(mx_quant(swiglu(fc1(x_q)) * p_s))."""

    name = "triton-fused-moe-mlp"
    numeric_profile = PROFILE

    def __init__(self) -> None:
        from rl_engine.kernels.ops.triton.moe import fused_mlp as tf
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
        _check_routed_batch(batch, x_q, name=self.name, profile=PROFILE, align=MX_BLOCK)
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
