# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""SM90 fused routed-expert MLP (MXFP8 x MXFP4, FP8 WGMMA) -- inference forward.

Numeric profile ``p5-sm90-fused-mlp-v1``: deterministic and batch-invariant,
NOT byte-aligned to ``oracle-fp32-serial-v1`` (tensor-core accumulation order
inside each 32-wide MX block). See ``docs/operators/dsv4-moe.md``.

Two execution paths share one GEMM core (``csrc/cuda/moe/sm90_fused_moe_mlp.cu``):

* ``two_launch``: fc1 with SwiGLU * p_s and MX re-quantization fused into the
  epilogue (h_q written to global memory), then fc3. This is the fast path.
* ``fused``: a single launch with h_q resident in shared memory. Bit-identical
  to ``two_launch``; currently slower (two pipeline stages fit next to the
  128 KB h_q), kept as the reference for the follow-up BK=64 variant.

Fail-closed: no oracle fallback. A batch must carry this provider's numeric
profile, base weights only (no LoRA), and CUDA tensors on an SM90 device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.moe.contract import ExpertBatch
from rl_engine.moe.mx_format import MXTensor

PROFILE = "p5-sm90-fused-mlp-v1"
_SYMBOLS = (
    "sm90_moe_prepare_weight_ref",
    "sm90_moe_fc1_forward",
    "sm90_moe_fc1_swiglu_quant_forward",
    "sm90_moe_fc3_forward",
    "sm90_moe_fused_mlp_forward",
)


def _extension():
    try:
        from rl_engine import _C
    except ImportError as exc:  # fail-closed
        raise NotImplementedError("rl_engine._C is not built") from exc
    missing = [s for s in _SYMBOLS if not hasattr(_C, s)]
    if missing:
        raise NotImplementedError(
            f"rl_engine._C lacks {missing}; rebuild with KERNEL_ALIGN_MOE_SM90=1 on an SM90 host"
        )
    return _C


@dataclass(frozen=True)
class PreparedWeight:
    """An MXFP4 expert weight plus the per-column reference exponents and
    per-block residuals the kernel folds into the FP8 operand (computed once;
    weights are frozen)."""

    codes: torch.Tensor  # uint8 [E, N, K/2]
    scales: torch.Tensor  # uint8 [E, N, K/32]
    ref: torch.Tensor  # uint8 [E, N]
    res: torch.Tensor  # uint8 [E, N, K/32], (scale - ref) + 8 in [0, 14]


class Sm90FusedMoeMlp:
    """Routed-expert MLP forward: y = fc3(mx_quant(swiglu(fc1(x_q)) * p_s))."""

    name = "sm90-fused-moe-mlp"
    numeric_profile = PROFILE

    def __init__(self) -> None:
        self._ext = _extension()
        self._cache: dict[tuple[int, int], PreparedWeight] = {}

    # ------------------------------------------------------------ metadata
    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "operators": ["routed_mlp_fwd"],
            "paths": ["two_launch", "fused"],
            "geometry": ["packed"],
            "devices": ["cuda:sm90"],
            "lora": False,
            "backward": False,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": self.name,
            "numeric_profile": self.numeric_profile,
            "torch_version": torch.__version__,
            "tensor_core": "wgmma.m64n64k32.e4m3.e4m3.f32",
            "tiles": {"BM": 64, "BN": 128, "BK": 128, "stages": 4, "fused_stages": 2},
            "mx_scaling": "exact per 32-block: weight scale folded into FP8 (per-column ref), "
            "activation scale promoted on CUDA cores",
            "split_k": 1,
            "batch_invariant": True,
        }

    # ------------------------------------------------------------ weights
    def prepare(self, w: MXTensor) -> PreparedWeight:
        if w.elem_format != "e2m1":
            raise TypeError(f"expected an e2m1 (MXFP4) weight, got {w.elem_format!r}")
        key = (w.codes.data_ptr(), w.scales.data_ptr())
        hit = self._cache.get(key)
        if hit is not None and hit.codes is w.codes and hit.scales is w.scales:
            return hit
        ref, res = self._ext.sm90_moe_prepare_weight_ref(w.scales.contiguous())
        prepared = PreparedWeight(w.codes.contiguous(), w.scales.contiguous(), ref, res)
        self._cache[key] = prepared
        return prepared

    # ------------------------------------------------------------ checks
    def _check_batch(self, batch: ExpertBatch, x_q: MXTensor) -> None:
        """Cheap per-launch checks: host-side only, no device sync, no hashing.

        Full contract validation (``ExpertBatch.validate``) belongs where the
        batch is built: it walks the offsets on device and SHA-256s every
        weight byte, which costs far more than the kernels it guards.
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
        if batch.numeric_profile != PROFILE:
            raise NotImplementedError(
                f"{self.name} implements {PROFILE!r}; batch declares "
                f"{batch.numeric_profile!r} (fail-closed, no fallback)"
            )
        if batch.lora is not None:
            raise NotImplementedError(f"{self.name}: LoRA is not supported in v1 (base weights only)")
        if not batch.x.is_cuda:
            raise NotImplementedError(f"{self.name} requires CUDA tensors, got {batch.x.device}")
        if x_q.elem_format != "e4m3":
            raise TypeError("x_q must be an e4m3 (MXFP8) activation")
        if tuple(x_q.shape) != tuple(batch.x.shape):
            raise ValueError(f"x_q shape {x_q.shape} != x shape {tuple(batch.x.shape)}")
        if batch.hidden % 128 != 0 or batch.ffn % 128 != 0:
            raise NotImplementedError(
                f"{self.name}: hidden and ffn must be multiples of 128, got "
                f"{batch.hidden} / {batch.ffn}"
            )

    # ------------------------------------------------------------ forward
    def fc1_z(self, batch: ExpertBatch, x_q: MXTensor) -> torch.Tensor:
        """Unfused fc1 -> FP32 z [M, 2F] (gate | up), for validation."""
        self._check_batch(batch, x_q)
        w1 = self.prepare(batch.w1)
        return self._ext.sm90_moe_fc1_forward(
            x_q.codes, x_q.scales, w1.codes, w1.scales, w1.ref, w1.res, batch.expert_offsets
        )

    def fc1_swiglu_quant(self, batch: ExpertBatch, x_q: MXTensor) -> MXTensor:
        """fc1 with clamp-SwiGLU * p_s and MX re-quantization fused: h_q [M, F]."""
        self._check_batch(batch, x_q)
        w1 = self.prepare(batch.w1)
        codes, scales = self._ext.sm90_moe_fc1_swiglu_quant_forward(
            x_q.codes, x_q.scales, w1.codes, w1.scales, w1.ref, w1.res,
            batch.expert_offsets, batch.p_s.contiguous(),
        )
        return MXTensor(codes=codes, scales=scales, elem_format="e4m3", shape=(batch.rows, batch.ffn))

    def fc3(self, batch: ExpertBatch, h_q: MXTensor) -> torch.Tensor:
        """fc3 (down projection): BF16 y [M, H]."""
        w2 = self.prepare(batch.w2)
        return self._ext.sm90_moe_fc3_forward(
            h_q.codes, h_q.scales, w2.codes, w2.scales, w2.ref, w2.res, batch.expert_offsets
        )

    def forward(self, batch: ExpertBatch, x_q: MXTensor, path: str = "two_launch") -> torch.Tensor:
        """Routed MLP forward for base weights. ``path``: 'two_launch' | 'fused'."""
        self._check_batch(batch, x_q)
        if path == "two_launch":
            return self.fc3(batch, self.fc1_swiglu_quant(batch, x_q))
        if path == "fused":
            w1, w2 = self.prepare(batch.w1), self.prepare(batch.w2)
            return self._ext.sm90_moe_fused_mlp_forward(
                x_q.codes, x_q.scales, w1.codes, w1.scales, w1.ref, w1.res,
                w2.codes, w2.scales, w2.ref, w2.res, batch.expert_offsets, batch.p_s.contiguous(),
            )
        raise ValueError(f"unknown path {path!r} (expected 'two_launch' or 'fused')")
