# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 shared expert on DeepGEMM FP8 (SM90), matching the vLLM rollout.

Forward, as vLLM's ``DeepseekV4MLP`` runs it::

    x  (BF16)  -> fp8_group_quant -> DeepGEMM fp8_gemm_nt(W1) -> z  (BF16 [T, 2F])
    z          -> silu_and_mul_clamp(limit 10)                -> h  (BF16 [T, F])
    h          -> fp8_group_quant -> DeepGEMM fp8_gemm_nt(W2) -> y  (BF16 [T, H])

The SwiGLU and the second quantization run as one CUDA kernel
(``csrc/cuda/activation/silu_and_mul_clamp.cu``) whose bytes equal vLLM's two
separate kernels; the SwiGLU backward is a CUDA kernel too.

Both GEMMs (forward and dgrad) are :class:`~rl_engine.kernels.ops.cuda.fp8_linear.Fp8Linear`,
the DSv4 dense FP8 linear shared with the attention projections.

Weights are 128x128-block FP8 with UE8M0-valued FP32 scales, the checkpoint
format; :func:`prepare_shared_weights` quantizes BF16 weights the same way, and
:meth:`DeepGemmSharedExpertProvider.prepare_fp8` takes checkpoint tensors.

The shared-expert weights are frozen, so backward returns dX only (P5
contract), as an FP32 tensor::

    dh = dgrad(dy, W2.T) ; dz = silu_and_mul_clamp_backward(dh, z) ; dX = dgrad(dz, W1.T)

``backward="fp8"`` (default) runs both dgrads on DeepGEMM FP8 with the
transposed FP8 weights (block-quantizing ``W.T`` equals transposing the
block-quantized ``W``, so they are exact). ``backward="bf16_det_gemm"`` runs
them on det_gemm in BF16 against the dequantized weights (exact for UE8M0
scales), as the reference for judging FP8 gradients.

Every step is row-local or a DeepGEMM GEMM verified batch-invariant on SM90
(``scripts/probe_dsv4_deepgemm_bi.py``), so outputs and dX are batch-invariant.
Numeric profile ``dsv4-deepgemm-fp8-v1``; it is not byte-equal to the P5 FP32
oracle, whose shared-expert SwiGLU applies no clamp. Fail-closed: no fallback
to another implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.activation import silu_and_mul_clamp as cuda_swiglu
from rl_engine.kernels.ops.cuda.deepgemm import require_deepgemm
from rl_engine.kernels.ops.cuda.fp8_linear import Fp8BlockWeight, Fp8Linear, quantize_weight
from rl_engine.kernels.ops.cuda.quant import Fp8GroupQuantCudaOp
from rl_engine.kernels.ops.pytorch.activation.silu_and_mul_clamp import SWIGLU_LIMIT
from rl_engine.moe.contract import SharedBatch
from rl_engine.moe.provider import ReferenceProvider

PROFILE = "dsv4-deepgemm-fp8-v1"
BACKWARD_MODES = ("fp8", "bf16_det_gemm")


@dataclass
class PreparedSharedWeights:
    """fc1 ``[2F, H]`` and fc2 ``[H, F]`` as FP8 linears, BF16 copies for the fallback."""

    fc1: Fp8Linear
    fc2: Fp8Linear
    _bf16: tuple[Tensor, Tensor] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        two_ffn, hidden = self.fc1.weight.shape
        if tuple(self.fc2.weight.shape) != (hidden, two_ffn // 2):
            raise ValueError(
                f"w2 shape {tuple(self.fc2.weight.shape)} != {(hidden, two_ffn // 2)} for w1 "
                f"{tuple(self.fc1.weight.shape)}"
            )

    def bf16(self) -> tuple[Tensor, Tensor]:
        if self._bf16 is None:
            self._bf16 = (self.fc1.weight.dequantize(), self.fc2.weight.dequantize())
        return self._bf16


def prepare_shared_weights(w_fc1: Tensor, w_fc2: Tensor) -> PreparedSharedWeights:
    return PreparedSharedWeights(
        Fp8Linear(quantize_weight(w_fc1)), Fp8Linear(quantize_weight(w_fc2))
    )


class DeepGemmSharedExpertProvider(ReferenceProvider):
    """P5 shared-expert provider on DeepGEMM FP8 (forward and dX)."""

    name = "shared-expert-deepgemm-fp8"
    numeric_profile = PROFILE

    def __init__(self, backward: str = "fp8", swiglu_limit: float = SWIGLU_LIMIT) -> None:
        if backward not in BACKWARD_MODES:
            raise ValueError(f"backward must be one of {BACKWARD_MODES}, got {backward!r}")
        self.backward = backward
        self.swiglu_limit = float(swiglu_limit)
        self._capability = require_deepgemm()[1]
        Fp8GroupQuantCudaOp()  # fail closed without the quantizer
        cuda_swiglu._extension()  # fail closed without the SwiGLU kernels
        self._det = None
        if backward == "bf16_det_gemm":
            from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

            if not _EXT_AVAILABLE or _C is None or not hasattr(_C, "det_gemm_fwd"):
                raise NotImplementedError("rl_engine._C lacks det_gemm_fwd (fail-closed)")
            self._det = _C
        # BF16 weights are frozen; keep each prepared copy with its source tensors.
        self._prepared: dict[tuple, tuple[Tensor, Tensor, PreparedSharedWeights]] = {}

    # Weights -------------------------------------------------------------
    def prepare(self, w_fc1: Tensor, w_fc2: Tensor) -> PreparedSharedWeights:
        key = (
            w_fc1.data_ptr(),
            w_fc1._version,
            tuple(w_fc1.shape),
            w_fc2.data_ptr(),
            w_fc2._version,
            tuple(w_fc2.shape),
            w_fc1.device,
        )
        entry = self._prepared.get(key)
        if entry is None:
            entry = (w_fc1, w_fc2, prepare_shared_weights(w_fc1, w_fc2))
            self._prepared[key] = entry
        return entry[2]

    @staticmethod
    def prepare_fp8(w1: tuple[Tensor, Tensor], w2: tuple[Tensor, Tensor]) -> PreparedSharedWeights:
        """Checkpoint FP8 weights: ``(q, scale)`` pairs for fc1 and fc2."""
        return PreparedSharedWeights(Fp8Linear(Fp8BlockWeight(*w1)), Fp8Linear(Fp8BlockWeight(*w2)))

    # Provider surface ------------------------------------------------------
    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "operators": ["shared_expert_mlp_fwd", "shared_expert_mlp_bwd"],
            "geometry": ["one-row", "packed"],
            "devices": ["cuda"],
        }

    def provenance(self) -> dict[str, Any]:
        cap = self._capability
        return {
            "requested_backend": self.name,
            "actual_backend": self.name,
            "numeric_profile": self.numeric_profile,
            "torch_version": torch.__version__,
            "deep_gemm": cap.version,
            "deep_gemm_pinned": cap.pinned,
            "jit_nvcc": ".".join(map(str, cap.nvcc_version)),
            "activation_quant": "fp8_group_quant (1x128, UE8M0, vLLM per_token_group_fp8_quant)",
            "weights": "FP8 E4M3, 128x128 blocks, UE8M0-valued FP32 scales",
            "swiglu": f"vLLM silu_and_mul_with_clamp, limit {self.swiglu_limit}",
            "backward": self.backward,
            "split_k": 1,
        }

    def _check_batch(self, batch: SharedBatch) -> None:
        batch.validate()
        if batch.numeric_profile != self.numeric_profile:
            raise NotImplementedError(
                f"{self.name} implements {self.numeric_profile!r}, "
                f"got {batch.numeric_profile!r} (fail-closed, no fallback)"
            )
        if not batch.x.is_cuda:
            raise NotImplementedError(
                f"{self.name} requires CUDA tensors, got device {batch.x.device} "
                "(fail-closed, no fallback)"
            )

    def forward(self, x: Tensor, weights: PreparedSharedWeights) -> tuple[Tensor, dict[str, Any]]:
        x = x.contiguous()
        z = weights.fc1(x)
        if not z.shape[0]:
            empty = torch.empty(0, weights.fc2.out_features, device=x.device, dtype=torch.bfloat16)
            return empty, {"z": z, "weights": weights}
        h_q = cuda_swiglu.silu_and_mul_clamp_quant_cuda(z, self.swiglu_limit)
        y = weights.fc2.forward_quantized(h_q)
        return y, {"z": z, "weights": weights}

    def backward_dx(self, dy: Tensor, saved: dict[str, Any]) -> Tensor:
        weights: PreparedSharedWeights = saved["weights"]
        dy = dy.to(torch.bfloat16).contiguous()
        if not dy.shape[0]:  # nothing to launch on any backward path
            return torch.empty(0, weights.fc1.in_features, device=dy.device, dtype=torch.float32)
        if self.backward == "fp8":
            dh = weights.fc2.backward_dx(dy)
        else:
            dh = self._det.det_gemm_fwd(dy, weights.bf16()[1])
        dz = cuda_swiglu.silu_and_mul_clamp_backward_cuda(dh, saved["z"], self.swiglu_limit)
        if self.backward == "fp8":
            return weights.fc1.backward_dx(dz).float()
        return self._det.det_gemm_fwd(dz, weights.bf16()[0]).float()

    def shared_expert_mlp_fwd(self, batch: SharedBatch) -> tuple[Tensor, dict[str, Any]]:
        self._check_batch(batch)
        return self.forward(batch.x, self.prepare(batch.w_fc1, batch.w_fc2))

    def shared_expert_mlp_bwd(
        self, dy: Tensor, batch: SharedBatch, saved: dict[str, Any]
    ) -> Tensor:
        self._check_batch(batch)
        return self.backward_dx(dy, saved)


class DeepGemmSharedExpertBf16BwdProvider(DeepGemmSharedExpertProvider):
    """Same forward; dX on det_gemm BF16, the reference for FP8 gradients."""

    name = "shared-expert-deepgemm-fp8-bf16bwd"

    def __init__(self) -> None:
        super().__init__(backward="bf16_det_gemm")
