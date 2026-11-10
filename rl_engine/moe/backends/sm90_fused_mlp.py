# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""SM90 fused routed-expert MLP (MXFP8 x MXFP4, FP8 WGMMA) -- inference forward.

Numeric profile ``p5-sm90-fused-mlp-v1``: deterministic and batch-invariant,
NOT byte-aligned to ``oracle-fp32-serial-v1`` (tensor-core accumulation order
inside each 32-wide MX block). See ``docs/operators/dsv4-moe.md``.

Two execution paths share one GEMM core (``csrc/cuda/moe/sm90_fused_moe_mlp.cu``):

* ``two_launch``: fc1 with SwiGLU * p_s and MX re-quantization fused into the
  epilogue (h_q written to global memory), then fc3. This is the fast path.
  With many rows per expert the folded FP8 weights are expanded once per call
  (``sm90_moe_expand_w8``, the bytes the kernel would build in shared memory)
  and every operand arrives by TMA; same bytes, so the choice may follow M.
* ``fused``: a single launch with h_q resident in shared memory. Bit-identical
  to ``two_launch``; currently slower (two pipeline stages fit next to the
  128 KB h_q), kept as the reference for the follow-up BK=64 variant.

Backward (``backward``, profile ``p5-sm90-bwd-bf16-v1``, frozen base, no dW):
fc1 is recomputed for ``z`` (FP32), then ``dh = BF16(dy @ W2)`` fused with the
clamp-SwiGLU backward (``dz``, ``dp_s``), then ``dx = dz @ W1`` (FP32), on a BF16
WGMMA grouped GEMM that expands MXFP4 to BF16 in shared memory exactly
(``csrc/cuda/moe/sm90_moe_mlp_backward.cu``). Batch-invariant and run-to-run
identical; close to, not byte-equal with, the oracle's serial reduction. With
many rows per expert the weights are expanded once per call into a BF16 scratch
(``sm90_moe_dequant_t``, same conversion code) instead of once per token tile;
both variants give the same bytes, so the choice may depend on the batch size.

Fail-closed: no oracle fallback. A batch must carry this provider's numeric
profile, base weights only (no LoRA), and CUDA tensors on an SM90 device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.moe.backends.routed_checks import check_routed_batch
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
_BWD_SYMBOLS = ("sm90_moe_bwd_dx", "sm90_moe_bwd_dh_swiglu", "sm90_moe_dequant_t")
_EXPAND_SYMBOL = "sm90_moe_expand_w8"
BWD_PROFILE = "p5-sm90-bwd-bf16-v1"
# The backward expands e2m1 * 2^(s - 127) to BF16; codes in [2, 252] keep every
# value a normal BF16 (exact).
_BWD_SCALE_RANGE = (2, 252)
# Expand W ahead of time when the average expert holds at least this many rows
# (several token tiles would re-convert the same weight tile) and the scratch
# (forward: E x N x K bytes of E4M3; backward: twice that in BF16; freed after the
# call) stays under the byte cap.
_PREEXPAND_ROWS = 256
_PREEXPAND_MAX_BYTES = 4 << 30


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

    def __init__(self, preexpand: bool | None = None) -> None:
        """``preexpand``: None picks per call; True / False force the weight
        pre-expansion (forward and backward) on / off. Same bytes either way."""
        self._ext = _extension()
        self._preexpand = preexpand
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
            "backward": hasattr(self._ext, "sm90_moe_bwd_dx"),
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": self.name,
            "numeric_profile": self.numeric_profile,
            "torch_version": torch.__version__,
            "tensor_core": "wgmma.m64n{64,128}k32.e4m3.e4m3.f32",
            "tiles": {
                "inline": {"BM": 64, "BN": 128, "BK": 128, "stages": 4},
                "preexpanded": {"BM": 128, "BN": 128, "BK": 128, "stages": 6},
                "fused_stages": 2,
            },
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
        check_routed_batch(batch, x_q, name=self.name, profile=PROFILE, align=128)

    # ------------------------------------------------------------ forward
    def fc1_z(self, batch: ExpertBatch, x_q: MXTensor) -> torch.Tensor:
        """Unfused fc1 -> FP32 z [M, 2F] (gate | up), for validation."""
        self._check_batch(batch, x_q)
        w1 = self.prepare(batch.w1)
        return self._ext.sm90_moe_fc1_forward(
            x_q.codes,
            x_q.scales,
            w1.codes,
            w1.scales,
            w1.ref,
            w1.res,
            batch.expert_offsets,
            self._expanded_w8(w1, batch.expert_offsets, batch.rows),
        )

    def fc1_swiglu_quant(self, batch: ExpertBatch, x_q: MXTensor) -> MXTensor:
        """fc1 with clamp-SwiGLU * p_s and MX re-quantization fused: h_q [M, F]."""
        self._check_batch(batch, x_q)
        w1 = self.prepare(batch.w1)
        codes, scales = self._ext.sm90_moe_fc1_swiglu_quant_forward(
            x_q.codes,
            x_q.scales,
            w1.codes,
            w1.scales,
            w1.ref,
            w1.res,
            batch.expert_offsets,
            batch.p_s.contiguous(),
            self._expanded_w8(w1, batch.expert_offsets, batch.rows),
        )
        return MXTensor(
            codes=codes, scales=scales, elem_format="e4m3", shape=(batch.rows, batch.ffn)
        )

    def fc3(self, batch: ExpertBatch, h_q: MXTensor) -> torch.Tensor:
        """fc3 (down projection): BF16 y [M, H]."""
        w2 = self.prepare(batch.w2)
        return self._ext.sm90_moe_fc3_forward(
            h_q.codes,
            h_q.scales,
            w2.codes,
            w2.scales,
            w2.ref,
            w2.res,
            batch.expert_offsets,
            self._expanded_w8(w2, batch.expert_offsets, batch.rows),
        )

    def _use_preexpand(self, w_shape: tuple[int, ...], rows: int, elem_bytes: int) -> bool:
        if self._preexpand is not None:
            return self._preexpand
        num_experts, n, k = w_shape
        return (
            rows >= _PREEXPAND_ROWS * num_experts
            and num_experts * n * k * elem_bytes <= _PREEXPAND_MAX_BYTES
        )

    def _expanded_w8(
        self, w: PreparedWeight, offsets: torch.Tensor, rows: int
    ) -> torch.Tensor | None:
        """Folded E4M3 ``W`` scratch for the forward GEMM, or None for in-kernel expansion."""
        num_experts, n, k_half = w.codes.shape
        if not hasattr(self._ext, _EXPAND_SYMBOL):
            return None
        if not self._use_preexpand((num_experts, n, 2 * k_half), rows, 1):
            return None
        return self._ext.sm90_moe_expand_w8(w.codes, w.res, offsets)

    def forward(self, batch: ExpertBatch, x_q: MXTensor, path: str = "two_launch") -> torch.Tensor:
        """Routed MLP forward for base weights. ``path``: 'two_launch' | 'fused'."""
        self._check_batch(batch, x_q)
        if path == "two_launch":
            return self.fc3(batch, self.fc1_swiglu_quant(batch, x_q))
        if path == "fused":
            w1, w2 = self.prepare(batch.w1), self.prepare(batch.w2)
            return self._ext.sm90_moe_fused_mlp_forward(
                x_q.codes,
                x_q.scales,
                w1.codes,
                w1.scales,
                w1.ref,
                w1.res,
                w2.codes,
                w2.scales,
                w2.ref,
                w2.res,
                batch.expert_offsets,
                batch.p_s.contiguous(),
            )
        raise ValueError(f"unknown path {path!r} (expected 'two_launch' or 'fused')")

    # ------------------------------------------------------------ backward
    def _check_backward_weight(self, w: MXTensor) -> None:
        key = (w.scales.data_ptr(), "bwd")
        if key in self._cache:
            return
        lo, hi = _BWD_SCALE_RANGE
        if bool(((w.scales < lo) | (w.scales > hi)).any()):
            raise NotImplementedError(
                f"{self.name} backward needs MXFP4 scale codes in [{lo}, {hi}] "
                "(exact BF16 expansion); fail-closed"
            )
        self._cache[key] = True

    def backward(
        self, batch: ExpertBatch, x_q: MXTensor, dy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(dx FP32 [M, H], dp_s FP32 [M])`` for ``dy`` ``[M, H]`` (cast to BF16)."""
        missing = [s for s in _BWD_SYMBOLS if not hasattr(self._ext, s)]
        if missing:
            raise NotImplementedError(f"rl_engine._C lacks {missing}; rebuild")
        self._check_batch(batch, x_q)
        self._check_backward_weight(batch.w1)
        self._check_backward_weight(batch.w2)
        z = self.fc1_z(batch, x_q)
        offsets = batch.expert_offsets
        w1, w2 = batch.w1, batch.w2
        w2_t = self._expanded(w2, offsets, batch.x.shape[0])
        dz, dp_s = self._ext.sm90_moe_bwd_dh_swiglu(
            dy.to(torch.bfloat16).contiguous(),
            w2.codes.contiguous(),
            w2.scales.contiguous(),
            offsets,
            z,
            batch.p_s.contiguous(),
            w2_t,
        )
        del w2_t
        w1_t = self._expanded(w1, offsets, batch.x.shape[0])
        dx = self._ext.sm90_moe_bwd_dx(
            dz, w1.codes.contiguous(), w1.scales.contiguous(), offsets, w1_t
        )
        return dx, dp_s

    def _expanded(self, w: MXTensor, offsets: torch.Tensor, rows: int) -> torch.Tensor | None:
        """BF16 ``W^T`` scratch for the backward GEMM, or None for in-kernel expansion."""
        if not self._use_preexpand(tuple(w.shape), rows, 2):
            return None
        return self._ext.sm90_moe_dequant_t(w.codes.contiguous(), w.scales.contiguous(), offsets)
