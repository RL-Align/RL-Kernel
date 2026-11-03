# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5-5 (#64) Shared Expert MLP providers (CUDA and Triton strict backends).

Both backends implement the frozen math ``fc1 -> one-round SwiGLU -> fc2``
under the ``oracle-fp32-serial-v1`` numeric profile and are byte-equal to the
FP32 oracle running on the same device. Only the two shared-expert methods are
overridden; every other operator stays on the oracle per the S0 start kit, so
the full acceptance command runs unchanged:

    python scripts/check_p5.py \
        --provider rl_engine.moe.backends.shared_expert:CudaSharedExpertProvider \
        --device cuda

Fail-closed: unsupported input (non-CUDA device, missing extension/triton,
schema violations) raises instead of falling back to another implementation.
The shared output is produced from ``SharedBatch`` alone -- no route weight,
no routed combine (that boundary belongs to P6).
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.moe.contract import ORACLE_PROFILE, SharedBatch
from rl_engine.moe.provider import ReferenceProvider


class _StrictSharedExpertProvider(ReferenceProvider):
    """Common composite: strict GEMMs + one-round SwiGLU, dX only (frozen base)."""

    name = "shared-expert-strict"
    numeric_profile = ORACLE_PROFILE

    # Backend hooks -------------------------------------------------------
    def _gemm(self, a: torch.Tensor, b: torch.Tensor, trans_b: bool) -> torch.Tensor:
        raise NotImplementedError

    def _swiglu_fwd(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _swiglu_bwd(self, dh: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # Provider surface ----------------------------------------------------
    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "operators": ["shared_expert_mlp_fwd", "shared_expert_mlp_bwd"],
            "geometry": ["one-row", "packed"],
            "devices": ["cuda"],
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": self.name,
            "numeric_profile": self.numeric_profile,
            "torch_version": torch.__version__,
            # Changing any of these changes the addition order (P5-5 s4).
            "split_k": 1,
            "reduction": "serial-ascending-k",
            "rounding": "mul-then-add, no FMA",
            "workspace": "none",
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

    def shared_expert_mlp_fwd(self, batch: SharedBatch) -> tuple[torch.Tensor, dict[str, Any]]:
        self._check_batch(batch)
        x = batch.x.contiguous()
        w_fc1 = batch.w_fc1.contiguous()
        w_fc2 = batch.w_fc2.contiguous()
        z = self._gemm(x, w_fc1, False)  # [T, 2F] FP32, kept for backward
        h_bf16 = self._swiglu_fwd(z)  # [T, F] BF16, the one round
        y = self._gemm(h_bf16, w_fc2, False).to(torch.bfloat16)  # [T, H]
        saved: dict[str, Any] = {"z32": z, "h_bf16": h_bf16}
        return y, saved

    def shared_expert_mlp_bwd(
        self, dy: torch.Tensor, batch: SharedBatch, saved: dict[str, Any]
    ) -> torch.Tensor:
        self._check_batch(batch)
        z = saved["z32"]
        dy_bf16 = dy.to(torch.bfloat16).contiguous()
        # dh = BF16(dY @ W2), dz = swiglu_bwd, dX = dz @ W1 (FP32 accumulator).
        dh = self._gemm(dy_bf16, batch.w_fc2.contiguous(), True).to(torch.bfloat16)
        dz = self._swiglu_bwd(dh, z)
        dx = self._gemm(dz, batch.w_fc1.contiguous(), True)
        return dx


class CudaSharedExpertProvider(_StrictSharedExpertProvider):
    """CUDA backend: csrc/cuda/moe/shared_expert_mlp.cu via rl_engine._C."""

    name = "shared-expert-cuda"

    def __init__(self) -> None:
        try:
            from rl_engine import _C
        except ImportError as exc:  # fail-closed: no oracle fallback
            raise NotImplementedError(
                "rl_engine._C is not built; install with RL_KERNEL_REQUIRE_EXT=1"
            ) from exc
        for symbol in ("p5_strict_gemm", "p5_swiglu_shared_forward", "p5_swiglu_shared_backward"):
            if not hasattr(_C, symbol):
                raise NotImplementedError(f"rl_engine._C lacks {symbol}; rebuild the extension")
        self._ext = _C

    def _gemm(self, a: torch.Tensor, b: torch.Tensor, trans_b: bool) -> torch.Tensor:
        return self._ext.p5_strict_gemm(a, b, trans_b)

    def _swiglu_fwd(self, z: torch.Tensor) -> torch.Tensor:
        return self._ext.p5_swiglu_shared_forward(z)

    def _swiglu_bwd(self, dh: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self._ext.p5_swiglu_shared_backward(dh, z)


class TritonSharedExpertProvider(_StrictSharedExpertProvider):
    """Triton backend: rl_engine/kernels/ops/triton/moe/shared_expert.py."""

    name = "shared-expert-triton"

    def __init__(self) -> None:
        from rl_engine.kernels.ops.triton.moe import shared_expert as tk

        if not tk.TRITON_AVAILABLE:
            raise NotImplementedError("triton is not installed (fail-closed, no fallback)")
        self._tk = tk

    def _gemm(self, a: torch.Tensor, b: torch.Tensor, trans_b: bool) -> torch.Tensor:
        return self._tk.strict_gemm(a, b, trans_b)

    def _swiglu_fwd(self, z: torch.Tensor) -> torch.Tensor:
        return self._tk.swiglu_shared_fwd(z)

    def _swiglu_bwd(self, dh: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self._tk.swiglu_shared_bwd(dh, z)


class CudaDetSharedExpertProvider(CudaSharedExpertProvider):
    """Performance CUDA backend: det_gemm (csrc/cuda/gemm/det_gemm_kernel.cu).

    Deterministic and batch-invariant (fixed K order, no split-K; TMA+mma.sync
    on SM90+, scalar K-tree fallback elsewhere). Round positions match the
    P5-5 contract (fc1 out and dX stay FP32; y/dh round once to BF16), but the
    in-GEMM reduction order differs from the oracle, so outputs are close, not
    byte-equal. SwiGLU stays on the strict CUDA core.
    """

    name = "shared-expert-cuda-det"
    numeric_profile = "p5-det-gemm-v1"
    # det_gemm rounds each BK=32 partial to BF16 and merges the K dimension
    # with a BF16 mid-split tree (its TP-equivalence design), so its deviation
    # from the FP32-serial oracle is BF16-tree-sized, not FP32-sized.
    oracle_tolerance = {"rtol": 1e-1, "atol": 6e-2}

    def _gemm(self, a: torch.Tensor, b: torch.Tensor, trans_b: bool) -> torch.Tensor:
        # det_gemm rounds every BK=32 leaf to BF16 and merges the K dimension
        # with a BF16 tree, so an "FP32 output" mode could only widen that BF16
        # result -- verified bit-identical on both the SM90 and scalar paths.
        # Widening here instead keeps the MoE-specific entry points out of
        # csrc/cuda/gemm/.
        if trans_b:  # b is the logical [K, N] operand
            return self._ext.det_gemm_fwd(a, b).float()
        return self._ext.det_gemm_fwd_rhs_transposed(a, b).float()

    def provenance(self) -> dict[str, Any]:
        info = super().provenance()
        info.update(
            {
                "split_k": 1,
                "reduction": "fixed-k-tile-tree (det_gemm)",
                "rounding": "FP32 accumulate, FMA/MMA inside tiles",
                "sm90_tensor_core": bool(getattr(self._ext, "det_gemm_sm90_compiled")()),
            }
        )
        return info


class TritonDetSharedExpertProvider(TritonSharedExpertProvider):
    """Performance Triton backend: tl.dot tiles with fixed geometry.

    Same guarantees and caveats as :class:`CudaDetSharedExpertProvider`, with
    profile ``p5-triton-dot-v1`` (tile reduction order differs per backend).
    """

    name = "shared-expert-triton-det"
    numeric_profile = "p5-triton-dot-v1"
    # Full-FP32 accumulators (only the contract's BF16 rounds), so deviation
    # from the oracle is reduction-order noise only.
    oracle_tolerance = {"rtol": 2e-2, "atol": 2e-2}

    def _gemm(self, a: torch.Tensor, b: torch.Tensor, trans_b: bool) -> torch.Tensor:
        return self._tk.det_dot_gemm(a, b, trans_b)

    def provenance(self) -> dict[str, Any]:
        info = super().provenance()
        info.update(
            {
                "split_k": 1,
                "reduction": "tl.dot 64x64x32 tiles, ascending-k",
                "rounding": "FP32 accumulate, MMA inside tiles",
            }
        )
        return info


class CudaFusedSharedExpertProvider(_StrictSharedExpertProvider):
    """Forward-only CUDA backend with fc1 and the SwiGLU fused into one kernel.

    ``csrc/cuda/moe/fused_shared_expert_mlp.cu`` computes ``h`` directly from
    ``x`` and ``w_fc1``, so the FP32 ``z`` [T, 2F] intermediate never reaches
    global memory. fc2 stays on ``det_gemm``.

    Bit-identical to :class:`CudaDetSharedExpertProvider` -- the fused kernel
    reuses det_gemm's mid-split K-tree with the same 32-wide FP32 leaf, and its
    epilogue reproduces the strict SwiGLU core's instruction sequence. Fusing
    is therefore a pure performance change; the tests assert the equality
    rather than a tolerance.

    Forward only. The backward needs ``z``, which this kernel deliberately does
    not write; use :class:`CudaDetSharedExpertProvider` for training, or
    recompute ``z`` with a plain det_gemm call first.
    """

    name = "shared-expert-cuda-fused"
    numeric_profile = "p5-det-gemm-v1"
    oracle_tolerance = {"rtol": 1e-1, "atol": 6e-2}

    def __init__(self) -> None:
        try:
            from rl_engine import _C
        except ImportError as exc:  # fail-closed: no oracle fallback
            raise NotImplementedError(
                "rl_engine._C is not built; install with RL_KERNEL_REQUIRE_EXT=1"
            ) from exc
        for symbol in ("fused_shared_expert_fc1_swiglu", "det_gemm_fwd_rhs_transposed"):
            if not hasattr(_C, symbol):
                raise NotImplementedError(f"rl_engine._C lacks {symbol}; rebuild the extension")
        self._ext = _C

    def _gemm(self, a: torch.Tensor, b: torch.Tensor, trans_b: bool) -> torch.Tensor:
        # Only fc2 reaches this; see CudaDetSharedExpertProvider for why the
        # BF16 output is widened here rather than asked for in FP32.
        if trans_b:  # b is the logical [K, N] operand
            return self._ext.det_gemm_fwd(a, b).float()
        return self._ext.det_gemm_fwd_rhs_transposed(a, b).float()

    def shared_expert_mlp_fwd(self, batch: SharedBatch) -> tuple[torch.Tensor, dict[str, Any]]:
        self._check_batch(batch)
        h_bf16 = self._ext.fused_shared_expert_fc1_swiglu(
            batch.x.contiguous(), batch.w_fc1.contiguous()
        )
        y = self._gemm(h_bf16, batch.w_fc2.contiguous(), False).to(torch.bfloat16)
        # No "z32": the fused kernel never materializes it, which is the point.
        return y, {"h_bf16": h_bf16}

    def shared_expert_mlp_bwd(
        self, dy: torch.Tensor, batch: SharedBatch, saved: dict[str, Any]
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.name} is forward-only: the fused kernel does not write z, which the "
            "backward needs. Use CudaDetSharedExpertProvider for training (bit-identical)."
        )

    def provenance(self) -> dict[str, Any]:
        info = super().provenance()
        info.update(
            {
                "split_k": 1,
                "reduction": "fixed-k-tile-tree (det_gemm), fc1 fused with SwiGLU",
                "fused_operators": ["fc1", "swiglu"],
                "backward": False,
                "sm90_tensor_core": bool(getattr(self._ext, "det_gemm_sm90_compiled")()),
            }
        )
        return info


class TritonFusedSharedExpertProvider(_StrictSharedExpertProvider):
    """Forward-only Triton backend: fused fc1+SwiGLU, then a TP-invariant fc2.

    The portable counterpart of :class:`CudaFusedSharedExpertProvider`, built
    from two kernels in ``rl_engine/kernels/ops/triton/moe/fused_mlp.py``:

    * ``fused_shared_fc1_swiglu`` computes ``h`` straight from ``x`` and
      ``w_fc1`` with a ``tl.dot`` main loop and the SwiGLU in the epilogue, so
      the FP32 ``z`` [T, 2F] never reaches global memory. fc1 is
      column-parallel under TP, so it is TP-invariant with no extra work.
    * ``shared_fc3`` reduces fc2's K over a fixed 8-leaf BF16 tree, the same
      tree the deterministic all-reduce uses across ranks, so TP=1, 2, 4 and 8
      produce identical bytes.

    Not byte-equal to the CUDA fused kernel, by design: no Triton exponential
    reproduces nvcc's ``expf`` (measured at T*F = 262144: ``tl.sigmoid``
    differs in 6 elements, libdevice ``exp`` in 2), and the fc2 tree has 8
    F/8-wide FP32 leaves rather than det_gemm's 32-wide ones. Hence its own
    profile: no other backend is allowed to claim byte-equality with it.

    Forward only, for the same reason as the CUDA fused provider: the backward
    needs ``z``, which the kernel deliberately does not write.
    """

    name = "shared-expert-triton-fused"
    numeric_profile = "p5-triton-fused-tp8-v1"
    oracle_tolerance = {"rtol": 1e-1, "atol": 6e-2}

    def __init__(self, tp_size: int = 1) -> None:
        from rl_engine.kernels.ops.triton.moe import fused_mlp as tf
        from rl_engine.kernels.ops.triton.moe import shared_expert as tk

        if not tk.TRITON_AVAILABLE:
            raise NotImplementedError("triton is not installed (fail-closed, no fallback)")
        self._tf = tf
        # With tp_size > 1 the batch holds this rank's weight shards and the
        # returned y is the rank's BF16 partial, to be summed by the
        # deterministic all-reduce.
        self._tp_size = tp_size

    def shared_expert_mlp_fwd(self, batch: SharedBatch) -> tuple[torch.Tensor, dict[str, Any]]:
        self._check_batch(batch)
        h_bf16 = self._tf.fused_shared_fc1_swiglu(
            batch.x.contiguous(), batch.w_fc1.contiguous()
        )
        y = self._tf.shared_fc3(h_bf16, batch.w_fc2.contiguous(), self._tp_size)
        # No "z32": not materializing it is the point of the fusion.
        return y, {"h_bf16": h_bf16}

    def shared_expert_mlp_bwd(
        self, dy: torch.Tensor, batch: SharedBatch, saved: dict[str, Any]
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.name} is forward-only: the fused kernel does not write z, which the "
            "backward needs. Use TritonDetSharedExpertProvider for training."
        )

    def provenance(self) -> dict[str, Any]:
        info = super().provenance()
        dev = torch.device("cuda", torch.cuda.current_device())
        info.update(
            {
                "split_k": 1,
                "reduction": "fc1: tl.dot ascending-k, flat FP32; "
                "fc2: 8-leaf BF16 tree over F/8-wide FP32 leaves",
                "rounding": "FP32 accumulate; BF16 round on h, on each fc2 leaf and tree node",
                "fused_operators": ["fc1", "swiglu"],
                "tiles": {k: self._tf.tiles(k, dev) for k in ("shared_fc1", "shared_fc3")},
                "tp_equivalent": True,
                "tp_size": self._tp_size,
                "backward": False,
            }
        )
        return info
