# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5-3 shared_grouped_lora_delta providers (#62).

Three backends for the LoRA delta ``Y = (X @ A.T) @ B.T * alpha``:

  LoRADeltaProvider        torch-native, the benchmark baseline. Fast, but
                           torch.matmul picks its tile/split-k strategy from M,
                           so it is NOT batch-invariant.
  LoRADeltaCudaProvider    composed from ``_C.det_gemm_*``.
  LoRADeltaTritonProvider  composed from ``TritonDetGemmOp``.

Both deterministic backends reuse the repo's shared fixed-K primitives rather
than adding a GEMM. They differ in how close they stay to the oracle:

  Triton   FP32 accumulator across the whole K loop, one round at the store.
           Byte-equal at the P5 fixture geometry (which is what check_p5.py
           measures); ~1e-6 relative at production width, because the oracle
           accumulates term by term while this path accumulates tile by tile.
  CUDA     det_gemm's mid-split tree rounds to BF16 at every 32-wide leaf, so
           it sits ~5e-3 from the oracle but the result also survives tensor
           parallelism -- a contiguous half-K shard is one child of the tree.

Neither declares ORACLE_PROFILE. Batch invariance is a separate property and
holds for both at every probed shape, including production width.

Each provider overrides only this one operator; the other eight stay on the
oracle via ReferenceProvider, so the full acceptance pipeline runs from day one.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.moe.provider import ReferenceProvider


class LoRADeltaProvider(ReferenceProvider):
    """Torch-native baseline. Not batch-invariant -- benchmark reference only."""

    name = "p5-3-lora-delta"
    numeric_profile = "torch-native-nondeterministic"

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": "torch-native",
            "numeric_profile": self.numeric_profile,
            "torch_version": torch.__version__,
        }

    def shared_grouped_lora_delta_fwd(
        self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Y = (X @ A.T) @ B.T * alpha, FP32 accumulate. Returns (y_fp32, u_bf16).

        Inputs are BF16 per the contract, so both operands are promoted before
        each matmul -- torch never promotes implicitly, and bf16 @ fp32 raises.
        Only ``u`` rounds back to BF16 (rounding point 1); the output is FP32.
        """
        u_fp32 = x.float() @ a.float().T
        u_bf16 = u_fp32.to(torch.bfloat16)
        y_fp32 = (u_bf16.float() @ b.float().T) * float(alpha)
        return y_fp32, u_bf16

    def shared_grouped_lora_delta_bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        alpha: float,
        u_bf16: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (dX, dA, dB) as FP32. No dW -- base weights are frozen.

        ``dY * alpha`` and ``dU`` round to BF16 between the GEMMs (contract D4);
        every accumulation stays FP32 and the association order is frozen as
        written, so ``dY @ (alpha * B)`` is not an admissible rewrite.
        """
        dys = (dy.float() * float(alpha)).to(torch.bfloat16)
        du_fp32 = dys.float() @ b.float()
        du_bf16 = du_fp32.to(torch.bfloat16)
        db = dys.float().T @ u_bf16.float()
        da = du_bf16.float().T @ x.float()
        dx = du_bf16.float() @ a.float()
        return dx, da, db


DET_GEMM_PROFILE = "det-gemm-midsplit-tree"


def det_gemm_lora_delta_fwd(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``Y = (X @ A.T) @ B.T * alpha`` on the CUDA det_gemm primitives.

    Single source of truth for this composition: both
    :class:`LoRADeltaCudaProvider` and
    :class:`~rl_engine.moe.backends.grouped_gemm.CudaP5GemmProvider` call it.
    """
    import rl_engine._C as _C

    u_bf16 = _C.det_gemm_fwd_rhs_transposed(x, a)
    y = _C.det_gemm_fwd_rhs_transposed(u_bf16, b).float() * float(alpha)
    return y, u_bf16


def det_gemm_lora_delta_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    u_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(dX, dA, dB)``. No ``dW`` -- the base weights are frozen."""
    import rl_engine._C as _C

    dys = (dy.float() * float(alpha)).to(torch.bfloat16)
    du_bf16 = _C.det_gemm_fwd(dys, b)  # dys @ b   -> [M, r]
    db = _C.det_gemm_db(dys, u_bf16).float()  # dys.T @ u -> [N, r]
    da = _C.det_gemm_db(du_bf16, x).float()  # du.T @ x  -> [r, K]
    dx = _C.det_gemm_fwd(du_bf16, a).float()  # du @ a    -> [M, K]
    return dx, da, db


class LoRADeltaCudaProvider(LoRADeltaProvider):
    """Batch-invariant via the CUDA det_gemm primitives."""

    name = "p5-3-lora-delta-detgemm"
    numeric_profile = DET_GEMM_PROFILE

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": "cuda-det_gemm-composed",
            "numeric_profile": self.numeric_profile,
            "reduction": "det_gemm mid-split k-tree, leaf=32, no split-K, no atomics",
            "torch_version": torch.__version__,
        }

    shared_grouped_lora_delta_fwd = staticmethod(det_gemm_lora_delta_fwd)
    shared_grouped_lora_delta_bwd = staticmethod(det_gemm_lora_delta_bwd)


class LoRADeltaTritonProvider(LoRADeltaProvider):
    """Batch-invariant via TritonDetGemmOp, and the closest match to the oracle.

    Invariance comes from three properties of that op, none of which involve
    avoiding tensor cores: BLOCK sizes are pinned (autotune would pick
    per-shape configs), there is no split-K, and TF32 is disabled. ``tl.dot``
    itself is deterministic once the tile shape stops depending on M.

    It is NOT byte-equal to the oracle in general, so it does not declare
    ORACLE_PROFILE. Both keep an FP32 accumulator, but the oracle adds one
    product at a time while this path adds one BLOCK_K=32 tile at a time. Those
    two parenthesisations agree only while nothing rounds: BF16 carries 8
    mantissa bits, so a BF16xBF16 product needs at most 16 and lands in FP32's
    24 exactly, and while the partial sums stay inside that headroom the
    accumulation order cannot change a single bit. As K grows the accumulator
    climbs into the remaining bits and begins dropping low ones, so the
    agreement degrades rather than breaking at a threshold -- measured over 12
    seeds at M=24 N=2048, byte-equal holds 12/12 up to K=512, then 11/12 at
    1024, 9/12 at 2048 and 7/12 at 4096.

    The P5 fixture geometry is K=128 for fc1 and K=64 for fc2, comfortably
    inside the headroom, which is why check_p5.py passes. The gap at production
    width is FP32-epsilon sized (~1e-6 relative, a handful of elements), never
    a correctness error, and batch invariance is unaffected because that
    compares this path against itself.
    """

    name = "p5-3-lora-delta-triton"
    # A profile is a promise about the reduction, and check_p5.py enforces
    # ORACLE_PROFILE with zero tolerance -- which this path cannot honour at
    # arbitrary K (see the class docstring). Declaring its own profile keeps the
    # claim honest; S0 has no mechanism yet for attaching an explicit tolerance
    # to a profile, so the bound is recorded in provenance() instead.
    numeric_profile = "triton-det-gemm-fp32-accum"

    def __init__(self) -> None:
        from rl_engine.kernels.ops.triton.matmul.det_gemm import TritonDetGemmOp
        self._op = TritonDetGemmOp()

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": "triton-det_gemm-composed",
            "numeric_profile": self.numeric_profile,
            "reduction": "pinned BLOCK 64/64/32, no split-K, no autotune, allow_tf32=False",
            # Byte-equal at the P5 fixture geometry; degrades with K as the FP32
            # accumulator runs out of headroom (see the class docstring).
            "oracle_agreement": "bitwise at K<=512; ~1e-6 relative by K=4096",
            "torch_version": torch.__version__,
        }

    def shared_grouped_lora_delta_fwd(self, x, a, b, alpha):
        op = self._op
        u_bf16 = op(x, a.t().contiguous())
        y = op.forward_accum_fp32(u_bf16, b.t().contiguous()) * float(alpha)
        return y, u_bf16

    def shared_grouped_lora_delta_bwd(self, dy, x, a, b, alpha, u_bf16):
        op = self._op
        dys = (dy.float() * float(alpha)).to(torch.bfloat16)

        du_bf16 = op(dys, b)
        db = op.forward_accum_fp32(dys.t().contiguous(), u_bf16)
        da = op.forward_accum_fp32(du_bf16.t().contiguous(), x)
        dx = op.forward_accum_fp32(du_bf16, a)
        return dx, da, db
