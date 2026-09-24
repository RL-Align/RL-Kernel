# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P1-2 ``fp32_gemm_rms``: deterministic FP32 controller GEMM + RMS scale.

CUDA and Triton backends for the oracle pair
:func:`rl_engine.mhc.oracle.fp32_gemm_rms_fwd` /
:func:`rl_engine.mhc.oracle.fp32_gemm_rms_bwd`, plus the P1-D6 fixed-K GEMM
core, an autograd entry point, and the fixed-K reference / bit-equality
harness deliverables.

Contract (issue #2 / DSV4 P1-2 spec):

- ``P[t, n] = sum_k X[t, k] * W[n, k]`` with one FP32 accumulator per output,
  k ascending, mul-then-add rounding, no Split-K / Stream-K / atomics / TF32.
- ``s = sum_k X[t, k]^2`` (same fold); ``norm = sqrt(s)``;
  ``q = norm / sqrt(K)``; ``r = 1 / (q + eps)``. This is the *controller* RMS
  ``1/(sqrt(mean(X^2)) + eps)`` -- deliberately **not** the
  ``rsqrt(mean + eps)`` of ``rmsnorm_residual``.
- Backward: ``dX_gemm = dP @ W`` (n ascending), ``dW = dP.T @ X`` (t
  ascending), ``dX_rms = dr * ((-(r^2) * X) / (K * q))``,
  ``dX = dX_gemm + dX_rms`` in that fixed order.
- ``P`` and ``r`` stay FP32; both backends reproduce the oracle bytes and
  therefore keep ``numeric_profile = ORACLE_PROFILE``. Unsupported inputs
  fail closed -- there is no silent fallback to the oracle.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from rl_engine.mhc import oracle
from rl_engine.mhc.contract import ORACLE_PROFILE, tensor_bytes
from rl_engine.mhc.provider import ReferenceProvider
from rl_engine.mhc.reduction import fixed_dot

_CUDA_EXT: Any = None
_CUDA_EXT_ERROR: str | None = None


def _cuda_ext() -> Any:
    """Import the standalone ``rl_engine._C_mhc`` extension, fail-closed."""
    global _CUDA_EXT, _CUDA_EXT_ERROR
    if _CUDA_EXT is None and _CUDA_EXT_ERROR is None:
        try:
            from rl_engine import _C_mhc  # noqa: PLC0415

            _CUDA_EXT = _C_mhc
        except ImportError as exc:  # pragma: no cover - build-environment path
            _CUDA_EXT_ERROR = str(exc)
    if _CUDA_EXT is None:
        raise RuntimeError(
            "rl_engine._C_mhc is not built; run `pip install -e .` (or "
            "`python setup.py build_ext --inplace`) with CUDA available. "
            f"Import error: {_CUDA_EXT_ERROR} (fail-closed, no oracle fallback)"
        )
    return _CUDA_EXT


def _validate(x_flat: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    if x_flat.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"x_flat/weight must be 2-D, got {x_flat.dim()}-D / {weight.dim()}-D")
    if x_flat.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError(f"x_flat/weight must be FP32, got {x_flat.dtype} / {weight.dtype}")
    if weight.shape[1] != x_flat.shape[1]:
        raise ValueError(f"weight K {weight.shape[1]} != x_flat K {x_flat.shape[1]}")
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError("eps must be positive and finite")
    if not x_flat.is_cuda or not weight.is_cuda:
        # NotImplementedError so check_p1 reports a failed case instead of
        # crashing; it stays a RuntimeError subclass for callers.
        raise NotImplementedError(
            "the CUDA/Triton fp32_gemm_rms backends require CUDA tensors "
            "(fail-closed; use the reference provider on CPU)"
        )


def _pack_saved(
    s: torch.Tensor, norm: torch.Tensor, q: torch.Tensor, r: torch.Tensor, k: int
) -> dict[str, Any]:
    # Same saved schema as the oracle so providers are interchangeable in
    # mhc_pre_bwd and check_p1.
    return {"s": s, "norm": norm, "q": q, "r": r, "k": k}


class CudaGemmRMSProvider(ReferenceProvider):
    """P1-2 CUDA backend; every other operator stays on the oracle."""

    name = "cuda-gemm-rms"
    numeric_profile = ORACLE_PROFILE  # byte-equal to the oracle by contract

    def capabilities(self) -> dict[str, Any]:
        caps = super().capabilities()
        caps["backend"] = "cuda-fp32-gemm-rms"
        caps["devices"] = ["cuda"]
        return caps

    def provenance(self) -> dict[str, Any]:
        prov = super().provenance()
        prov["requested_backend"] = self.name
        prov["actual_backend"] = self.name
        prov["p1_2_kernel"] = "rl_engine._C_mhc"
        return prov

    _fwd_fn = None

    def fp32_gemm_rms_fwd(
        self, x_flat: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        _validate(x_flat, weight, eps)
        fwd = self._fwd_fn
        if fwd is None:
            fwd = type(self)._fwd_fn = _cuda_ext().fp32_gemm_rms_forward
        if not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        p, r, s, norm, q = fwd(x_flat, weight, eps)
        return p, r, _pack_saved(s, norm, q, r, int(x_flat.shape[1]))

    def fp32_gemm_rms_bwd(
        self,
        dp: torch.Tensor,
        dr: torch.Tensor,
        x_flat: torch.Tensor,
        weight: torch.Tensor,
        saved: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dx, dw = _cuda_ext().fp32_gemm_rms_backward(
            dp.to(torch.float32).contiguous(),
            dr.to(torch.float32).contiguous(),
            x_flat.contiguous(),
            weight.contiguous(),
            saved["q"].contiguous(),
            saved["r"].contiguous(),
        )
        return dx, dw

    def fixed_k_gemm_fwd(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise NotImplementedError("CUDA fixed_k_gemm requires CUDA tensors (fail-closed)")
        return _cuda_ext().fixed_k_gemm_forward(
            x.to(torch.float32).contiguous(), w.to(torch.float32).contiguous()
        )

    def fixed_k_gemm_bwd(
        self, dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not x.is_cuda:
            raise NotImplementedError("CUDA fixed_k_gemm requires CUDA tensors (fail-closed)")
        dx, dw = _cuda_ext().fixed_k_gemm_backward(
            dy.to(torch.float32).contiguous(),
            x.to(torch.float32).contiguous(),
            w.to(torch.float32).contiguous(),
        )
        return dx, dw


class TritonGemmRMSProvider(ReferenceProvider):
    """P1-2 Triton backend; every other operator stays on the oracle."""

    name = "triton-gemm-rms"
    numeric_profile = ORACLE_PROFILE

    def capabilities(self) -> dict[str, Any]:
        caps = super().capabilities()
        caps["backend"] = "triton-fp32-gemm-rms"
        caps["devices"] = ["cuda"]
        return caps

    def provenance(self) -> dict[str, Any]:
        prov = super().provenance()
        prov["requested_backend"] = self.name
        prov["actual_backend"] = self.name
        prov["p1_2_kernel"] = "rl_engine.mhc.fp32_gemm_rms_triton"
        return prov

    def fp32_gemm_rms_fwd(
        self, x_flat: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        from rl_engine.mhc import fp32_gemm_rms_triton as tri  # noqa: PLC0415

        _validate(x_flat, weight, eps)
        p, r, s, norm, q = tri.triton_gemm_rms_forward(x_flat, weight, float(eps))
        return p, r, _pack_saved(s, norm, q, r, int(x_flat.shape[1]))

    def fp32_gemm_rms_bwd(
        self,
        dp: torch.Tensor,
        dr: torch.Tensor,
        x_flat: torch.Tensor,
        weight: torch.Tensor,
        saved: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from rl_engine.mhc import fp32_gemm_rms_triton as tri  # noqa: PLC0415

        return tri.triton_gemm_rms_backward(
            dp.to(torch.float32),
            dr.to(torch.float32),
            x_flat,
            weight,
            saved["q"],
            saved["r"],
        )

    def fixed_k_gemm_fwd(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        from rl_engine.mhc import fp32_gemm_rms_triton as tri  # noqa: PLC0415

        return tri.triton_fixed_k_gemm_forward(x.to(torch.float32), w.to(torch.float32))

    def fixed_k_gemm_bwd(
        self, dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from rl_engine.mhc import fp32_gemm_rms_triton as tri  # noqa: PLC0415

        return tri.triton_fixed_k_gemm_backward(
            dy.to(torch.float32), x.to(torch.float32), w.to(torch.float32)
        )


_PROVIDERS = {
    "reference": ReferenceProvider,
    "cuda": CudaGemmRMSProvider,
    "triton": TritonGemmRMSProvider,
}


class _GemmRMSFunction(torch.autograd.Function):
    """Autograd entry over a provider's exact fwd/bwd pair (no torch graph)."""

    @staticmethod
    def forward(ctx, x_flat, weight, eps, provider):
        p, r, saved = provider.fp32_gemm_rms_fwd(x_flat, weight, eps)
        ctx.provider = provider
        ctx.save_for_backward(x_flat, weight, saved["q"], saved["r"])
        ctx.k = saved["k"]
        return p, r

    @staticmethod
    def backward(ctx, dp, dr):
        x_flat, weight, q, r = ctx.saved_tensors
        if dp is None:
            dp = torch.zeros(
                (x_flat.shape[0], weight.shape[0]),
                dtype=torch.float32,
                device=x_flat.device,
            )
        if dr is None:
            dr = torch.zeros((x_flat.shape[0],), dtype=torch.float32, device=x_flat.device)
        saved = {"q": q, "r": r, "k": ctx.k}
        dx, dw = ctx.provider.fp32_gemm_rms_bwd(dp, dr, x_flat, weight, saved)
        return dx, dw, None, None


_PROVIDER_CACHE: dict[str, Any] = {}


def fp32_gemm_rms(
    x_flat: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    backend: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable ``(P, r)`` through the chosen deterministic backend."""
    provider = _PROVIDER_CACHE.get(backend)
    if provider is None:
        if backend not in _PROVIDERS:
            raise ValueError(f"unknown backend {backend!r}; want {sorted(_PROVIDERS)}")
        provider = _PROVIDER_CACHE[backend] = _PROVIDERS[backend]()
    if not torch.is_grad_enabled() or not (x_flat.requires_grad or weight.requires_grad):
        # Inference fast path: identical kernels and bytes, without the
        # autograd Function machinery.
        p, r, _ = provider.fp32_gemm_rms_fwd(x_flat, weight, eps)
        return p, r
    return _GemmRMSFunction.apply(x_flat, weight, eps, provider)


# ---------------------------------------------------------------------------
# P1-2 auxiliary deliverables (task spec): fixed-K reference + byte harness
# ---------------------------------------------------------------------------


def fixed_k_gemm_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Golden fixed-K GEMM: single FP32 accumulator, k ascending, one output
    cast performed by the caller. Runs on any device; defines the bytes that
    every fast path (DeepGEMM, TE, ...) must reproduce before swapping in."""
    return fixed_dot(x, w)


def fixed_k_gemm_bitequal_harness(
    candidate: Any, x: torch.Tensor, w: torch.Tensor
) -> dict[str, Any]:
    """Run ``candidate(x, w)`` against :func:`fixed_k_gemm_reference` and
    report raw-byte equality. Downstream consumers gate fast-path swaps on
    ``result["bitwise_equal"]`` (the swap condition is byte-equal *and*
    faster)."""
    want = fixed_k_gemm_reference(x, w)
    got = candidate(x, w)
    if got.shape != want.shape:
        raise ValueError(f"candidate shape {tuple(got.shape)} != {tuple(want.shape)}")
    if got.dtype != want.dtype:
        raise TypeError(f"candidate dtype {got.dtype} != {want.dtype}")
    equal = tensor_bytes(got.cpu()) == tensor_bytes(want.cpu())
    diff = (got.to(torch.float64) - want.to(torch.float64)).abs()
    return {
        "bitwise_equal": bool(equal),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mismatch_count": int((got != want).sum().item()),
        "numel": int(want.numel()),
    }


def fp32_gemm_rms_reference(
    x_flat: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """The pinned oracle pair, re-exported for tests and the gtest gold path."""
    return oracle.fp32_gemm_rms_fwd(x_flat, weight, eps)


__all__ = [
    "CudaGemmRMSProvider",
    "TritonGemmRMSProvider",
    "fixed_k_gemm_bitequal_harness",
    "fixed_k_gemm_reference",
    "fp32_gemm_rms",
    "fp32_gemm_rms_reference",
]
