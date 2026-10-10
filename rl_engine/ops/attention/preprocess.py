# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Bitwise-bound QK-Norm and RoPE handoff for WS2 Attention.

Transformer Engine RMSNorm is the first-choice QK-Norm implementation on CUDA
and ROCm. It is admitted only after a same-input bitwise probe against the
platform RL-Kernel path. RoPE remains on the shared RL-Kernel implementation so
training and paged-cache rollout expose the same post-RoPE boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch
from torch import Tensor
from torch.autograd import Function

QK_RMSNORM_BACKEND_ID = "rlkernel.cuda.rmsnorm"
ROPE_BACKEND_ID = "rlkernel.cuda.rope_sm90"
ROCM_QK_RMSNORM_BACKEND_ID = "rlkernel.rocm.triton_rmsnorm"
ROCM_ROPE_BACKEND_ID = "rlkernel.rocm.deterministic_rope"
TE_CUDA_QK_RMSNORM_BACKEND_ID = "transformer_engine.cuda.rmsnorm"
TE_ROCM_QK_RMSNORM_BACKEND_ID = "transformer_engine.rocm.rmsnorm"
NATIVE_QK_RMSNORM_BACKEND_ID = TE_CUDA_QK_RMSNORM_BACKEND_ID
NATIVE_ROPE_BACKEND_ID = "native.rope"
PREPROCESS_POLICY_ID = "ws2.attention.preprocess.v3"
MANDATED_ATTENTION_PREPROCESS_BACKENDS: Mapping[str, str] = MappingProxyType(
    {
        "qk_rmsnorm": QK_RMSNORM_BACKEND_ID,
        "rope": ROPE_BACKEND_ID,
    }
)
ROCM_ATTENTION_PREPROCESS_BACKENDS: Mapping[str, str] = MappingProxyType(
    {
        "qk_rmsnorm": ROCM_QK_RMSNORM_BACKEND_ID,
        "rope": ROCM_ROPE_BACKEND_ID,
    }
)
ALLOWED_ATTENTION_PREPROCESS_BACKENDS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "qk_rmsnorm": frozenset(
            {
                QK_RMSNORM_BACKEND_ID,
                ROCM_QK_RMSNORM_BACKEND_ID,
                TE_CUDA_QK_RMSNORM_BACKEND_ID,
                TE_ROCM_QK_RMSNORM_BACKEND_ID,
            }
        ),
        "rope": frozenset({ROPE_BACKEND_ID, ROCM_ROPE_BACKEND_ID, NATIVE_ROPE_BACKEND_ID}),
    }
)


class TransformerEngineRMSNormUnavailable(RuntimeError):
    """Raised when the exact TE RMSNorm functional contract is unavailable."""


@lru_cache(maxsize=1)
def _load_transformer_engine_rmsnorm():
    try:
        import transformer_engine
        import transformer_engine.pytorch  # noqa: F401 - loads the platform extension
        from transformer_engine.pytorch.constants import TE_DType
        from transformer_engine.pytorch.cpp_extensions import rmsnorm_bwd, rmsnorm_fwd
    except (ImportError, OSError, RuntimeError) as exc:
        raise TransformerEngineRMSNormUnavailable(
            "Transformer Engine RMSNorm forward/backward is unavailable"
        ) from exc
    return rmsnorm_fwd, rmsnorm_bwd, TE_DType, getattr(transformer_engine, "__version__", "unknown")


class _TransformerEngineRMSNormFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float) -> Tensor:
        rmsnorm_fwd, rmsnorm_bwd, te_dtype, _version = _load_transformer_engine_rmsnorm()
        original_shape = x.shape
        hidden = original_shape[-1]
        x_2d = x.contiguous().view(-1, hidden)
        weight_1d = weight.contiguous().view(hidden)
        try:
            out, _unused, rsigma = rmsnorm_fwd(
                x_2d,
                weight_1d,
                float(eps),
                None,
                None,
                te_dtype[x.dtype],
                0,
                False,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise TransformerEngineRMSNormUnavailable(
                "installed Transformer Engine has an incompatible RMSNorm API"
            ) from exc
        ctx.save_for_backward(x_2d, rsigma, weight_1d)
        ctx.rmsnorm_bwd = rmsnorm_bwd
        ctx.original_shape = original_shape
        return out.view(original_shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x_2d, rsigma, weight = ctx.saved_tensors
        dy = grad_output.contiguous().view_as(x_2d)
        try:
            dx, dw = ctx.rmsnorm_bwd(dy, x_2d, rsigma, weight, 0, False)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TransformerEngineRMSNormUnavailable(
                "installed Transformer Engine has an incompatible RMSNorm backward API"
            ) from exc
        return dx.view(ctx.original_shape), dw.view_as(weight), None


class TransformerEngineRMSNormOp:
    """External-weight TE RMSNorm used by both training and rollout adapters."""

    def __init__(self) -> None:
        _fwd, _bwd, _dtype, version = _load_transformer_engine_rmsnorm()
        platform = "rocm" if torch.version.hip is not None else "cuda"
        self.backend_id = f"transformer_engine.{platform}.rmsnorm"
        self.package_version = version

    def __call__(self, x: Tensor, weight: Tensor, *, eps: float = 1.0e-6) -> Tensor:
        if x.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != x.dtype:
            raise TypeError(
                "Transformer Engine RMSNorm requires matching FP16/BF16 input and weight"
            )
        if not x.is_cuda or not weight.is_cuda or x.device != weight.device:
            raise ValueError("Transformer Engine RMSNorm requires input and weight on one GPU")
        if weight.shape != (x.shape[-1],):
            raise ValueError("RMSNorm weight must match the input hidden dimension")
        return _TransformerEngineRMSNormFunction.apply(x, weight, float(eps))


@dataclass(frozen=True)
class AttentionPreprocessResult:
    """Post-QK-Norm, post-RoPE tensors plus executed backend evidence.

    ``probe_id`` identifies the probe configuration, not tensor contents, and
    must never be used as an admission-result cache key.
    """

    q: Tensor
    k: Tensor
    backend_ids: Mapping[str, str]
    fallback: bool
    device_capability: tuple[int, int]
    fallback_reason: str | None = None
    probe_id: str = ""
    policy_id: str = PREPROCESS_POLICY_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_ids", MappingProxyType(dict(self.backend_ids)))

    def evidence(self) -> dict[str, Any]:
        return {
            "backends": dict(self.backend_ids),
            "fallback": self.fallback,
            "device_capability": list(self.device_capability),
        }

    def readback_fields(self) -> dict[str, Any]:
        """Keyword fields consumed by ``AttentionRuntimeReadback``."""

        return {
            "preprocess_backends": dict(self.backend_ids),
            "preprocess_fallback": self.fallback,
            "preprocess_fallback_reason": self.fallback_reason,
            "preprocess_probe_id": self.probe_id,
            "preprocess_policy_id": self.policy_id,
        }


class H100AttentionPreprocessor:
    """Reuse TE QK-Norm with the RL-Kernel SM90 RoPE boundary.

    ``native_qk_norm`` and ``native_rope`` are framework-owned callables.  They
    are intentionally injected instead of importing TE/vLLM here, so the same
    policy can be used by both runtimes.  The deterministic callables default to
    RL-Kernel's CUDA operators and are always run to establish the probe oracle.
    """

    def __init__(
        self,
        device: torch.device | str | int | None = None,
        *,
        native_qk_norm: Callable[..., Tensor] | None = None,
        native_rope: Callable[..., Tensor] | None = None,
        native_qk_norm_backend_id: str = NATIVE_QK_RMSNORM_BACKEND_ID,
        native_rope_backend_id: str = NATIVE_ROPE_BACKEND_ID,
        reuse_transformer_engine_qk_norm: bool = True,
        require_transformer_engine_qk_norm: bool = True,
        policy_id: str = PREPROCESS_POLICY_ID,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("H100AttentionPreprocessor requires an available CUDA runtime")

        current_device = torch.cuda.current_device()
        self.device = torch.device("cuda", current_device)
        if device is not None:
            self.device = (
                torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
            )
        if self.device.type != "cuda":
            raise RuntimeError(f"H100AttentionPreprocessor requires CUDA, got {self.device}")
        if self.device.index is None:
            self.device = torch.device("cuda", current_device)

        capability = torch.cuda.get_device_capability(self.device)
        self.device_capability: tuple[int, int] = (int(capability[0]), int(capability[1]))
        if self.device_capability[0] != 9:
            raise RuntimeError(
                "H100AttentionPreprocessor requires Hopper SM90; "
                f"got sm_{self.device_capability[0]}{self.device_capability[1]}"
            )

        # Import only after the hardware gate so CPU tools can inspect the module.
        from rl_engine.backends.cuda.norm.rmsnorm import RMSNormCudaOp
        from rl_engine.backends.cuda.rope.rope import RoPESM90Op

        self.rmsnorm: Callable[..., Tensor] = RMSNormCudaOp()
        self.rope: Callable[..., Tensor] = RoPESM90Op()
        self.deterministic_backend_ids = MANDATED_ATTENTION_PREPROCESS_BACKENDS
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("policy_id must be a non-empty string")
        if native_qk_norm is None and reuse_transformer_engine_qk_norm:
            try:
                native_qk_norm = TransformerEngineRMSNormOp()
                native_qk_norm_backend_id = native_qk_norm.backend_id
            except TransformerEngineRMSNormUnavailable:
                if require_transformer_engine_qk_norm:
                    raise
        self.native_qk_norm = native_qk_norm
        self.native_rope = native_rope
        self.require_native_qk_norm = bool(
            require_transformer_engine_qk_norm and reuse_transformer_engine_qk_norm
        )
        self.native_qk_norm_backend_id = native_qk_norm_backend_id
        self.native_rope_backend_id = native_rope_backend_id
        self.policy_id = policy_id

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        q_weight: Tensor,
        k_weight: Tensor,
        positions: Tensor,
        *,
        eps: float = 1.0e-6,
        theta: float = 1_000_000.0,
    ) -> AttentionPreprocessResult:
        return self.forward(
            q,
            k,
            q_weight,
            k_weight,
            positions,
            eps=eps,
            theta=theta,
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        q_weight: Tensor,
        k_weight: Tensor,
        positions: Tensor,
        *,
        eps: float = 1.0e-6,
        theta: float = 1_000_000.0,
    ) -> AttentionPreprocessResult:
        _validate_inputs(q, k, q_weight, k_weight, positions, self.device)
        q_norm_det = self.rmsnorm(q, q_weight, eps=eps)
        k_norm_det = self.rmsnorm(k, k_weight, eps=eps)
        q_det = _apply_deterministic_rope(self.rope, q_norm_det, positions, theta)
        k_det = _apply_deterministic_rope(self.rope, k_norm_det, positions, theta)
        probe_id = _probe_configuration_id(q, k, q_weight, k_weight, positions, eps, theta)

        native_qk_norm = self.native_qk_norm
        native_rope = self.native_rope
        if native_qk_norm is not None:
            try:
                q_norm_native = native_qk_norm(q, q_weight, eps=eps)
                k_norm_native = native_qk_norm(k, k_weight, eps=eps)
                norm_matches = torch.equal(q_norm_native, q_norm_det) and torch.equal(
                    k_norm_native, k_norm_det
                )
                if norm_matches and native_rope is None:
                    return AttentionPreprocessResult(
                        q=_apply_deterministic_rope(self.rope, q_norm_native, positions, theta),
                        k=_apply_deterministic_rope(self.rope, k_norm_native, positions, theta),
                        backend_ids=MappingProxyType(
                            {
                                "qk_rmsnorm": self.native_qk_norm_backend_id,
                                "rope": self.deterministic_backend_ids["rope"],
                            }
                        ),
                        fallback=False,
                        device_capability=self.device_capability,
                        probe_id=probe_id,
                        policy_id=self.policy_id,
                    )
                if norm_matches and native_rope is not None:
                    q_native = native_rope(q_norm_native, positions, theta=theta)
                    k_native = native_rope(k_norm_native, positions, theta=theta)
                    if torch.equal(q_native, q_det) and torch.equal(k_native, k_det):
                        return AttentionPreprocessResult(
                            q=q_native,
                            k=k_native,
                            backend_ids=MappingProxyType(
                                {
                                    "qk_rmsnorm": self.native_qk_norm_backend_id,
                                    "rope": self.native_rope_backend_id,
                                }
                            ),
                            fallback=False,
                            device_capability=self.device_capability,
                            probe_id=probe_id,
                            policy_id=self.policy_id,
                        )
                fallback_reason = "native_preprocess_bitwise_probe_failed"
            except Exception as exc:  # framework backend failures use the common path
                fallback_reason = f"native_preprocess_unavailable:{type(exc).__name__}"
        else:
            fallback_reason = "native_preprocess_not_supplied"
        if self.require_native_qk_norm:
            raise TransformerEngineRMSNormUnavailable(fallback_reason)
        return AttentionPreprocessResult(
            q=q_det,
            k=k_det,
            backend_ids=self.deterministic_backend_ids,
            fallback=True,
            device_capability=self.device_capability,
            fallback_reason=fallback_reason,
            probe_id=probe_id,
            policy_id=self.policy_id,
        )


class RocmAttentionPreprocessor(H100AttentionPreprocessor):
    """Reuse TE QK-Norm with the RL-Kernel deterministic ROCm RoPE boundary."""

    def __init__(
        self,
        device: torch.device | str | int | None = None,
        *,
        native_qk_norm: Callable[..., Tensor] | None = None,
        reuse_transformer_engine_qk_norm: bool = True,
        require_transformer_engine_qk_norm: bool = True,
        policy_id: str = PREPROCESS_POLICY_ID,
    ) -> None:
        if torch.version.hip is None or not torch.cuda.is_available():
            raise RuntimeError("RocmAttentionPreprocessor requires an available ROCm runtime")
        current_device = torch.cuda.current_device()
        self.device = torch.device("cuda", current_device)
        if device is not None:
            self.device = (
                torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
            )
        if self.device.type != "cuda":
            raise RuntimeError(f"RocmAttentionPreprocessor requires ROCm, got {self.device}")
        if self.device.index is None:
            self.device = torch.device("cuda", current_device)

        from rl_engine.backends.rocm.rope.rope import RocmDeterministicRoPEOp
        from rl_engine.backends.shared.triton.norm.rmsnorm import RMSNormTritonOp

        self.rmsnorm = RMSNormTritonOp()
        self.rope = RocmDeterministicRoPEOp()
        self.deterministic_backend_ids = ROCM_ATTENTION_PREPROCESS_BACKENDS
        self.device_capability = (0, 0)
        if native_qk_norm is None and reuse_transformer_engine_qk_norm:
            try:
                native_qk_norm = TransformerEngineRMSNormOp()
            except TransformerEngineRMSNormUnavailable:
                if require_transformer_engine_qk_norm:
                    raise
        self.native_qk_norm = native_qk_norm
        self.native_rope = None
        self.require_native_qk_norm = bool(
            require_transformer_engine_qk_norm and reuse_transformer_engine_qk_norm
        )
        self.native_qk_norm_backend_id = (
            getattr(native_qk_norm, "backend_id", TE_ROCM_QK_RMSNORM_BACKEND_ID)
            if native_qk_norm is not None
            else TE_ROCM_QK_RMSNORM_BACKEND_ID
        )
        self.native_rope_backend_id = ROCM_ROPE_BACKEND_ID
        self.policy_id = policy_id


def _apply_deterministic_rope(
    rope: Callable[..., Tensor],
    x: Tensor,
    positions: Tensor,
    theta: float,
) -> Tensor:
    """Adapt per-sample positions to the CUDA RoPE operator's 1-D contract."""

    if positions.dim() == 1:
        return rope(x, positions, theta=theta)
    return torch.cat(
        [rope(x[index : index + 1], positions[index], theta=theta) for index in range(x.shape[0])],
        dim=0,
    )


def _probe_configuration_id(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    positions: Tensor,
    eps: float,
    theta: float,
) -> str:
    payload = {
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "q_dtype": str(q.dtype),
        "k_dtype": str(k.dtype),
        "weight_dtype": str(q_weight.dtype),
        "positions_shape": list(positions.shape),
        "positions_sha256": hashlib.sha256(positions.detach().cpu().numpy().tobytes()).hexdigest(),
        "eps": float(eps),
        "theta": float(theta),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    positions: Tensor,
    device: torch.device,
) -> None:
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("q and k must use [B, H, S, D] layout")
    if q.shape[0] != k.shape[0] or q.shape[-2:] != k.shape[-2:]:
        raise ValueError("q and k must have the same batch, sequence, and head dimensions")
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16:
        raise TypeError("the frozen Attention experiment requires BF16 q and k")
    if q.device != device or k.device != device:
        raise ValueError(f"q and k must both be on the configured device {device}")
    for name, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
        if weight.shape != (q.shape[-1],):
            raise ValueError(f"{name} must have shape ({q.shape[-1]},)")
        if weight.device != device or weight.dtype is not torch.bfloat16:
            raise ValueError(f"{name} must be BF16 on {device}")
    if positions.device != device:
        raise ValueError(f"positions must be on {device}")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must use int32 or int64 global token indices")
    expected = (q.shape[-2],) if positions.dim() == 1 else (q.shape[0], q.shape[-2])
    if positions.dim() not in (1, 2) or tuple(positions.shape) != expected:
        raise ValueError(f"positions must have shape [S] or [B, S], expected {expected}")


__all__ = [
    "ALLOWED_ATTENTION_PREPROCESS_BACKENDS",
    "AttentionPreprocessResult",
    "H100AttentionPreprocessor",
    "MANDATED_ATTENTION_PREPROCESS_BACKENDS",
    "ROCM_ATTENTION_PREPROCESS_BACKENDS",
    "RocmAttentionPreprocessor",
    "QK_RMSNORM_BACKEND_ID",
    "ROCM_QK_RMSNORM_BACKEND_ID",
    "ROCM_ROPE_BACKEND_ID",
    "ROPE_BACKEND_ID",
    "NATIVE_QK_RMSNORM_BACKEND_ID",
    "NATIVE_ROPE_BACKEND_ID",
    "PREPROCESS_POLICY_ID",
    "TE_CUDA_QK_RMSNORM_BACKEND_ID",
    "TE_ROCM_QK_RMSNORM_BACKEND_ID",
    "TransformerEngineRMSNormOp",
    "TransformerEngineRMSNormUnavailable",
]
