# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Unified Attention entry point for the PR230 cross-configuration matrix.

The matrix needs one stable callable shape even though training and rollout may
materialize different Attention backends. This adapter owns the common
contract checks and provenance only; numerical work remains in the qualified
CUDA/ROCm production cores or the explicit RL-Kernel reference core.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, cast

import torch
from torch import Tensor

from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_CORE_ID,
    STRICT_ATTENTION_SCHEDULE_ID,
    AttentionContract,
    AttentionContractError,
    AttentionDType,
    SplitKVMode,
)

BACKEND_ID = "rlkernel.attention.deterministic.v1"
REFERENCE_BACKEND_ID = "rlkernel.attention.reference.v1"
_STRICT_AG_RS_BACKENDS = frozenset({"self_owned_cuda_ag_rs", "cuda_ag_rs", "rccl_ag_rs"})

_TORCH_DTYPES = {
    AttentionDType.BF16: torch.bfloat16,
    AttentionDType.FP16: torch.float16,
    AttentionDType.FP32: torch.float32,
}


@dataclass(frozen=True)
class AttentionAblationConfig:
    """Per-invocation settings materialized by the ablation runner."""

    backend: str = "auto"
    deterministic: bool = True
    communication_backend: str = "none"
    return_lse: bool = True
    return_gradients: bool = False
    strict_core_id: str = STRICT_ATTENTION_CORE_ID
    strict_schedule: str = STRICT_ATTENTION_SCHEDULE_ID
    validate: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise AttentionContractError("Attention backend must be a non-empty string")
        if not isinstance(self.deterministic, bool):
            raise AttentionContractError("deterministic must be a bool")
        if (
            not isinstance(self.communication_backend, str)
            or not self.communication_backend.strip()
        ):
            raise AttentionContractError("communication_backend must be a non-empty string")
        for name in ("return_lse", "return_gradients", "validate"):
            if not isinstance(getattr(self, name), bool):
                raise AttentionContractError(f"{name} must be a bool")
        if not isinstance(self.strict_core_id, str) or not self.strict_core_id.strip():
            raise AttentionContractError("strict_core_id must be a non-empty string")
        if not isinstance(self.strict_schedule, str) or not self.strict_schedule.strip():
            raise AttentionContractError("strict_schedule must be a non-empty string")
        object.__setattr__(self, "backend", self.backend.strip().lower())
        object.__setattr__(self, "communication_backend", self.communication_backend.strip())


@dataclass(frozen=True)
class AttentionAblationResult:
    """Standardized Attention result consumed by cross-config artifacts."""

    out: Tensor
    lse: Tensor | None
    dq: Tensor | None = None
    dk: Tensor | None = None
    dv: Tensor | None = None
    backend_id: str = BACKEND_ID
    deterministic: bool = True
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.out, Tensor):
            raise TypeError("Attention result out must be a torch.Tensor")
        if self.lse is not None and not isinstance(self.lse, Tensor):
            raise TypeError("Attention result lse must be a torch.Tensor or None")
        for name in ("dq", "dk", "dv"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(f"Attention result {name} must be a torch.Tensor or None")
        if not isinstance(self.backend_id, str) or not self.backend_id.strip():
            raise ValueError("Attention result backend_id must be non-empty")
        if not isinstance(self.deterministic, bool):
            raise TypeError("Attention result deterministic must be a bool")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("Attention result provenance must be a mapping")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def out_lse(self) -> tuple[Tensor, Tensor | None]:
        """Compatibility tuple for callers that consume ``(out, lse)``."""

        return self.out, self.lse

    def readback(self) -> dict[str, Any]:
        """Return JSON-compatible execution evidence for PR230 artifacts."""

        return {
            "backend_id": self.backend_id,
            "deterministic": self.deterministic,
            "out_shape": list(self.out.shape),
            "out_dtype": str(self.out.dtype).replace("torch.", ""),
            "lse_shape": None if self.lse is None else list(self.lse.shape),
            "lse_dtype": (None if self.lse is None else str(self.lse.dtype).replace("torch.", "")),
            "gradients": {
                "dq": self.dq is not None,
                "dk": self.dk is not None,
                "dv": self.dv is not None,
            },
            "provenance": dict(self.provenance),
        }


class AttentionAblationOp:
    """PR230/PR314-style unified Attention wrapper.

    ``core`` and ``reference`` are injectable so the wrapper is usable by the
    semantic operator session without importing CUDA at construction time.
    ``core`` should expose ``forward_with_lse``; ``reference`` is the existing
    pure-PyTorch CP reference with the same method.
    """

    op_class = "attention"
    is_batch_invariant = True
    backend_id = BACKEND_ID

    def __init__(
        self,
        *,
        core: Any | None = None,
        reference: Any | None = None,
        native: Any | None = None,
        cp_backend: Any | None = None,
        communication_backend: str = "none",
    ) -> None:
        if not isinstance(communication_backend, str) or not communication_backend.strip():
            raise AttentionContractError("communication_backend must be a non-empty string")
        self.core = core
        self.reference = reference
        self.native = native
        # CP production execution is injected by the runtime adapter. Keeping
        # it separate from the single-device core prevents an accidental
        # fallback to the PyTorch reference when AG/RS is required.
        self.cp_backend = cp_backend
        self.communication_backend = communication_backend.strip()
        self._runtime_group: Any = None
        self._runtime_platform: str | None = None

    def bind_accelerator_runtime(
        self,
        tensor: Tensor,
        *,
        process_group: Any = None,
    ) -> Any:
        """Bind the strict runtime matching a CUDA or ROCm tensor.

        PyTorch intentionally exposes AMD GPUs through the ``cuda`` device
        type.  The HIP build marker, rather than ``tensor.device.type``, is
        therefore the platform discriminator used by every framework bridge.
        """

        if tensor.device.type != "cuda":
            raise AttentionContractError(
                "production Attention requires a CUDA or ROCm accelerator tensor"
            )
        platform = "rocm" if torch.version.hip is not None else "cuda"
        return self.bind_runtime(platform=platform, process_group=process_group)

    def bind_runtime(self, *, platform: str, process_group: Any = None) -> Any:
        """Bind one platform runtime and reject process-local identity drift."""

        normalized = platform.strip().lower()
        if normalized not in {"cuda", "rocm"}:
            raise AttentionContractError("Attention runtime platform must be 'cuda' or 'rocm'")
        if self._runtime_platform is not None:
            if normalized != self._runtime_platform:
                raise AttentionContractError(
                    "Attention runtime is already bound to another accelerator platform"
                )
            if process_group is not self._runtime_group:
                raise AttentionContractError(
                    "Attention runtime is already bound to another process group"
                )
            return self.core

        runtime: Any
        if normalized == "rocm":
            from rl_engine.kernels.ops.rocm.attention.strict_runtime import (
                StrictRocmAttentionRuntime,
            )

            runtime = StrictRocmAttentionRuntime(process_group=process_group)
        else:
            from rl_engine.kernels.ops.cuda.attention.strict_runtime import (
                StrictCUDAAttentionRuntime,
            )

            runtime = StrictCUDAAttentionRuntime(process_group=process_group)
        self.core = runtime
        self.cp_backend = runtime
        self._runtime_group = process_group
        self._runtime_platform = normalized
        return runtime

    def bind_cuda_runtime(self, *, process_group: Any = None) -> Any:
        """Compatibility entry point for CUDA-only callers."""

        return self.bind_runtime(platform="cuda", process_group=process_group)

    def bind_rocm_runtime(self, *, process_group: Any = None) -> Any:
        """Bind the AITER/CK + RCCL production runtime once per process."""

        return self.bind_runtime(platform="rocm", process_group=process_group)

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        contract: AttentionContract,
        config: AttentionAblationConfig | Mapping[str, Any] | None = None,
        backend: str | Callable[..., Any] | None = None,
        deterministic: bool | None = None,
        return_lse: bool | None = None,
        return_gradients: bool | None = None,
        dout: Tensor | None = None,
        communication_backend: str | None = None,
        validate: bool | None = None,
        **kwargs: Any,
    ) -> AttentionAblationResult:
        return self.apply(
            q,
            k,
            v,
            contract=contract,
            config=config,
            backend=backend,
            deterministic=deterministic,
            return_lse=return_lse,
            return_gradients=return_gradients,
            dout=dout,
            communication_backend=communication_backend,
            validate=validate,
            **kwargs,
        )

    def apply(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        contract: AttentionContract,
        config: AttentionAblationConfig | Mapping[str, Any] | None = None,
        backend: str | Callable[..., Any] | None = None,
        deterministic: bool | None = None,
        return_lse: bool | None = None,
        return_gradients: bool | None = None,
        dout: Tensor | None = None,
        communication_backend: str | None = None,
        validate: bool | None = None,
        **kwargs: Any,
    ) -> AttentionAblationResult:
        if not isinstance(contract, AttentionContract):
            raise AttentionContractError("contract must be an AttentionContract")
        backend_request = (
            backend
            if callable(backend)
            or hasattr(backend, "forward_with_lse")
            or hasattr(backend, "apply")
            else None
        )
        cfg = _resolve_config(
            config,
            backend=backend if isinstance(backend, str) else None,
            deterministic=deterministic,
            return_lse=return_lse,
            return_gradients=return_gradients,
            communication_backend=(
                communication_backend
                if communication_backend is not None
                else self.communication_backend
            ),
            validate=validate,
        )
        if cfg.validate:
            self._validate_inputs(q, k, v, contract)
        if cfg.deterministic and contract.split_kv.mode is not SplitKVMode.DISABLED:
            raise AttentionContractError(
                "strict deterministic Attention requires Split-KV to be disabled"
            )
        if (
            cfg.deterministic
            and contract.sharding.cp_world_size > 1
            and cfg.communication_backend not in _STRICT_AG_RS_BACKENDS
        ):
            raise AttentionContractError(
                "strict CP Attention requires an explicit CUDA AG/RS or ROCm RCCL AG/RS backend"
            )
        if cfg.deterministic and not cfg.return_lse:
            raise AttentionContractError("strict deterministic Attention must return LSE")
        if cfg.return_gradients and dout is None:
            raise AttentionContractError("dout is required when return_gradients=True")
        if dout is not None and dout.shape != q.shape:
            raise AttentionContractError("dout must have the same shape as q")

        requested = backend_request if backend_request is not None else cfg.backend
        selected, selected_id = self._select_backend(
            requested,
            q,
            contract,
            communication_backend=cfg.communication_backend,
        )
        if cfg.deterministic and selected_id == "native":
            raise AttentionContractError(
                "deterministic=True cannot execute an unverified native Attention backend"
            )
        selected_core_id = getattr(selected, "core_id", None)
        selected_schedule = getattr(selected, "strict_schedule", None)
        if cfg.deterministic and selected_id not in {BACKEND_ID, REFERENCE_BACKEND_ID}:
            if selected_core_id != cfg.strict_core_id or selected_schedule != cfg.strict_schedule:
                raise AttentionContractError(
                    "deterministic Attention requires the shared strict core and schedule"
                )

        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("contract", contract)
        call_kwargs.setdefault("causal", contract.causal)
        call_kwargs.setdefault("scale", 1.0 / math.sqrt(contract.head_dim))
        call_kwargs.setdefault("cp_world_size", contract.sharding.cp_world_size)
        if contract.split_kv.mode is SplitKVMode.FIXED:
            call_kwargs.setdefault("kv_chunk_size", contract.split_kv.fixed_split_size)
        out, lse, backend_provenance = self._invoke(selected, q, k, v, call_kwargs, contract)
        if cfg.validate:
            self._validate_outputs(out, lse, q, contract)
        _validate_runtime_provenance(
            selected,
            selected_id,
            backend_provenance,
            cfg,
        )

        dq = dk = dv = None
        if cfg.return_gradients:
            dq, dk, dv = self._backward(selected, q, k, v, out, dout, call_kwargs)

        provenance = {
            "schema_version": "rlkernel.attention.ablation_result.v1",
            "semantic_operator": "attention",
            "backend_id": selected_id,
            "deterministic": cfg.deterministic,
            "strict_core_id": (cfg.strict_core_id if cfg.deterministic else None),
            "strict_schedule": cfg.strict_schedule if cfg.deterministic else None,
            "core_id": cfg.strict_core_id if cfg.deterministic else selected_id,
            "backend_deterministic": cfg.deterministic,
            "native_attention_arithmetic": (
                False if cfg.deterministic else selected_id == "native"
            ),
            "communication_backend": cfg.communication_backend,
            "communication_executed": bool(getattr(selected, "communication_executed", False)),
            "split_kv": contract.split_kv.to_dict(),
            "actual_split_kv": _actual_split_provenance(
                contract,
                total_kv_tokens=k.size(2) * contract.sharding.cp_world_size,
                backend=selected_id,
            ),
            "reduction": _reduction_provenance(contract),
            "contract_fingerprint": _contract_fingerprint(contract),
            "return_lse": cfg.return_lse,
            "return_gradients": cfg.return_gradients,
        }
        provenance.update(backend_provenance)
        provenance.setdefault("actual_backend", selected_id)
        provenance.setdefault("communication_backend", cfg.communication_backend)
        provenance.setdefault("production_ready", False)
        return AttentionAblationResult(
            out=out,
            lse=lse if cfg.return_lse else None,
            dq=dq,
            dk=dk,
            dv=dv,
            backend_id=selected_id,
            deterministic=cfg.deterministic,
            provenance=provenance,
        )

    def apply_fp32(self, *args: Any, **kwargs: Any) -> AttentionAblationResult:
        """Stable fingerprint entry point used by ``OperatorSession``."""

        return self.apply(*args, **kwargs)

    def _select_backend(
        self,
        requested: str | Callable[..., Any],
        q: Tensor,
        contract: AttentionContract,
        *,
        communication_backend: str,
    ) -> tuple[Any, str]:
        if (
            callable(requested)
            or hasattr(requested, "forward_with_lse")
            or hasattr(requested, "apply")
        ):
            return requested, _callable_backend_id(requested)
        normalized = str(requested).strip().lower()
        if normalized in {"native", "te", "flashinfer"}:
            if self.native is None:
                raise AttentionContractError(
                    "native Attention backend was requested but no native callable was injected"
                )
            return self.native, "native"
        if normalized in {"reference", "pytorch_reference"}:
            return self._reference_backend(), REFERENCE_BACKEND_ID
        if normalized not in {"auto", "deterministic", "rlkernel"}:
            raise AttentionContractError(f"unsupported Attention backend {requested!r}")
        if contract.sharding.cp_world_size > 1:
            if communication_backend in _STRICT_AG_RS_BACKENDS:
                if self.cp_backend is None:
                    raise AttentionContractError(
                        "CP production Attention requires an injected AG/RS backend"
                    )
                return self.cp_backend, _callable_backend_id(self.cp_backend)
            return self._reference_backend(), REFERENCE_BACKEND_ID
        if q.device.type != "cuda":
            return self._reference_backend(), REFERENCE_BACKEND_ID
        if torch.version.hip is not None:
            if self.core is None:
                self.bind_rocm_runtime()
            return self.core, _callable_backend_id(self.core)
        if self.core is None:
            from rl_engine.kernels.ops.cuda.attention.deterministic_attn import (
                DeterministicAttentionOp,
            )

            self.core = DeterministicAttentionOp()
        return self.core, BACKEND_ID

    def _reference_backend(self) -> Any:
        if self.reference is None:
            from rl_engine.kernels.ops.pytorch.attention.cp_attention import (
                DeterministicCPAttentionReferenceOp,
            )

            self.reference = DeterministicCPAttentionReferenceOp(strict_bitwise=True)
        return self.reference

    @staticmethod
    def _validate_inputs(q: Tensor, k: Tensor, v: Tensor, contract: AttentionContract) -> None:
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise AttentionContractError("q, k, and v must use [B, H, S, D] layout")
        expected_dtype = _TORCH_DTYPES[contract.dtype]
        if (
            q.dtype is not expected_dtype
            or k.dtype is not expected_dtype
            or v.dtype is not expected_dtype
        ):
            raise AttentionContractError(
                f"q, k, and v must match contract dtype {contract.dtype.value}"
            )
        if q.device != k.device or q.device != v.device:
            raise AttentionContractError("q, k, and v must be on the same device")
        batch, q_heads, q_seq, dim = q.shape
        if batch != contract.batch_size:
            raise AttentionContractError(
                f"q batch={batch} does not match contract batch_size={contract.batch_size}"
            )
        if q_seq != contract.query_sequence_length:
            raise AttentionContractError(
                "q sequence length does not match AttentionContract query_sequence_length"
            )
        sharding = contract.sharding
        if q_heads != sharding.local_q_heads or k.shape[1] != sharding.local_kv_heads:
            raise AttentionContractError("q/k head counts do not match TP sharding in contract")
        if (
            k.shape[0] != batch
            or v.shape[:3] != k.shape[:3]
            or k.shape[-1] != dim
            or v.shape[-1] != dim
        ):
            raise AttentionContractError("q, k, and v shapes are inconsistent")
        if dim != contract.head_dim:
            raise AttentionContractError("tensor head_dim does not match AttentionContract")

    @staticmethod
    def _validate_outputs(out: Tensor, lse: Tensor, q: Tensor, contract: AttentionContract) -> None:
        if out.shape != q.shape:
            raise AttentionContractError(
                f"Attention output shape {tuple(out.shape)} does not match q {tuple(q.shape)}"
            )
        expected_lse = (q.shape[0], q.shape[1], q.shape[2])
        if lse.shape != expected_lse:
            raise AttentionContractError(
                f"attention-domain LSE shape {tuple(lse.shape)} does not match {expected_lse}"
            )
        if lse.dtype is not torch.float32:
            raise AttentionContractError("attention-domain LSE must remain fp32")
        expected_dtype = _TORCH_DTYPES[contract.dtype]
        if out.dtype is not expected_dtype:
            raise AttentionContractError(
                f"Attention output must be written in {contract.dtype.value}, got {out.dtype}"
            )

    @staticmethod
    def _invoke(
        backend: Any,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kwargs: Mapping[str, Any],
        contract: AttentionContract,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        method = getattr(backend, "forward_with_lse", None)
        if not callable(method):
            method = getattr(backend, "apply", None)
        if not callable(method):
            method = backend if callable(backend) else None
        if method is None:
            raise AttentionContractError(
                "Attention backend must expose forward_with_lse, apply, or __call__"
            )
        accepted = _accepted_kwargs(method, kwargs)
        if contract.sharding.cp_world_size > 1 and not _declares_keyword(method, "cp_world_size"):
            raise AttentionContractError(
                "CP>1 requires an Attention backend that explicitly accepts cp_world_size"
            )
        result = method(q, k, v, **accepted)
        backend_provenance: dict[str, Any] = {}
        out: Any
        lse: Any
        if isinstance(result, AttentionAblationResult):
            out, lse = result.out, result.lse
        elif isinstance(result, tuple) and len(result) == 2:
            out, lse = result
        else:
            out = getattr(result, "out", None)
            lse = getattr(result, "lse", None)
            raw_provenance = getattr(result, "provenance", {})
            if isinstance(raw_provenance, Mapping):
                backend_provenance = dict(raw_provenance)
            if out is None or lse is None:
                raise AttentionContractError(
                    "Attention backend must return (out, lse), AttentionAblationResult, "
                    "or an object with out/lse/provenance"
                )
        if isinstance(result, AttentionAblationResult):
            backend_provenance = dict(result.provenance)
        if not isinstance(out, Tensor) or not isinstance(lse, Tensor):
            raise AttentionContractError("Attention backend returned non-tensor output or LSE")
        return out, lse, backend_provenance

    @staticmethod
    def _backward(
        backend: Any,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out: Tensor,
        dout: Tensor | None,
        kwargs: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        backward = getattr(backend, "backward_reference", None)
        if callable(backward):
            result = backward(q, k, v, dout, **_accepted_kwargs(backward, kwargs))
            gradients = getattr(result, "gradients", None)
            if gradients is not None:
                return gradients.dq, gradients.dk, gradients.dv
        if dout is None:
            raise AttentionContractError("dout is required to compute Attention gradients")
        if not out.requires_grad:
            raise AttentionContractError(
                "Attention backend did not retain an autograd graph for gradients"
            )
        return cast(
            tuple[Tensor, Tensor, Tensor],
            torch.autograd.grad(
                out,
                (q, k, v),
                grad_outputs=dout.to(dtype=out.dtype),
                allow_unused=False,
                retain_graph=True,
            ),
        )


def _resolve_config(
    config: AttentionAblationConfig | Mapping[str, Any] | None,
    **overrides: Any,
) -> AttentionAblationConfig:
    if config is None:
        values: dict[str, Any] = {}
    elif isinstance(config, AttentionAblationConfig):
        values = {
            name: getattr(config, name)
            for name in (
                "backend",
                "deterministic",
                "communication_backend",
                "return_lse",
                "return_gradients",
                "strict_core_id",
                "strict_schedule",
                "validate",
            )
        }
    elif isinstance(config, Mapping):
        values = dict(config)
    else:
        raise AttentionContractError("config must be AttentionAblationConfig, mapping, or None")
    values.update({name: value for name, value in overrides.items() if value is not None})
    return AttentionAblationConfig(**values)


def _accepted_kwargs(method: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return dict(kwargs)
    parameters = signature.parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return dict(kwargs)
    return {name: value for name, value in kwargs.items() if name in signature.parameters}


def _declares_keyword(method: Callable[..., Any], name: str) -> bool:
    """Return whether ``method`` explicitly declares a keyword-capable parameter."""

    try:
        parameter = inspect.signature(method).parameters.get(name)
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


def _callable_backend_id(value: Any) -> str:
    explicit = getattr(value, "backend_id", None) or getattr(
        value, "__attention_backend_id__", None
    )
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return f"injected.{type(value).__module__}.{type(value).__qualname__}"


def _validate_runtime_provenance(
    selected: Any,
    selected_id: str,
    runtime: Mapping[str, Any],
    config: AttentionAblationConfig,
) -> None:
    """Fail closed when a production strict backend did not prove its identity."""

    if not config.deterministic:
        return
    if selected_id == "native":
        raise AttentionContractError(
            "deterministic Attention cannot execute native Attention arithmetic"
        )

    external_production_backend = selected_id not in {BACKEND_ID, REFERENCE_BACKEND_ID}
    if not external_production_backend:
        return

    expected = {
        "strict_core_id": config.strict_core_id,
        "strict_schedule": config.strict_schedule,
        "actual_backend": selected_id,
        "production_ready": True,
        "fallback": False,
        "reference_only": False,
    }
    if config.communication_backend in _STRICT_AG_RS_BACKENDS:
        expected["communication_backend"] = config.communication_backend
    mismatches = [name for name, value in expected.items() if runtime.get(name) != value]
    if not isinstance(runtime.get("native_attention_arithmetic"), bool):
        mismatches.append("native_attention_arithmetic")
    if mismatches:
        raise AttentionContractError(
            "production Attention runtime provenance is incomplete or mismatched: "
            + ", ".join(mismatches)
        )


def _actual_split_provenance(
    contract: AttentionContract,
    *,
    total_kv_tokens: int,
    backend: str = BACKEND_ID,
) -> dict[str, Any]:
    plan = contract.split_kv.resolve(
        total_kv_tokens,
        backend=backend,
    )
    return plan.to_dict()


def _reduction_provenance(contract: AttentionContract) -> dict[str, Any]:
    reduction = contract.reduction
    return {
        "merge": reduction.merge.value,
        "acc_dtype": reduction.acc_dtype.value,
        "order": reduction.order.value,
        "downcast_at": reduction.downcast_at.value,
        "engine": reduction.engine.value,
    }


def _contract_fingerprint(contract: AttentionContract) -> str:
    payload = json.dumps(contract.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AttentionAblationConfig",
    "AttentionAblationOp",
    "AttentionAblationResult",
    "BACKEND_ID",
    "REFERENCE_BACKEND_ID",
]
