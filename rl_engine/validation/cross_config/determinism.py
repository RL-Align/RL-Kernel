# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Cross-side determinism probing for the Megatron + vLLM cross-config target.

Both frameworks ship a "make this deterministic" switch, but they mean different
things by it, and neither knows the other exists:

``Megatron`` ``ModelParallelConfig.deterministic_mode``
    Asserts ``NCCL_ALGO`` is one of five values, forbids FlashAttention and fused
    cross-entropy, calls ``torch.use_deterministic_algorithms(True)``, and requires
    ``NVTE_ALLOW_NONDETERMINISTIC_ALGO == 0``. It does **not** touch TF32, BF16
    reduced-precision reduction, cuBLAS workspace, NCCL protocol, or NCCL channel
    counts.

``vLLM`` ``VLLM_BATCH_INVARIANT``
    Replaces ``aten::mm/addmm/matmul/linear/bmm``, ``log_softmax``/``softmax``,
    ``mean.dim`` and ``rms_norm`` with Triton kernels, disables TF32 and BF16/FP16
    reduced-precision reduction, pins cuBLAS workspace and the BLAS library, and
    hard-sets ten NCCL environment variables.

So a run can have both switches on and still be comparing two different notions of
determinism. This module makes that difference explicit and, where it changes the
numerics, blocking. It never imports Megatron or vLLM: probes are built from plain
mappings so the logic is testable on any machine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from rl_engine.integrations.common.attention_binding import (
    BindingErrorCode,
    BindingIssue,
    BindingTier,
)

__all__ = [
    "COMPARED_NCCL_KEYS",
    "DeterminismProbe",
    "DeterminismReport",
    "compare_determinism",
    "megatron_probe_from_config",
    "vllm_probe_from_env",
]


#: Environment keys whose value can change a reduction result. Compared across
#: sides; a difference is reported, and a difference in the *arithmetic* subset is
#: blocking. Ordering is fixed so the fingerprint is stable.
COMPARED_NCCL_KEYS: tuple[str, ...] = (
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_NTHREADS",
    "NCCL_SOCKET_NTHREADS",
    "NCCL_COLLNET_ENABLE",
    "NCCL_NVLS_ENABLE",
    "NCCL_P2P_NET_DISABLE",
    "NCCL_LAUNCH_MODE",
    "CUBLAS_WORKSPACE_CONFIG",
)


#: The subset above that changes arithmetic rather than only scheduling. A mismatch
#: here fails the binding closed; a mismatch in the remainder is recorded only.
_ARITHMETIC_NCCL_KEYS: frozenset[str] = frozenset(
    {"NCCL_ALGO", "NCCL_PROTO", "CUBLAS_WORKSPACE_CONFIG"}
)


@dataclass(frozen=True)
class DeterminismProbe:
    """What one side actually has switched on.

    ``tf32_disabled`` and ``bf16_reduced_precision_reduction`` are tri-state on
    purpose: ``None`` means "the framework does not manage this", which is exactly
    Megatron's situation and is itself the finding.
    """

    side: str
    framework: str
    mode_flag: str
    enabled: bool
    env: Mapping[str, Any] = field(default_factory=dict)
    tf32_disabled: Optional[bool] = None
    bf16_reduced_precision_reduction: Optional[bool] = None
    forbids_flash_attention: Optional[bool] = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "cross_config.determinism_probe.v1"

    def __post_init__(self) -> None:
        if self.side not in ("rollout", "training"):
            raise ValueError("side must be 'rollout' or 'training'")
        if not self.framework:
            raise ValueError("framework must not be empty")
        object.__setattr__(self, "env", dict(self.env))
        object.__setattr__(self, "evidence", dict(self.evidence))

    @property
    def env_fingerprint(self) -> str:
        payload = {key: self.env.get(key) for key in COMPARED_NCCL_KEYS}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "side": self.side,
            "framework": self.framework,
            "mode_flag": self.mode_flag,
            "enabled": self.enabled,
            "env": {key: self.env.get(key) for key in COMPARED_NCCL_KEYS},
            "env_fingerprint": self.env_fingerprint,
            "tf32_disabled": self.tf32_disabled,
            "bf16_reduced_precision_reduction": self.bf16_reduced_precision_reduction,
            "forbids_flash_attention": self.forbids_flash_attention,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class DeterminismReport:
    """Cross-side comparison result."""

    rollout: DeterminismProbe
    training: DeterminismProbe
    issues: tuple[BindingIssue, ...] = ()
    differences: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    schema_version: str = "cross_config.determinism_report.v1"

    @property
    def compatible(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "compatible": self.compatible,
            "rollout": self.rollout.to_dict(),
            "training": self.training.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "differences": {key: dict(value) for key, value in self.differences.items()},
        }


def megatron_probe_from_config(
    config: Any,
    env: Optional[Mapping[str, str]] = None,
) -> DeterminismProbe:
    """Build a training-side probe from a Megatron config object.

    ``config`` is duck-typed (anything exposing ``deterministic_mode`` and
    optionally ``attention_backend`` / ``cross_entropy_loss_fusion``) so this works
    against a real ``ModelParallelConfig``, a test double, or a plain namespace,
    and so importing this module never requires Megatron.

    ``tf32_disabled`` and ``bf16_reduced_precision_reduction`` are reported as
    ``None`` because Megatron does not manage them -- a ``grep`` for ``allow_tf32``
    and ``fp32_precision`` across ``megatron/`` returns nothing. That asymmetry
    against vLLM is the point of :func:`compare_determinism`.
    """

    environ = dict(env or {})
    enabled = bool(getattr(config, "deterministic_mode", False))
    return DeterminismProbe(
        side="training",
        framework="megatron",
        mode_flag="deterministic_mode",
        enabled=enabled,
        env={key: environ.get(key) for key in COMPARED_NCCL_KEYS},
        tf32_disabled=None,
        bf16_reduced_precision_reduction=None,
        forbids_flash_attention=enabled,
        evidence={
            "nvte_allow_nondeterministic_algo": environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO"),
            "cross_entropy_loss_fusion": getattr(config, "cross_entropy_loss_fusion", None),
            "attention_backend": _enum_value(getattr(config, "attention_backend", None)),
            "tensor_model_parallel_size": getattr(config, "tensor_model_parallel_size", None),
            "context_parallel_size": getattr(config, "context_parallel_size", None),
            "sequence_parallel": getattr(config, "sequence_parallel", None),
            "manages_tf32": False,
            "manages_bf16_reduced_precision_reduction": False,
        },
    )


def vllm_probe_from_env(
    env: Mapping[str, str],
    *,
    model_config: Any = None,
) -> DeterminismProbe:
    """Build a rollout-side probe from the vLLM process environment.

    ``VLLM_BATCH_INVARIANT`` is read from ``env`` rather than ``vllm.envs`` so the
    probe can be constructed from a remote worker's reported environment, which is
    how vime's Ray actors expose it.
    """

    enabled = str(env.get("VLLM_BATCH_INVARIANT", "0")).strip() in ("1", "true", "True")
    return DeterminismProbe(
        side="rollout",
        framework="vllm",
        mode_flag="VLLM_BATCH_INVARIANT",
        enabled=enabled,
        env={key: env.get(key) for key in COMPARED_NCCL_KEYS},
        # vLLM sets both to "ieee"/disabled inside init_batch_invariance().
        tf32_disabled=enabled or None,
        bf16_reduced_precision_reduction=(False if enabled else None),
        forbids_flash_attention=False,
        evidence={
            "vllm_allreduce_use_symm_mem": env.get("VLLM_ALLREDUCE_USE_SYMM_MEM"),
            "vllm_use_aot_compile": env.get("VLLM_USE_AOT_COMPILE"),
            "enforce_eager": getattr(model_config, "enforce_eager", None),
            "disable_cascade_attn": getattr(model_config, "disable_cascade_attn", None),
            "quantization": getattr(model_config, "quantization", None),
            "manages_tf32": True,
            "manages_bf16_reduced_precision_reduction": True,
        },
    )


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def compare_determinism(
    *,
    rollout: DeterminismProbe,
    training: DeterminismProbe,
) -> DeterminismReport:
    """Compare two probes and produce blocking issues plus recorded differences."""

    if rollout.side != "rollout" or training.side != "training":
        raise ValueError("compare_determinism expects one rollout probe and one training probe")

    issues: list[BindingIssue] = []
    differences: dict[str, dict[str, Any]] = {}

    for probe in (rollout, training):
        if not probe.enabled:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.DETERMINISM_INCOMPATIBLE,
                    tier=BindingTier.SEMANTIC,
                    field=f"{probe.side}.{probe.mode_flag}",
                    rollout=rollout.enabled,
                    training=training.enabled,
                    message=(
                        f"{probe.framework} {probe.mode_flag} is not enabled; the "
                        f"{probe.side} side is not batch-invariant and cannot anchor a "
                        "cross-config comparison"
                    ),
                )
            )

    for key in COMPARED_NCCL_KEYS:
        rollout_value = rollout.env.get(key)
        training_value = training.env.get(key)
        if rollout_value == training_value:
            continue
        differences[key] = {"rollout": rollout_value, "training": training_value}
        if key in _ARITHMETIC_NCCL_KEYS:
            issues.append(
                BindingIssue(
                    code=BindingErrorCode.DETERMINISM_INCOMPATIBLE,
                    tier=BindingTier.SEMANTIC,
                    field=f"env.{key}",
                    rollout=rollout_value,
                    training=training_value,
                    message=(
                        f"{key} differs between sides; the two sides would reduce with "
                        "different arithmetic and the resulting drift is not attributable"
                    ),
                )
            )

    # Megatron reports None for these because it does not manage them at all. That is
    # recorded rather than blocking: under a pure BF16 GEMM path TF32 does not fire, and
    # forcing Megatron to manage it is out of scope for this PR. It is surfaced so the
    # asymmetry appears in every artifact instead of being invisible.
    for name in ("tf32_disabled", "bf16_reduced_precision_reduction"):
        rollout_value = getattr(rollout, name)
        training_value = getattr(training, name)
        if rollout_value != training_value:
            differences[name] = {
                "rollout": rollout_value,
                "training": training_value,
                "note": (
                    "megatron does not manage this setting; vllm sets it inside "
                    "init_batch_invariance()"
                ),
            }

    return DeterminismReport(
        rollout=rollout,
        training=training,
        issues=tuple(issues),
        differences=differences,
    )
