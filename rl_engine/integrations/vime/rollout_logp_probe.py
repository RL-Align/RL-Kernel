# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import inspect
import os
from dataclasses import asdict, dataclass
from typing import Any

ENV_ENABLED = "RL_KERNEL_VIME_LOGP_PROBE"
ENV_STRICT = "RL_KERNEL_VIME_LOGP_STRICT"
METADATA_KEY = "rl_kernel"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_CALL_COUNT = 0


@dataclass(frozen=True)
class RLKernelProbeResult:
    """Structured evidence that the Vime shim reached RL-Kernel."""

    enabled: bool
    invoked: bool
    call_count: int = 0
    op: str = "logp"
    backend: str | None = None
    fallback: bool = False
    fallback_reason: str | None = None
    output_shape: tuple[int, ...] | None = None
    output_sum: float | None = None


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _probe_tensors():
    import torch

    from rl_engine.platforms.device import device_ctx

    logits = torch.tensor(
        [
            [0.25, 1.5, -0.5, 0.0],
            [2.0, -1.0, 0.5, 0.25],
        ],
        device=device_ctx.device,
        dtype=torch.float32,
    )
    token_ids = torch.tensor([1, 0], device=device_ctx.device, dtype=torch.long)
    return logits, token_ids


def _fallback(reason: str) -> RLKernelProbeResult:
    return RLKernelProbeResult(
        enabled=True,
        invoked=False,
        call_count=_CALL_COUNT,
        fallback=True,
        fallback_reason=reason,
    )


def _supports_evaluation_arg(fn: Any) -> bool:
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        param.name == "evaluation" or param.kind == inspect.Parameter.VAR_KEYWORD
        for param in parameters
    )


def run_logp_probe(*, strict: bool | None = None) -> RLKernelProbeResult:
    """Invoke one RL-Kernel logp operator on a small deterministic tensor.

    The probe intentionally uses synthetic tensors instead of Vime rollout
    logits. Vime's rollout HTTP path exposes selected logprobs, not logits, so
    this is an invocation proof rather than a rollout-logprob replacement.
    """

    if strict is None:
        strict = _env_enabled(ENV_STRICT)

    try:
        from rl_engine.kernels.registry import kernel_registry

        logits, token_ids = _probe_tensors()
        op = kernel_registry.get_op("logp")
        output = op(logits, token_ids)
        global _CALL_COUNT
        _CALL_COUNT += 1
        return RLKernelProbeResult(
            enabled=True,
            invoked=True,
            call_count=_CALL_COUNT,
            backend=op.__class__.__name__,
            output_shape=tuple(output.shape),
            output_sum=float(output.detach().float().sum().item()),
        )
    except Exception as exc:
        if strict:
            raise
        return _fallback(f"{type(exc).__name__}: {exc}")


def _record_probe(sample: Any, result: RLKernelProbeResult) -> None:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        setattr(sample, "metadata", metadata)

    rl_kernel_metadata = metadata.get(METADATA_KEY)
    if not isinstance(rl_kernel_metadata, dict):
        rl_kernel_metadata = {}
        metadata[METADATA_KEY] = rl_kernel_metadata

    rl_kernel_metadata["vime_logp_probe"] = asdict(result)


async def custom_generate(
    args: Any,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Any:
    """Vime ``--custom-generate-function-path`` entry point.

    This shim preserves Vime's native generation path and only adds opt-in
    RL-Kernel invocation evidence. Enable it with
    ``RL_KERNEL_VIME_LOGP_PROBE=1``.
    """

    from vime.rollout.vllm_rollout import generate

    if _supports_evaluation_arg(generate):
        sample = await generate(args, sample, sampling_params, evaluation=evaluation)
    else:
        sample = await generate(args, sample, sampling_params)

    if not _env_enabled(ENV_ENABLED):
        _record_probe(sample, RLKernelProbeResult(enabled=False, invoked=False))
        return sample

    result = run_logp_probe(strict=_env_enabled(ENV_STRICT))
    _record_probe(sample, result)
    return sample
