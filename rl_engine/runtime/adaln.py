# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""AdaLN execution policy; arithmetic stays in the existing backend operators."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import torch

SHARED_PROFILE = "adaln-shared-fp32-tree-v1"
SELECT01_PROFILE = "adaln-select01-framework-v1"
INDEXED_PROFILE = "adaln-indexed-fp32-tree-v1"
_BACKENDS = {
    "rl_engine.reference.norm.adaln_modulation": ("NativeAdaLNModulationOp", "pytorch"),
    "rl_engine.backends.cuda.norm.adaln_modulation": ("CudaAdaLNModulationOp", "cuda"),
    "rl_engine.backends.shared.triton.norm.adaln_modulation": ("TritonAdaLNModulationOp", "triton"),
}


@dataclass(frozen=True)
class IndexedAdaLNQualification:
    """Explicit activation after external evidence review, never an attestation.

    Pass the same scope to acceptance and delivery. No scope is installed by
    default. Source/binary hashes and evidence stay in the external manifest.
    """

    backend: str
    device: str
    device_name: str
    capability: tuple[int, int]
    torch_version: str
    cuda_version: str
    triton_version: str | None = None

    def __post_init__(self):
        device = torch.device(self.device)
        if self.backend not in {"cuda", "triton"}:
            raise ValueError("indexed qualification requires cuda or triton backend")
        if device.type != "cuda" or device.index is None:
            raise ValueError("indexed qualification requires an explicit cuda device ordinal")
        if not self.device_name or not self.torch_version or not self.cuda_version:
            raise ValueError("indexed qualification requires explicit target/runtime identity")
        if len(self.capability) != 2 or any(type(v) is not int or v < 0 for v in self.capability):
            raise ValueError("indexed qualification requires a compute capability pair")
        if self.backend == "triton" and not self.triton_version:
            raise ValueError("indexed Triton qualification requires its runtime version")

    def check(self, backend, device, *, execution=False):
        """Fail closed outside the externally reviewed backend/target runtime."""
        device = torch.device(device)
        target = torch.device(self.device)
        compatible = (
            backend == self.backend
            and device.type == "cuda"
            and (device.index is None or device.index == target.index)
            and torch.version.hip is None
            and torch.__version__ == self.torch_version
            and torch.version.cuda == self.cuda_version
        )
        if compatible and backend == "triton":
            import triton

            compatible = triton.__version__ == self.triton_version
        if compatible and execution:
            compatible = (
                device.index == target.index
                and torch.cuda.get_device_name(device) == self.device_name
                and torch.cuda.get_device_capability(device) == self.capability
            )
        if not compatible:
            raise RuntimeError(
                "strict AdaLN rejects indexed backend/target/runtime outside qualification"
            )


def adaln_backend(op: Any) -> str | None:
    cls = type(op)
    entry = _BACKENDS.get(cls.__module__)
    if entry and getattr(sys.modules.get(cls.__module__), entry[0], None) is cls:
        return entry[1]
    return None


class AdaLNDispatch:
    """A resolution-local trace, updated only after successful operator execution.

    Strict enforces arithmetic and an explicit indexed activation scope. Forward
    observations never attest hardware, binaries, backward execution or profiling.
    """

    def __init__(
        self,
        implementation,
        trace,
        *,
        hidden,
        strict=False,
        indexed_qualification=None,
        device=None,
    ):
        self.implementation = implementation
        self.trace = trace
        self.hidden = hidden
        self.strict = strict
        self.indexed_qualification = indexed_qualification
        self.device = torch.device(device) if device is not None else None
        trace.update(
            actual_backend=None,
            observation=None,
            execution_dtype=None,
            output_dtype=None,
            execution_device=None,
            binary_path=None,
            binary_sha256=None,
            hardware_qualified=None,
            profiler_observed=False,
            semantic_version="adaln-v1",
            strict=strict,
        )
        self._resolution_trace = trace.copy()

    def __call__(self, x, modulation, *, eps=1e-6, modulate_index=None):
        return self.forward(x, modulation, eps=eps, modulate_index=modulate_index)

    def forward(self, x, modulation, *, eps=1e-6, modulate_index=None):
        self.trace.clear()
        self.trace.update(self._resolution_trace)
        backend = adaln_backend(self.implementation)
        indexed = modulate_index is not None
        profile = SELECT01_PROFILE if indexed else SHARED_PROFILE
        implementation = self.implementation
        shared_unsupported = False
        if indexed:
            from rl_engine.reference.norm.adaln_modulation import (
                _indexed_domain_reason,
                _validate_indexed_inputs,
                _validate_indexed_metadata,
            )

            _validate_indexed_metadata(x, modulation, modulate_index, eps)
            reason = _indexed_domain_reason(x, modulation, modulate_index, eps)
        else:
            from rl_engine.reference.norm.adaln_modulation import _int32_addressable, _validate

            _validate(x, modulation, eps)
            shared_unsupported = backend in {"cuda", "triton"} and (
                x.device.type != "cuda"
                or not _int32_addressable(x, modulation)
                or (
                    backend == "cuda"
                    and (torch.version.hip is not None or x.shape[0] > 65535 or x.shape[-1] > 4096)
                )
            )
        if self.strict:
            try:
                if shared_unsupported:
                    raise RuntimeError("strict AdaLN rejects unsupported shared backend domain")
                if indexed:
                    if reason is not None:
                        raise RuntimeError(
                            f"strict AdaLN rejects indexed/select01: requires {reason}"
                        )
                    if not getattr(self.implementation, "indexed_available", False):
                        raise RuntimeError(
                            "strict AdaLN rejects select01: indexed symbols unavailable"
                        )
                    if self.indexed_qualification is None:
                        raise RuntimeError(
                            "strict AdaLN rejects unqualified indexed/select01 execution"
                        )
                    self.indexed_qualification.check(backend, x.device, execution=True)
                if backend is None or (backend == "pytorch" and x.device.type != "cpu"):
                    raise RuntimeError("strict AdaLN backend has no declared replay profile")
                if backend in {"cuda", "triton"} and (
                    x.device.type != "cuda" or torch.version.hip is not None
                ):
                    raise RuntimeError("strict AdaLN GPU profile requires NVIDIA CUDA")
                if x.ndim != 3 or x.shape[-1] != self.hidden:
                    raise ValueError("strict AdaLN input hidden differs from resolved hidden")
                if self.device is not None and (
                    self.device.type != x.device.type
                    or (self.device.index is not None and self.device.index != x.device.index)
                ):
                    raise RuntimeError("strict AdaLN rejects device differing from resolution")
            except (ValueError, RuntimeError):
                # Preserve input-error precedence on rejection. Successful calls
                # delegate value validation to the public backend exactly once.
                if indexed and reason not in {"int32 indexing", "B <= 65535"}:
                    _validate_indexed_inputs(x, modulation, modulate_index, eps)
                raise
        if indexed:
            if backend in {"cuda", "triton"}:
                if getattr(implementation, "indexed_available", False) and reason is None:
                    profile = INDEXED_PROFILE
                else:
                    from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp

                    implementation = NativeAdaLNModulationOp()
                    backend = "pytorch"
            outputs = implementation(x, modulation, eps=eps, modulate_index=modulate_index)
        else:
            if shared_unsupported:
                from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp

                implementation = NativeAdaLNModulationOp()
                backend = "pytorch"
            outputs = implementation(x, modulation, eps=eps)
        self.trace.update(
            actual_backend=backend,
            observation="successful_forward",
            arithmetic_profile=profile,
            execution_dtype=str(x.dtype).removeprefix("torch."),
            output_dtype=str(outputs[0].dtype).removeprefix("torch."),
            execution_device=str(x.device),
        )
        if indexed and profile == SELECT01_PROFILE:
            self.trace.update(
                fallback=True,
                fallback_reason=(
                    f"indexed_{adaln_backend(self.implementation)}_domain_unsupported"
                    if self._resolution_trace["arithmetic_profile"] == INDEXED_PROFILE
                    else "modulate_index_requires_select01"
                ),
                reduction_order="pairwise_lower_upper_H; framework_select01_backward",
                kernel_fingerprint="adaln_modulation_select01_native_v1",
            )
        elif indexed:
            self.trace.update(
                fallback=False,
                fallback_reason=None,
                reduction_order="pairwise_lower_upper_H; ascending_selected_token_S_per_bank",
                kernel_fingerprint=f"adaln_indexed_{backend}_v1",
            )
        elif self._resolution_trace["arithmetic_profile"] in {INDEXED_PROFILE, SELECT01_PROFILE}:
            self.trace.update(
                fallback=bool(self.trace["rejected_backends"]),
                fallback_reason="; ".join(self.trace["rejected_backends"]) or None,
                reduction_order="pairwise_lower_upper_H; ascending_logical_token_S",
                kernel_fingerprint=f"adaln_modulation_v1:{self.trace['selected_backend']}",
            )
        if shared_unsupported:
            self.trace.update(
                fallback=True,
                fallback_reason=f"shared_{adaln_backend(self.implementation)}_domain_unsupported",
                kernel_fingerprint="adaln_modulation_v1:PYTORCH_ADALN_MODULATION",
            )
        if backend == "cuda":
            from rl_engine.backends.extension import _C

            self.trace["binary_path"] = getattr(_C, "__file__", None)
        return outputs
